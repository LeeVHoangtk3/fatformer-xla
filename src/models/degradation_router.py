"""Per-image degradation signals, fitted routing and calibrated fallback."""
import math
import warnings
import torch
from torch import nn
from torch.nn import functional as F
from .gating import DynamicFrequencyGating


def _images(x):
    if x.ndim not in (3, 4) or x.shape[-3] != 3 or not x.is_floating_point():
        raise ValueError('Expected floating RGB [B,3,H,W] or [3,H,W]')
    if min(x.shape[-2:]) <= 8:
        raise ValueError('Blockiness requires H,W > 8')
    if not torch.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError('Router images must be finite, unnormalized RGB in [0,1]')
    return x if x.ndim == 4 else x.unsqueeze(0)


def blockiness(x):
    batched = x.ndim == 4
    x = _images(x)
    g = .299*x[:, 0] + .587*x[:, 1] + .114*x[:, 2]
    dh = (g[:, :, 8::8] - g[:, :, 7:-1:8]).abs().mean((1, 2))
    dv = (g[:, 8::8, :] - g[:, 7:-1:8, :]).abs().mean((1, 2))
    nh = (g[:, :, 1:] - g[:, :, :-1]).abs().mean((1, 2))
    nv = (g[:, 1:, :] - g[:, :-1, :]).abs().mean((1, 2))
    score = (dh+dv)/(nh+nv+2e-6)-1
    return score if batched else score[0]


def laplacian_variance(x):
    batched = x.ndim == 4
    x = _images(x)
    g = (.299*x[:, 0]+.587*x[:, 1]+.114*x[:, 2]).unsqueeze(1)
    kernel = x.new_tensor([[0,1,0],[1,-4,1],[0,1,0]]).view(1,1,3,3)
    score = torch.log1p(F.conv2d(g, kernel, padding=1).var((1,2,3), unbiased=False))
    return score if batched else score[0]


def extract_lambda_bar(value):
    """Reduce actual gating output; never substitute a constant for missing features."""
    if not isinstance(value, torch.Tensor):
        raise ValueError('Pass actual gating output, not a model')
    if value.ndim == 3 and value.shape[0] == value.shape[2] == 1:
        value = value[0, :, 0]
    elif value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError('Gating output must be [1,B,1], [B,1] or [B]')
    return value


@torch.no_grad()
def forward_with_gating(model, normalized_x):
    """One eval forward; average adapter gates, never the batch axis."""
    if model.training:
        raise ValueError('Gating collection requires model.eval()')
    values, handles = [], []
    def capture(_module, _inputs, output):
        values.append(extract_lambda_bar(output).float())
    try:
        for module in model.modules():
            if isinstance(module, DynamicFrequencyGating):
                handles.append(module.register_forward_hook(capture))
        if not handles:
            raise ValueError('Checkpoint model has no DynamicFrequencyGating')
        logits = model(normalized_x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if len(values) != len(handles):
            raise ValueError('Not every adapter gate executed exactly once')
        return logits.float(), torch.stack(values).mean(0)
    finally:
        for handle in handles:
            handle.remove()


def _table(table):
    previous = 0.
    if not table:
        raise ValueError('A fitted threshold table is required')
    for row in table:
        lo, hi, threshold = (float(row[k]) for k in ('w_min','w_max','threshold'))
        if (not all(map(math.isfinite, (lo,hi,threshold))) or abs(lo-previous)>1e-10
                or not lo < hi <= 1 or not 0 <= threshold <= 1.000001):
            raise ValueError('Invalid/gapped threshold table')
        previous = hi
    if previous != 1:
        raise ValueError('Threshold table must cover [0,1]')


def threshold_for(w, threshold_table, default_thr=.5):
    _table(threshold_table)
    if not torch.isfinite(w).all() or ((w<0)|(w>1)).any():
        raise ValueError('Invalid routing weights')
    bounds = w.new_tensor([r['w_max'] for r in threshold_table[:-1]])
    thresholds = w.new_tensor([r['threshold'] for r in threshold_table])
    return thresholds[torch.bucketize(w.contiguous(), bounds, right=True)]


def fuse(p_clean, p_forensic, w, threshold_table, fallback_threshold=.5):
    if p_clean.shape != p_forensic.shape or w.shape != p_clean.shape:
        raise ValueError('Probabilities and weights must have identical per-image shapes')
    p = (1-w)*p_clean+w*p_forensic
    threshold = threshold_for(w, threshold_table)
    return p, (p>=threshold).long(), threshold


class DegradationRouter(nn.Module):
    def __init__(self, scaler_mean=None, scaler_std=None, coef=None, intercept=0.,
                 threshold_table=None, fallback_threshold=.5, fallback_mode=False,
                 fixed_branch='clean'):
        super().__init__()
        if fixed_branch not in ('clean','forensic') or not 0 <= fallback_threshold <= 1.000001:
            raise ValueError('Invalid calibrated fallback')
        for name, value, default in (('scaler_mean',scaler_mean,[0.]*3),
                                     ('scaler_std',scaler_std,[1.]*3), ('coef',coef,[0.]*3)):
            self.register_buffer(name, torch.tensor(value if value is not None else default, dtype=torch.float32))
            vector = getattr(self,name)
            if vector.shape != (3,) or not torch.isfinite(vector).all():
                raise ValueError('Expected three finite fitted parameters')
        self.register_buffer('intercept',torch.tensor(intercept,dtype=torch.float32))
        if (self.scaler_std<=0).any() or not torch.isfinite(self.intercept):
            raise ValueError('Invalid scaler/intercept')
        if threshold_table is not None:
            _table(threshold_table)
        self.threshold_table = threshold_table
        self.fallback_threshold = float(fallback_threshold)
        self.fixed_branch = fixed_branch
        self.fallback_mode = fallback_mode or coef is None or threshold_table is None

    @classmethod
    def from_params(cls, params):
        if (params.get('schema')!='degradation_router/v2' or params.get('evidence_level')!='measured'
                or params.get('status')!='fitted'
                or params.get('features') != ['blockiness','log1p_lapvar','lambda_bar']):
            raise ValueError('Require measured fitted v2 parameters; legacy synthetic artifacts rejected')
        return cls(params['scaler']['mean'],params['scaler']['std'],params['coef'],
                   params['intercept'],params['threshold_table'],params['fixed']['threshold'],
                   fixed_branch=params['fixed']['branch'])

    def get_extra_state(self):
        return dict(threshold_table=self.threshold_table,fallback_threshold=self.fallback_threshold,
                    fallback_mode=self.fallback_mode,fixed_branch=self.fixed_branch)

    def set_extra_state(self,state):
        if state['threshold_table'] is not None:
            _table(state['threshold_table'])
        for key,value in state.items():
            setattr(self,key,value)

    def extract_signals(self,x,lambda_bar=None):
        if lambda_bar is None:
            raise ValueError('Actual lambda_bar required; constant substitute forbidden')
        b,lap = blockiness(x),laplacian_variance(x)
        lam = extract_lambda_bar(lambda_bar).to(b)
        if lam.shape!=b.shape or not torch.isfinite(lam).all() or ((lam<0)|(lam>2)).any():
            raise ValueError('Invalid gating outputs')
        return torch.stack([b,lap,lam],-1)

    def predict_features(self,features):
        if (features.ndim!=2 or features.shape[1]!=3 or not torch.isfinite(features).all()
                or (features[:,1]<0).any() or ((features[:,2]<0)|(features[:,2]>2)).any()):
            raise ValueError('Expected finite measured features [B,3]')
        features = features.to(self.coef)
        return torch.sigmoid(((features-self.scaler_mean)/self.scaler_std)@self.coef+self.intercept)

    def predict_w(self,x,lambda_bar=None):
        return self.predict_features(self.extract_signals(x,lambda_bar))

    def route_features(self,features,p_clean,p_forensic):
        # Threshold fitting and inference share float32 precision, including ULP tie handling.
        p_clean, p_forensic = p_clean.float(), p_forensic.float()
        for p in (p_clean,p_forensic):
            if p.ndim!=1 or not torch.isfinite(p).all() or ((p<0)|(p>1)).any():
                raise ValueError('Expected finite calibrated probabilities [B]')
        if p_clean.shape!=p_forensic.shape:
            raise ValueError('Branch probability shapes differ')
        p = (p_clean if self.fixed_branch=='clean' else p_forensic).clone()
        threshold = torch.full_like(p,self.fallback_threshold)
        w = torch.full_like(p,float(self.fixed_branch=='forensic'))
        if not self.fallback_mode and features is not None:
            if features.shape!=(len(p),3):
                raise ValueError('Signals and probabilities have different batch sizes')
            valid = (torch.isfinite(features).all(-1) & (features[:,1]>=0)
                     & (features[:,2]>=0) & (features[:,2]<=2))
            if valid.any():
                w[valid] = self.predict_features(features[valid]).to(w)
                p[valid],_,threshold[valid] = fuse(p_clean[valid],p_forensic[valid],w[valid],self.threshold_table)
            if not valid.all():
                warnings.warn('[WARN] router disabled for invalid signals; calibrated fixed fallback',RuntimeWarning)
        elif not self.fallback_mode:
            warnings.warn('[WARN] router disabled: missing gating/signals; calibrated fixed fallback',RuntimeWarning)
        return p,(p>=threshold).long(),w,threshold

    def forward(self,x,p_clean,p_forensic,lambda_bar=None):
        features = None
        if not self.fallback_mode and lambda_bar is not None:
            if x.ndim!=4 or x.shape[:2]!=(len(p_clean),3):
                raise ValueError('Expected RGB batch matching probabilities')
            features = x.new_full((len(x),3),float('nan'))
            if min(x.shape[-2:])>8:
                lam = extract_lambda_bar(lambda_bar).to(x)
                if lam.shape!=(len(x),):
                    raise ValueError('Gating batch size mismatch')
                valid = (torch.isfinite(x).flatten(1).all(1) & (x>=0).flatten(1).all(1)
                         & (x<=1).flatten(1).all(1) & torch.isfinite(lam) & (lam>=0) & (lam<=2))
                if valid.any():
                    features[valid] = self.extract_signals(x[valid],lam[valid])
        return self.route_features(features,p_clean,p_forensic)
