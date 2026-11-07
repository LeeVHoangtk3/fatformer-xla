"""Regression fixtures are synthetic unit inputs, never scientific handoff artifacts."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.models.degradation_router import DegradationRouter, blockiness, forward_with_gating, fuse, threshold_for
from src.models.gating import DynamicFrequencyGating
from src.models.unified_fatformer import UnifiedFatFormerXLA, router_is_approved, validate_router_report
from tools.fit_router import fit_router_calibration, fit_table, fpr_threshold
from tools.generate_router_ablation import generate_router_ablation
from tools.router_data import FEATURES, PREPROCESS, SUBSETS, collect, load_cache, params_digest


@pytest.fixture(autouse=True)
def small_cpu_workload():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = DynamicFrequencyGating(d_model=3, hidden_dim=4)
        with torch.no_grad():
            self.gate.mlp[2].weight.fill_(.3)

    def forward(self, x):
        features = x.mean((2, 3)).unsqueeze(0)
        scale = self.gate(features)[0, :, 0]
        return torch.stack((-scale, scale), -1)


def cache_fixture(path, split="val_calib", offset=0, evidence="measured"):
    # Deliberately controlled fixtures under tmp_path, not measured model results.
    n = 120
    ids = np.asarray([f"{i+offset:064x}" for i in range(n)])
    condition = np.repeat([0, 1], n)
    labels = np.tile(np.arange(n) % 2, 2)
    t = np.tile(np.linspace(0, 1, n), 2)
    features = np.stack([t+condition, .1+t*.2-condition*.05, 1.5-condition*.9+t*.1], -1).astype(np.float32)
    score = (labels*2-1)*2+t
    logits = np.stack([-score, score], -1).astype(np.float32)
    source = {"manifest_sha256": "d"*64} if split == "val_calib" else {"evaluation_split_sha256": "e"*64}
    meta = dict(schema="router-data/v2", evidence_level=evidence, split=split, features=FEATURES,
                preprocess=PREPROCESS, checkpoint_sha256="a"*64, source=source,
                branch_design="single_logits_condition_temperature")
    np.savez_compressed(path, features=features, logits=logits, targets=labels, sample_ids=np.tile(ids, 2),
                        is_degraded=condition, subsets=np.tile(np.asarray(SUBSETS)[np.arange(n)//24], 2),
                        metadata=json.dumps(meta))
    return path


@pytest.fixture
def fitted(tmp_path, monkeypatch):
    # Fast temperature stub isolates router fitting; temperature integration is tested separately.
    from tools.calibrate_and_select import ModelWithTemperature
    monkeypatch.setattr(ModelWithTemperature, "fit", lambda *args: (1., 1.))
    cache = cache_fixture(tmp_path/"calib.npz")
    path = tmp_path/"params.json"
    return fit_router_calibration(cache, path), path


def complete_proof_fixture(params):
    """Structural activation fixture only: no claim these numbers were measured."""
    subsets = {}
    for i, subset in enumerate(SUBSETS):
        fixed, routed = 60., 62. if i < 3 else 60.
        router_ci = [61.5, 62.5] if i < 2 else [routed-2, routed+2]
        row = dict(n_source_images=24, n_real_source_images=12, n_fake_source_images=12, n_views=48,
                   delta_pp=routed-fixed, delta_ci95=[0., 4.] if i < 3 else [-2., 2.],
                   win=i < 3, router_ci_above_fixed=i < 2)
        for name, point, ci in (("fixed", fixed, [59., 61.]), ("router", routed, router_ci),
                                 ("oracle", 63., [62., 64.])):
            row[name] = dict(fake_at_real90=point, ci95=ci, by_condition={"clean": point, "q30": point})
        subsets[subset] = row
    return dict(schema="router_ablation/v2", status="evaluated", evidence_level="measured",
                params_digest=params_digest(params), checkpoint_sha256=params["source"]["checkpoint_sha256"],
                calibration_cache_sha256=params["source"]["cache_sha256"], params_file_sha256="b"*64,
                evaluation_cache_sha256="c"*64, evaluation_split_sha256="e"*64,
                preprocess=params["preprocess"], branch_design=params["branch_design"],
                bootstrap=dict(n=1000, seed=42, paired=True, unit="source image, both views retained",
                               strata="real_fake_source_images", confidence_level=.95),
                subsets=subsets, verdict=dict(criteria_met=True, fallback_required=False,
                                             wins=3, total_subsets=5, ci_separated=2))


def test_missing_gate_is_not_constant():
    with pytest.raises(ValueError, match="Actual lambda_bar"):
        DegradationRouter().extract_signals(torch.rand(2,3,16,16))


@pytest.mark.parametrize("temperatures,weights,preserves_order", [
    ((1., 10.), (.5, .5), True),
    ((1., 10.), (.99, .01), False),
    ((1., 1.), (.99, .01), True),
])
def test_shared_logits_ranking_depends_on_routing_weights(temperatures, weights, preserves_order):
    """A mathematical counterexample, not measured evidence of a project-level gain."""
    logits = torch.tensor([[0., 2.], [0., 1.]], dtype=torch.float64)
    pc = torch.softmax(logits / temperatures[0], -1)[:, 1]
    pf = torch.softmax(logits / temperatures[1], -1)[:, 1]
    assert pc[0] > pc[1] and pf[0] > pf[1]
    routed, _, _ = fuse(pc, pf, torch.tensor(weights, dtype=pc.dtype),
                         [dict(w_min=0., w_max=1., threshold=.5)])
    assert bool(routed[0] > routed[1]) == preserves_order


def test_hook_uses_real_gate_and_removes_handles():
    model = ToyBackbone().eval()
    x = torch.stack([torch.zeros(3,16,16), torch.ones(3,16,16)])
    logits, lam = forward_with_gating(model, x)
    assert logits.shape == (2,2)
    assert lam.shape == (2,) and lam[0] != lam[1]
    assert not model.gate._forward_hooks
    for i in range(2):
        _, single = forward_with_gating(model,x[i:i+1])
        torch.testing.assert_close(single,lam[i:i+1])
    with pytest.raises(ValueError,match="no Dynamic"):
        forward_with_gating(nn.Identity().eval(),x)


def test_hook_cleanup_on_forward_failure():
    class Broken(ToyBackbone):
        def forward(self,x):
            super().forward(x)
            raise RuntimeError("broken")
    model = Broken().eval()
    with pytest.raises(RuntimeError):
        forward_with_gating(model,torch.rand(1,3,16,16))
    assert not model.gate._forward_hooks


def test_threshold_ties_and_saturated_scores():
    y = np.zeros(100,dtype=int)
    for scores in (np.ones(100,dtype=np.float32),np.linspace(0,1,100,dtype=np.float32)):
        thr = fpr_threshold(scores,y)
        assert (scores >= thr).mean() <= .05
        assert (torch.from_numpy(scores) >= thr).float().mean() <= .05


def test_merge_sparse_bins_and_measure_fpr():
    w = np.linspace(0,1,120,dtype=np.float32)
    p = w.copy()
    y = np.arange(120)%2
    table, merges = fit_table(w,p,y)
    assert len(table)==1 and merges
    assert table[0]["n_real"]==60 and table[0]["fpr_measured"] <= .05
    with pytest.raises(ValueError,match="50 Real"):
        fit_table(w[:20],p[:20],y[:20])


def test_cache_rejects_wrong_split_and_synthetic(tmp_path):
    path = cache_fixture(tmp_path/"cache.npz",evidence="synthetic")
    with pytest.raises(ValueError,match="measured"):
        load_cache(path,"val_calib")
    cache_fixture(path,split="evaluation")
    with pytest.raises(ValueError,match="split"):
        fit_router_calibration(path,tmp_path/"not_written.json")
    assert not (tmp_path/"not_written.json").exists()


def test_cache_rejects_missing_partition_fingerprint_before_bootstrap(tmp_path, fitted):
    _, params_path = fitted
    path = cache_fixture(tmp_path/"incomplete_eval.npz", split="evaluation", offset=1000)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays["metadata"]))
    metadata["source"] = {}
    arrays["metadata"] = json.dumps(metadata)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="partition fingerprint"):
        generate_router_ablation(path, params_path, tmp_path/"not_written")
    assert not (tmp_path/"not_written").exists()


def test_fit_provenance_and_fpr(fitted):
    params, _ = fitted
    assert params["n_calib"]==120 and params["n_views"]==240
    assert params["cv"]["group"]=="source_image_sha256"
    assert not params["deployment_enabled"]
    assert all(row["fpr_measured"] <= .05 for row in params["threshold_table"])
    assert params["calibration"][params["fixed"]["branch"]]["calibration_fpr"] <= .05
    assert len(params["w_histogram"]["clean"])==20


def test_ablation_real_resampling_and_no_automatic_win(tmp_path,fitted):
    params,path = fitted
    data = cache_fixture(tmp_path/"eval.npz",split="evaluation",offset=1000)
    report = generate_router_ablation(data,path,tmp_path/"results")
    assert report["bootstrap"]["n"]==1000
    # Identical calibrated scores give zero gain, regardless of high raw model accuracy.
    assert not report["verdict"]["criteria_met"] and report["verdict"]["fallback_required"]
    assert all(abs(row["delta_pp"]) < 1e-10 for row in report["subsets"].values())
    assert all(row["delta_ci95"]==[0.,0.] for row in report["subsets"].values())
    assert report["params_digest"]==params_digest(params)
    assert validate_router_report(params, report) == report["verdict"]
    assert not router_is_approved(params, report)


def test_ablation_rejects_overlap_and_insufficient_bootstrap(tmp_path,fitted):
    _,path = fitted
    data = cache_fixture(tmp_path/"overlap.npz",split="evaluation")
    with pytest.raises(ValueError,match="overlap"):
        generate_router_ablation(data,path,tmp_path/"results")
    with pytest.raises(ValueError,match="1000"):
        generate_router_ablation(data,path,tmp_path/"results",n_boot=5)


def test_invalid_signals_fallback_per_image(fitted):
    params,_ = fitted
    router = DegradationRouter.from_params(params)
    x = torch.rand(2,3,16,16); x[1,0,0,0]=float("nan")
    pc,pf,lam = torch.tensor([.3,.7]),torch.tensor([.8,.2]),torch.tensor([.4,.8])
    alone = router(x[:1],pc[:1],pf[:1],lam[:1])
    with pytest.warns(RuntimeWarning,match="invalid signals"):
        batch = router(x,pc,pf,lam)
    for a,b in zip(alone,batch):
        torch.testing.assert_close(a,b[:1])
    assert torch.isfinite(batch[0]).all()
    with pytest.warns(RuntimeWarning):
        tiny = router(torch.rand(2,3,8,8),pc,pf,lam)
    assert torch.isfinite(tiny[0]).all()
    with pytest.raises(ValueError):
        blockiness(torch.rand(1,3,8,8))


def test_router_state_roundtrip(fitted):
    params,_ = fitted
    router = DegradationRouter.from_params(params)
    clone = DegradationRouter()
    clone.load_state_dict(router.state_dict())
    assert clone.get_extra_state()==router.get_extra_state()
    x,lam=torch.rand(2,3,16,16),torch.tensor([.3,1.6])
    torch.testing.assert_close(router.predict_w(x,lam),clone.predict_w(x,lam))


def test_wrapper_fallback_calibration_binding_and_state(fitted):
    params,_=fitted
    with pytest.warns(RuntimeWarning,match="router disabled"):
        wrapper=UnifiedFatFormerXLA(ToyBackbone(),router_params=params,checkpoint_sha256="a"*64).eval()
    assert not wrapper.router_enabled
    x=torch.rand(2,3,16,16)
    result=wrapper(x)
    assert torch.isfinite(result["prob_fake"]).all()
    assert (result["threshold"]==params["fixed"]["threshold"]).all()
    with pytest.raises(ValueError,match="another checkpoint"):
        UnifiedFatFormerXLA(ToyBackbone(),router_params=params,checkpoint_sha256="b"*64)
    assert "_extra_state" in wrapper.state_dict()


def test_wrapper_requires_matching_win_proof(fitted):
    params,_=fitted
    proof=complete_proof_fixture(params)
    wrapper=UnifiedFatFormerXLA(ToyBackbone(),router_params=params,router_ablation=proof,checkpoint_sha256="a"*64).eval()
    assert wrapper.router_enabled
    x=torch.rand(2,3,16,16)
    whole=wrapper(x)
    for i in range(2):
        single=wrapper(x[i:i+1])
        for key in whole:
            torch.testing.assert_close(whole[key][i:i+1],single[key],atol=1e-6,rtol=1e-6)
    proof["params_digest"]="changed"
    with pytest.warns(RuntimeWarning):
        rejected=UnifiedFatFormerXLA(ToyBackbone(),router_params=params,router_ablation=proof,checkpoint_sha256="a"*64)
    assert not rejected.router_enabled


@pytest.mark.parametrize("path,value", [
    (("status",), "pending"),
    (("evidence_level",), "synthetic"),
    (("subsets",), {}),
    (("subsets", SUBSETS[0]), None),
    (("subsets", "unexpected"), {}),
    (("evaluation_cache_sha256",), None),
    (("evaluation_split_sha256",), "not-a-hash"),
    (("params_file_sha256",), "x"*64),
    (("calibration_cache_sha256",), "b"*64),
    (("checkpoint_sha256",), "b"*64),
    (("preprocess",), "another-pipeline"),
    (("branch_design",), "another-model"),
    (("bootstrap", "n"), "1000"),
    (("bootstrap", "n"), 999),
    (("bootstrap", "seed"), True),
    (("bootstrap", "paired"), False),
    (("bootstrap", "confidence_level"), .90),
    (("bootstrap", "strata"), "individual-views"),
    (("verdict", "wins"), 5),
    (("verdict", "total_subsets"), 4),
    (("verdict", "ci_separated"), 5),
    (("verdict", "criteria_met"), False),
    (("verdict", "fallback_required"), True),
    (("subsets", SUBSETS[0], "delta_pp"), 20.),
    (("subsets", SUBSETS[0], "win"), False),
    (("subsets", SUBSETS[0], "router_ci_above_fixed"), False),
    (("subsets", SUBSETS[0], "router", "fake_at_real90"), float("nan")),
    (("subsets", SUBSETS[0], "router", "ci95"), [70., 60.]),
    (("subsets", SUBSETS[0], "oracle", "ci95"), [0., float("inf")]),
    (("subsets", SUBSETS[0], "router", "by_condition"), {"clean": 62.}),
    (("subsets", SUBSETS[0], "delta_ci95"), [-101., 1.]),
    (("subsets", SUBSETS[0], "n_views"), 47),
    (("subsets", SUBSETS[0], "n_real_source_images"), 0),
])
def test_malformed_or_inconsistent_proof_keeps_wrapper_on_fallback(fitted, path, value):
    params, _ = fitted
    proof = complete_proof_fixture(params)
    target = proof
    for key in path[:-1]:
        target = target[key]
    if value is None:
        target.pop(path[-1])
    else:
        target[path[-1]] = value
    assert not router_is_approved(params, proof)
    with pytest.warns(RuntimeWarning, match="router disabled"):
        wrapper = UnifiedFatFormerXLA(ToyBackbone(), router_params=params, router_ablation=proof,
                                     checkpoint_sha256="a"*64).eval()
    assert not wrapper.router_enabled and wrapper.router.fallback_mode
    output = wrapper(torch.rand(2, 3, 16, 16))
    assert torch.isfinite(output["prob_fake"]).all()
    torch.testing.assert_close(output["threshold"], torch.full((2,), params["fixed"]["threshold"]))


def test_summary_only_proof_is_rejected(fitted):
    params, _ = fitted
    full = complete_proof_fixture(params)
    summary = {key: copy.deepcopy(full[key]) for key in
               ("schema", "evidence_level", "params_digest", "checkpoint_sha256", "bootstrap", "verdict")}
    assert not router_is_approved(params, summary)


def test_consistent_losing_report_does_not_enable_router(fitted):
    params, _ = fitted
    report = complete_proof_fixture(params)
    # Three gains remain, but no separated intervals: the second criterion is mandatory.
    for row in report["subsets"].values():
        row["router"]["ci95"] = [55., 65.]
        row["router_ci_above_fixed"] = False
    report["verdict"].update(ci_separated=0, criteria_met=False, fallback_required=True)
    assert validate_router_report(params, report) == report["verdict"]
    assert not router_is_approved(params, report)


def test_legacy_artifacts_and_unfitted_defaults_rejected():
    with pytest.raises(ValueError,match="legacy"):
        DegradationRouter.from_params({"schema":"degradation_router/v1","evidence_level":"measured"})
    assert DegradationRouter().fallback_mode
    with pytest.raises(ValueError):
        threshold_for(torch.tensor([.5]),[{"w_min":.1,"w_max":1.,"threshold":.5}])


def test_collector_reads_images_hashes_and_actual_hooks(tmp_path,monkeypatch):
    from PIL import Image
    from tools import eval_communityforensics as evaluator
    from src.training.runtime_state import sha256_file
    records=[]
    for i in range(4):
        path=tmp_path/f"image{i}.png"
        Image.new("RGB",(240,256),color=(i*60,20+i*10,40)).save(path)
        records.append(dict(path=str(path),sha256=sha256_file(path),split="val_calib",label=i%2,family="fixture"))
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(records=records)),encoding="utf-8")
    model=ToyBackbone().eval()
    monkeypatch.setattr(evaluator,"build_cfe_model_v3",lambda *args: (model,{"sha256":"a"*64}))
    args=SimpleNamespace(manifest=str(manifest),cfe_dir=None,evaluation_split=None,device="cpu",
                         checkpoint="fixture",clip_path="fixture",batch_size=2,output=str(tmp_path/"cache.npz"))
    collect(args)
    data,meta=load_cache(args.output,"val_calib")
    assert data["features"].shape==(8,3)
    assert len(set(data["sample_ids"].tolist()))==4
    assert meta["source"]["manifest_sha256"]==sha256_file(manifest)
    assert np.ptp(data["features"][:,2])>0
    assert not model.gate._forward_hooks
    # No gray placeholder image or partial scientific result if a source disappears.
    records[0]["path"]=str(tmp_path/"missing.png")
    manifest.write_text(json.dumps(dict(records=records)),encoding="utf-8")
    args.output=str(tmp_path/"missing_cache.npz")
    with pytest.raises(FileNotFoundError):
        collect(args)
    assert not (tmp_path/"missing_cache.npz").exists()


def test_temperature_fit_uses_labels_and_finite_logits():
    from tools.calibrate_and_select import ModelWithTemperature
    logits=torch.tensor([[0.,4.],[4.,0.],[0.,3.],[3.,0.]])
    # Deliberately overconfident incorrect logits should be softened by temperature.
    temperature,_=ModelWithTemperature().fit(logits,torch.tensor([0,1,0,1]))
    assert np.isfinite(temperature) and temperature>1


def test_evaluation_helper_uses_fixed_fallback_and_requires_calibration():
    from tools.eval_communityforensics import compute_unified_router_predictions_v3
    pc=torch.tensor([[.9,.1],[.1,.9]])
    pf=1-pc
    labels=torch.tensor([0,1])
    with pytest.raises(ValueError,match="calibrated"):
        compute_unified_router_predictions_v3(pc,pf,torch.tensor([50.,0.]),labels)
    with pytest.warns(RuntimeWarning):
        metrics,probabilities=compute_unified_router_predictions_v3(pc,pf,torch.tensor([50.,0.]),labels,fixed_threshold=.95)
    torch.testing.assert_close(probabilities,pc)
    assert metrics["real_acc"]==100 and metrics["fake_acc"]==0


def test_wrapper_without_router_requires_matching_calibration():
    calibration=dict(schema="calibration/v1",evidence_level="measured",checkpoint={"sha256":"a"*64},
                     T=1.4,T_star={"0.05":.93})
    with pytest.warns(RuntimeWarning):
        wrapper=UnifiedFatFormerXLA(ToyBackbone(),calibration=calibration,checkpoint_sha256="a"*64).eval()
    result=wrapper(torch.rand(2,3,16,16))
    assert torch.isfinite(result["prob_fake"]).all()
    torch.testing.assert_close(result["threshold"],torch.full((2,),.93))
    with pytest.raises(ValueError,match="fallback calibration"):
        UnifiedFatFormerXLA(ToyBackbone(),calibration=calibration,checkpoint_sha256="b"*64)
