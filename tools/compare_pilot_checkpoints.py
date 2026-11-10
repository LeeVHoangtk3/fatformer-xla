"""Matched prediction comparison; never uses test scores to choose a new training arm."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from tools.calibration_data import sha256_file
from tools.phase7_corrected import CONDITIONS,write_json


def paired_arrays(candidate,reference):
    for field in ['sample_ids','targets','subsets']:
        if not np.array_equal(candidate[field],reference[field]):raise ValueError('Matched IDs/labels/subsets differ: '+field)
    y=np.asarray(candidate['targets']);a=np.asarray(candidate['prob_fake'],dtype=np.float64);b=np.asarray(reference['prob_fake'],dtype=np.float64)
    if y.ndim!=1 or a.shape!=y.shape or b.shape!=y.shape or set(y.tolist())!={0,1}:
        raise ValueError('Invalid matched prediction shapes/classes')
    if len(set(candidate['sample_ids'].tolist()))!=len(y):raise ValueError('Duplicate prediction IDs')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a<0)|(a>1)|(b<0)|(b>1)):
        raise ValueError('Invalid probabilities')
    return y,a,b


def discriminative(scores,targets):
    return dict(auc=100*float(roc_auc_score(targets,scores)),
                fake_at_real90=100*float((scores[targets==1]>=np.quantile(scores[targets==0],.90)).mean()))


def paired_ci(targets,a,b,n_boot=1000,seed=42):
    if n_boot<50:raise ValueError('At least 50 resamples required')
    groups=[np.flatnonzero(targets==label) for label in [0,1]]
    if not all(len(g) for g in groups):raise ValueError('Need both labels for paired bootstrap')
    rng=np.random.default_rng(seed);draws={k:[] for k in ['auc','fake_at_real90']}
    for _ in range(n_boot):
        ix=np.concatenate([rng.choice(group,len(group),replace=True) for group in groups])
        x,y=discriminative(a[ix],targets[ix]),discriminative(b[ix],targets[ix])
        for key in draws:draws[key].append(x[key]-y[key])
    return {key:np.quantile(value,[.025,.975]).tolist() for key,value in draws.items()}


def compare(run_root,output,n_boot=1000):
    run_root=Path(run_root);arms=['candidate_p0','incumbent_refit'];reports={};predictions={}
    plan=json.loads((Path(__file__).resolve().parents[1]/'configs/phase7_pilot_eval/plan.json').read_text())
    for arm in arms:
        folder=run_root/arm/'benchmark';report=json.loads((folder/'final_report.json').read_text())
        if report.get('status')!='completed' or report['source_checkpoint_sha256']!=plan['models'][arm]['sha256']:
            raise ValueError('Missing completed report for frozen checkpoint')
        for name,expected in report['artifacts'].items():
            if sha256_file(folder/name)!=expected:raise ValueError('Changed benchmark artifact: '+name)
        reports[arm]=report
        for name in [*['cfe_'+c for c in CONDITIONS],'indist']:
            with np.load(folder/('predictions_'+name+'.npz'),allow_pickle=False) as payload:
                if str(payload['package_sha256'])!=report['package_sha256']:raise ValueError('Prediction package hash differs')
                predictions[arm,name]={k:payload[k].copy() for k in ['sample_ids','targets','subsets','prob_fake','labels']}
    left,right=(reports[a] for a in arms)
    for key in ['environment','preprocess','precision_contract','bootstrap']:
        if left[key]!=right[key]:raise ValueError('Comparison protocol differs: '+key)
    for key in ['inference_precision','extraction_batch_size','probability_device','probability_batch_size','partition']:
        if left['calibration'][key]!=right['calibration'][key]:raise ValueError('Calibration protocol differs: '+key)
    cfe={}
    for condition in CONDITIONS:
        p,q=(predictions[arm,'cfe_'+condition] for arm in arms);y,a,b=paired_arrays(p,q)
        if len(y)!=1600 or int((y==0).sum())!=800:raise ValueError('Require frozen 1600 balanced CFE predictions')
        first,second=discriminative(a,y),discriminative(b,y)
        good_a=p['labels']==y;good_b=q['labels']==y
        cfe[condition]=dict(candidate=first,reference=second,
            candidate_minus_reference_pp={key:first[key]-second[key] for key in first},
            paired_ci95_pp=paired_ci(y,a,b,n_boot=n_boot),
            correctness=dict(both_correct=int((good_a&good_b).sum()),candidate_only_correct=int((good_a&~good_b).sum()),
                reference_only_correct=int((~good_a&good_b).sum()),both_wrong=int((~good_a&~good_b).sum())))
    y,a,b=paired_arrays(predictions[arms[0],'indist'],predictions[arms[1],'indist'])
    if len(y)!=82003:raise ValueError('Require all 82003 in-dist records')
    a_groups,b_groups=left['indist']['per_subset'],right['indist']['per_subset']
    if set(a_groups)!=set(b_groups) or len(a_groups)!=18:raise ValueError('Matched 18 subsets required')
    subset_delta=np.array([a_groups[k]['balanced_acc']-b_groups[k]['balanced_acc'] for k in sorted(a_groups)])
    rng=np.random.default_rng(42)
    bootstrap=subset_delta[rng.integers(0,18,size=(10000,18))].mean(axis=1)
    result=dict(schema='phase7-p0-matched-comparison/v1',status='completed',
        source_checkpoint_hashes={arm:r['source_checkpoint_sha256'] for arm,r in reports.items()},
        package_hashes={arm:r['package_sha256'] for arm,r in reports.items()},
        reference_scope=plan['reference_scope'],scientific_status=plan['scientific_status'],
        CFE=cfe,paired_bootstrap=dict(n=n_boot,seed=42,stratified_by_label=True,unit='image',
            note='Same indices for both frozen models; conditional on one training seed. No multiple-comparison correction; generator dependence is not captured.'),
        indist=dict(candidate_macro=left['indist']['macro_acc'],reference_macro=right['indist']['macro_acc'],
            macro_delta_pp=float(subset_delta.mean()),diagnostic_paired_subset_ci95_pp=np.quantile(bootstrap,[.025,.975]).tolist(),
            diagnostic_unit='18 datasets; shared real reference dependence remains; this is not a replacement gate rule',
            pooled_real_delta_pp=left['indist']['pooled_real_acc']-right['indist']['pooled_real_acc']),
        G72={arm:r['gates']['G7.2'] for arm,r in reports.items()},
        G73=dict(status='pending',matched_incumbent_comparison='completed',best_by_condition_reference='not_established'),
        training_selection='P0 was frozen by val_select before this run; no model/epoch/threshold selected from evaluation.',
        deployment_replaced=False)
    write_json(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--n_boot',type=int,default=1000)
    args=parser.parse_args();compare(**vars(args))
