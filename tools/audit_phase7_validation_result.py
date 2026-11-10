"""Verify submitted validation diagnostics, retain raw evidence, qualify claims."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import torch
from scipy.stats import binomtest

from tools import phase7_validation_diagnostics as diagnostic
from tools.audit_pilot_evaluation import copy_preserving,file_hashes,pinned_sources,same_tree
from tools.compare_pilot_checkpoints import paired_ci

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'DATASET/logs/phase7/improvement_round2/diagnostics'


def require(condition,message):
    if not condition:raise ValueError(message)


def csv_equal(left,right):
    with Path(left).open(encoding='utf-8',newline='') as stream:a=list(csv.DictReader(stream))
    with Path(right).open(encoding='utf-8',newline='') as stream:b=list(csv.DictReader(stream))
    require(len(a)==len(b),'CSV row count differs')
    for x,y in zip(a,b):
        require(set(x)==set(y),'CSV columns differ')
        for key in x:
            if x[key]==y[key]:continue
            try:match=math.isclose(float(x[key]),float(y[key]),rel_tol=0,abs_tol=1e-8)
            except (TypeError,ValueError):match=False
            require(match,'CSV value differs: '+key)
    return len(a)


def aligned(clean,q30):
    ids=clean['sample_ids'].tolist();other=q30['sample_ids'].tolist()
    require(len(ids)==len(set(ids)) and len(other)==len(set(other)) and set(ids)==set(other),'Unmatched clean/Q30 IDs')
    mapping={sid:i for i,sid in enumerate(other)};ix=np.array([mapping[sid] for sid in ids])
    result={k:v[ix] for k,v in q30.items() if isinstance(v,np.ndarray) and v.ndim and len(v)==len(other)}
    require(np.array_equal(clean['targets'],result['targets']),'Paired condition targets differ')
    return result


def interpretation(populations,plan):
    changes={};model_comparison={};distributions={};recalibrated={}
    for arm in plan['models']:
        clean=populations[arm,'clean'];q30=aligned(clean,populations[arm,'q30_post_crop']);y=clean['targets']
        per_group={}
        for generator in ['CIPS','Gansformer','stable_diffusion_v1_5','DiT','COCO','FFHQ','imagenet_replay']:
            mask=clean['generators']==generator;target=y[mask];label=int(target[0]);n=int(mask.sum())
            a=clean['labels'][mask]==target;b=q30['labels'][mask]==target
            lost=int((a&~b).sum());gained=int((~a&b).sum())
            intervals=[]
            for correct in [int(a.sum()),int(b.sum())]:
                ci=binomtest(correct,n).proportion_ci(confidence_level=.95,method='exact')
                intervals.append([100*ci.low,100*ci.high])
            per_group[generator]=dict(n=n,label=label,clean_correct=int(a.sum()),q30_correct=int(b.sum()),
                clean_recall_pct=100*float(a.mean()),q30_recall_pct=100*float(b.mean()),clean_correct_q30_wrong=lost,clean_wrong_q30_correct=gained,
                recall_exact_ci95_pct=intervals,exploratory_paired_exact_p=binomtest(lost,lost+gained,.5).pvalue if lost+gained else 1.,
                scope='Descriptive same-image validation; small groups and multiple comparisons; no representation-collapse or causal claim.')
        changes[arm]=dict(groups=per_group,paired_delta_clean_minus_q30_ci95_pp=paired_ci(y,clean['prob_fake'],q30['prob_fake'],n_boot=1000))
        distributions[arm]={}
        for family in ['real','commercial','latent_diffusion','pixel_diffusion']:
            p=clean['prob_fake'][clean['families']==family]
            distributions[arm][family]=dict(n=len(p),n_between_080_090=int(((p>=.8)&(p<=.9)).sum()),n_below_002=int((p<.02).sum()),quantiles_p10_p50_p90=np.quantile(p,[.1,.5,.9]).tolist())
    for condition in plan['protocol']['conditions']:
        a,b=(populations[arm,condition] for arm in plan['models'])
        require(np.array_equal(a['sample_ids'],b['sample_ids']),'Model comparison IDs differ')
        model_comparison[condition]=paired_ci(a['targets'],a['prob_fake'],b['prob_fake'],n_boot=1000)
    calibrations={
        'candidate_p0':ROOT/'DATASET/logs/phase7/improvement_round1/evaluation/runs/p0_eval_20261110_065901/raw/candidate_p0/calibration/candidate_p0.json',
        'incumbent':ROOT/'DATASET/logs/phase7/task_7.15/current/calibration/anchored_v2.json'}
    for arm,path in calibrations.items():
        calibration=diagnostic.read(path)
        require(calibration['checkpoint']['sha256']==plan['models'][arm]['sha256'],'Existing calibration belongs to another checkpoint')
        temperature=calibration['T'];threshold=calibration['T_star_default'];rows={}
        for condition in plan['protocol']['conditions']:
            data=populations[arm,condition];logits=data['logits'];y=data['targets']
            p32=torch.softmax(torch.tensor(logits).float()/temperature,-1)[:,1].numpy();pred=p32>=threshold
            margin=(logits[:,1].astype(float)-logits[:,0].astype(float))/temperature
            p64=np.exp(-np.logaddexp(0,-margin))
            rows[condition]=dict(real_recall_pct=100*float((~pred[y==0]).mean()),fake_recall_pct=100*float(pred[y==1].mean()),
                changed_labels_vs_historical_point=int((pred!=data['labels']).sum()),
                float32_vs_float64_label_disagreements=int((pred!=(p64>=threshold)).sum()),within_1e_minus6_of_threshold=int((np.abs(p32-threshold)<=1e-6).sum()))
        recalibrated[arm]=dict(calibration_file=path.relative_to(ROOT).as_posix(),sha256=diagnostic.sha(path),T=temperature,threshold=threshold,
            equivalent_raw_logit_margin=temperature*math.log(threshold/(1-threshold)),derived_cpu_diagnostics=rows,
            scope='Existing CFE calibration applied to stored validation logits using CPU float32; no fitting, no GPU revalidation, no new deployment artifact.')
    return dict(condition_changes=changes,model_delta_P0_minus_incumbent_ci95_pp=model_comparison,
        clean_probability_distributions=distributions,existing_checkpoint_calibration_diagnostic=recalibrated,
        bootstrap_scope='1000 paired resamples, seed42, stratified by label; one training seed; generator dependence and multiplicity not captured.',
        historical_reproduction_scope='Four aggregate recalls exactly reproduce; old per-image predictions/logits are unavailable, so bit-exact replay is not established.',
        causal_scope='Seen-generator val_select cannot establish why P0 underperformed on CFE or exclude backbone/representation limitations.')


def audit_and_import(source,notebook,base=BASE):
    source,notebook,base=(Path(p).resolve() for p in [source,notebook,base]);snapshot=file_hashes(source);nb_sha=diagnostic.sha(notebook)
    status=diagnostic.read(source/'status.json');identity=diagnostic.read(source/'identity.json');provenance=identity['source']
    require(status['status']=='completed' and status['training_started'] is False and status['deployment_replaced'] is False,'Diagnostic status not completed/nontraining')
    expected=status['files']
    require(set(snapshot)==set(expected)|{'status.json'},'Missing or extra raw files')
    for name,digest in expected.items():require(snapshot[name]==digest,'Submitted artifact hash differs: '+name)
    pinned=pinned_sources(ROOT,provenance['commit']);hashes={name:hashlib.sha256(data).hexdigest() for name,data in pinned.items()}
    require(hashes==provenance['files'],'Source proof differs from committed Git files')
    fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(fingerprint==provenance['tracked_files_fingerprint'],'Source fingerprint differs')
    plan=json.loads(pinned['configs/phase7_validation_diagnostics/plan.json'])
    require(diagnostic.read(source/'plan.json')==plan,'Exported plan semantics differ')
    require(hashes['configs/phase7_validation_diagnostics/plan.json']==identity['plan_sha256']==diagnostic.sha(diagnostic.PLAN),'Frozen plan hash differs')
    require(identity['manifest_sha256']==plan['manifest']['sha256'] and identity['validated_image_count']==1080,'Wrong diagnostic population')
    require(identity['precision']=='fp16' and identity['tf32'] is False,'Wrong numeric protocol')
    contract=diagnostic.read(source/'batch_contract.json')
    require(contract['same_for_both_models'] is True and contract['automatic_midrun_batch_change'] is False and contract['precision']=='fp16','Wrong common batch contract')
    run_ids=set(re.findall(r'val_diag_\d{8}_\d{6}',(source/'commands.log').read_text(encoding='utf-8')))
    require(len(run_ids)==1,'Missing/ambiguous diagnostic run ID');run_id=next(iter(run_ids))
    nb=diagnostic.read(notebook);cells=[c for c in nb['cells'] if c['cell_type']=='code']
    require([c.get('execution_count') for c in cells]==[1,2,3],'Notebook execution order differs')
    require(not any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[])),'Notebook contains errors')
    require(all(c.get('metadata',{}).get('executionInfo',{}).get('status')=='ok' for c in cells),'Notebook cells not completed')
    output='\n'.join(''.join(o.get('text',[])) for c in cells for o in c.get('outputs',[]))
    require(provenance['commit'] in output and run_id in output,'Notebook source/run differs from result')
    original_protection=diagnostic.read(BASE/'preparation/preservation.json')['files']
    for name,digest in original_protection.items():require(diagnostic.sha(ROOT/name)==digest,'Old evidence changed: '+name)
    target=base/'runs'/run_id;analysis=target/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='phase7_val_audit_') as temporary:
        recomputed=diagnostic.analyze(source,Path(temporary)/'recomputed')
        same_tree(recomputed,diagnostic.read(source/'analysis/diagnosis.json'))
        groups=csv_equal(Path(temporary)/'recomputed/group_metrics.csv',source/'analysis/group_metrics.csv')
        errors=csv_equal(Path(temporary)/'recomputed/errors.csv',source/'analysis/errors.csv')
        copy_preserving(Path(temporary)/'recomputed',analysis/'recomputed',file_hashes(Path(temporary)/'recomputed'))
    _,_,val=diagnostic.inputs();populations={}
    for arm in plan['models']:
        probe=diagnostic.read(source/arm/'probe.json');load=diagnostic.read(source/arm/'canonical_load.json')
        require(probe['status']=='completed' and probe['selected_batch_size']>=contract['batch_size'],'GPU probe/batch mismatch')
        for report in [load,probe['canonical_load']]:
            require(report['sha256']==plan['models'][arm]['sha256'] and report['missing']==report['unexpected']==0 and report['n_keys']==1200,'Checkpoint canonical chain differs')
        for condition in plan['protocol']['conditions']:
            populations[arm,condition]=diagnostic.load_predictions(source/arm/f'predictions_{condition}.npz',val,plan,arm,condition)
    scientific=interpretation(populations,plan)
    require(snapshot==file_hashes(source) and nb_sha==diagnostic.sha(notebook),'Submission changed during audit')
    copy_preserving(source,target/'raw',snapshot)
    copy_preserving(notebook.parent,target/'evidence',{notebook.name:nb_sha})
    verification=dict(schema='phase7-validation-diagnostic-audit/v1',passed=True,run_id=run_id,source_commit=provenance['commit'],source_files_checked=len(hashes),
        raw_files_checked=len(snapshot),raw_bytes=sum((source/name).stat().st_size for name in snapshot),prediction_arrays=4,prediction_records=4320,
        group_rows_recomputed=groups,error_rows_recomputed=errors,aggregate_and_csv_reports_reproduced=True,
        historical_aggregate_recall_differences=recomputed['historical_p0_comparison']['absolute_differences'],historical_logits_bit_exact_established=False,
        notebook_sha256=nb_sha,notebook_execution_counts=[1,2,3],notebook_wall_minutes=cells[-1]['metadata']['executionInfo']['elapsed']/60000,
        gpu_reported=identity['gpu'],batch_size=contract['batch_size'],local_model_forward_performed=False,
        scope='Local verification from committed code and submitted arrays; missing PTH and full image archives prevent independent model forward or image-label semantic review.')
    diagnostic.write(analysis/'verification.json',verification);diagnostic.write(analysis/'interpretation.json',scientific)
    diagnostic.write(target/'import_manifest.json',dict(source=source.relative_to(ROOT).as_posix(),raw_sha256=snapshot,notebook_source=notebook.relative_to(ROOT).as_posix(),
        notebook_sha256=nb_sha,old_evidence_files_unchanged=len(original_protection),weights_present=False))
    diagnostic.write(base/'index.json',dict(schema='phase7-validation-diagnostic-index/v1',current_run=run_id,path=target.relative_to(ROOT).as_posix(),
        status='completed_audited',training_started=False,deployment_replaced=False,retained_package_sha256='248690528f6e7ecadb06717fafbdb64ca5c101cb966ad3f06f8c00994c9245d0',
        next_step='Design controlled independent-calibration comparison first; targeted robustness is a separate hypothesis. No CFE-causal or bit-exact claim.'))
    return verification


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--notebook',required=True)
    args=parser.parse_args();print(json.dumps(audit_and_import(args.source,args.notebook),ensure_ascii=False,indent=2))
