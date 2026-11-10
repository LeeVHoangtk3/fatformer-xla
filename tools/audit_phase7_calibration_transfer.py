"""Audit submitted calibration transfer from saved GPU scores, without deploying."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from tools import phase7_calibration_transfer as transfer
from tools import phase7_validation_diagnostics as diagnostic
from tools.audit_pilot_evaluation import copy_preserving,file_hashes,pinned_sources,same_tree,temperature_numpy
from tools.audit_phase7_validation_result import csv_equal

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'DATASET/logs/phase7/improvement_round2/calibration_transfer'
require=transfer.require


def real_guard(summary,minimum):
    return {tag:all(rows[tag]['real_recall_pct']>=100*minimum for rows in summary.values()) for tag in ['cfe_control','v2_clean_fit']}


def check_scores(data,raw,point,arm,condition,plan,batch):
    for key in ['targets','sample_ids','families','generators','paths']:
        require(np.array_equal(data[key],raw[key]),'Scored array identity/order differs: '+key)
    for key,expected in [('checkpoint_sha256',plan['models'][arm]['sha256']),('plan_sha256',transfer.sha(transfer.PLAN)),
                         ('point_sha256',point['source_sha256']),('split','val_select'),('condition',condition),('precision','fp16'),('probability_device','cuda')]:
        require(str(data[key])==expected,'Scored array provenance differs: '+key)
    require(int(data['batch_size'])==batch and float(data['T'])==point['T'] and float(data['threshold'])==point['threshold'],'Scored point/batch differs')
    p,y,labels=data['prob_fake'],data['targets'],data['labels']
    require(p.shape==y.shape==labels.shape and np.isfinite(p).all() and np.all((p>=0)&(p<=1)),'Invalid scored probabilities')
    require(np.array_equal(labels,(p>=np.float32(point['threshold'])).astype(int)),'Saved GPU labels inconsistent with saved scores')
    cpu=transfer.calibration_probabilities(torch.tensor(raw['logits']),point['T'],'cpu',batch)
    error=float(np.max(np.abs(cpu-p)));require(error<=1e-6,'GPU scores differ from raw logits beyond numeric tolerance')
    return dict(cpu_max_probability_error=error,cpu_label_disagreements=int(((cpu>=np.float32(point['threshold']))!=labels).sum()))


def read_npz(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key].copy() for key in z.files}


def recompute_model(source,arm,plan,calibration_records,validation_records,batch,output):
    folder=source/arm;original=transfer.read(folder/'comparison/comparison.json');result=copy.deepcopy(original)
    require(original['schema']=='phase7-calibration-transfer-comparison/v1' and original['model']==arm,'Wrong comparison schema/model')
    require(original['experimental_point_only'] is True and not original['training_started'] and not original['deployment_replaced'],'Comparison attempts production promotion')
    require(original['calibration_fit_split']=='val_calib' and original['validation_split']=='val_select' and original['G72']=='unchanged' and original['G73']=='unchanged pending','Comparison split/gate scope differs')
    same_tree(original['limits'],plan['limits'])
    calibration=read_npz(folder/'calibration_logits.npz')
    transfer.validate_logits(calibration,calibration_records,'val_calib','clean',arm,plan,batch)
    require(str(calibration['origin'])=='new_forward','Calibration origin must be the independent new forward')
    fitted=transfer.read(folder/'comparison/v2_calibration.json')
    require(fitted['schema']=='phase7-calibration-transfer-fit/v1' and fitted['deployable'] is False and fitted['split']=='val_calib' and fitted['condition']=='clean','Wrong/nonresearch fit')
    require(fitted['checkpoint_sha256']==plan['models'][arm]['sha256'] and fitted['plan_sha256']==transfer.sha(transfer.PLAN),'Fit provenance differs')
    require(fitted['manifest_sha256']==plan['manifest']['sha256'] and fitted['batch_size']==batch and fitted['probability_device']=='cuda' and fitted['probability_dtype']=='float32' and fitted['comparison_operator']=='>=','Fit numeric/manifest contract differs')
    expected_ids=hashlib.sha256('\n'.join(calibration['sample_ids']).encode()).hexdigest()
    require(fitted['calibration_ids_sha256']==expected_ids,'Fit IDs differ')
    cpu_fit=transfer.fit_point(calibration,calibration_records,arm,plan,batch,'cpu')
    independent=temperature_numpy(calibration['logits'],calibration['targets'])
    require(abs(fitted['T']-cpu_fit['T'])<5e-6 and abs(fitted['T']-independent['T_float32'])<5e-6,'Independent temperature optimum differs')
    require(abs(fitted['threshold']-cpu_fit['threshold'])<=1e-6,'Calibration threshold differs beyond float32 tolerance')
    require(fitted['temperature_optimization']['converged'] is True,'Unconverged temperature')
    probability=transfer.calibration_probabilities(torch.tensor(calibration['logits']),fitted['T'],'cpu',batch)
    y=calibration['targets'];fp=int((probability[y==0]>=np.float32(fitted['threshold'])).sum())
    require(fitted['n']==1080 and fitted['n_real']==590 and fitted['n_fake']==490 and fitted['false_positive_budget']==29,'Fit counts/budget differ')
    require(fitted['false_positives']<=29 and fitted['empirical_fpr']==fitted['false_positives']/590,'Declared fit FPR invalid')
    require(fp==fitted['false_positives'],'CPU replay at GPU threshold does not reproduce calibration FP count; inspect ULP boundary')
    controls=transfer.read(ROOT/plan['controls'][arm]['file']);points=original['points']
    same_tree(points['cfe_control'],dict(T=controls['T'],threshold=controls['T_star_default'],source_sha256=plan['controls'][arm]['sha256']))
    same_tree(points['v2_clean_fit'],dict(T=fitted['T'],threshold=fitted['threshold'],source_sha256=transfer.sha(folder/'comparison/v2_calibration.json')))
    groups=[];errors=[];numerics={};summary={};ranks={};paired={};cache_exact=True
    for condition in plan['protocol']['conditions']:
        raw=read_npz(folder/f'validation_logits_{condition}.npz')
        transfer.validate_logits(raw,validation_records,'val_select',condition,arm,plan,batch)
        require(str(raw['origin'])=='frozen_diagnostic_logits','This submitted run must use frozen diagnostic logits')
        reference=read_npz(source/'reference_validation'/arm/f'predictions_{condition}.npz')
        require(raw['logits'].dtype==reference['logits'].dtype and raw['logits'].tobytes()==reference['logits'].tobytes(),'Cached validation logits are not bit-identical')
        require(np.array_equal(raw['sample_ids'],reference['sample_ids']) and np.array_equal(raw['targets'],reference['targets']),'Cache identities differ')
        margin=raw['logits'][:,1].astype(float)-raw['logits'][:,0].astype(float);y=raw['targets']
        ranks[condition]=dict(raw_margin_auc_pct=100*transfer.roc_auc_score(y,margin),raw_margin_ap_pct=100*transfer.average_precision_score(y,margin),scope=original['rank_from_frozen_logits'][condition]['scope'])
        values={};summary[condition]={}
        for tag,point in points.items():
            data=read_npz(folder/'comparison'/f'{tag}_{condition}.npz');values[tag]=data
            numerics[condition,tag]=check_scores(data,raw,point,arm,condition,plan,batch)
            p,labels=data['prob_fake'],data['labels'];summary[condition][tag]=transfer.metrics(p,y,labels)
            for field in ['families','generators']:
                for group in sorted(set(data[field])):
                    mask=data[field]==group;v=diagnostic.summarize_scores(p[mask],y[mask],labels[mask])
                    groups.append(dict(model=arm,condition=condition,point=tag,group_type=field,group=group,n=v['n'],real_recall_pct=v['real_recall_pct'],fake_recall_pct=v['fake_recall_pct'],p10=v['probability_quantiles'][0],p50=v['probability_quantiles'][1],p90=v['probability_quantiles'][2]))
            for i in np.flatnonzero(labels!=y):
                errors.append(dict(model=arm,condition=condition,point=tag,sample_id=str(data['sample_ids'][i]),path=str(data['paths'][i]),target=int(y[i]),label=int(labels[i]),prob_fake=float(p[i])))
        a,b=summary[condition]['v2_clean_fit'],summary[condition]['cfe_control'];new,old=values['v2_clean_fit'],values['cfe_control']
        paired[condition]=dict(new_minus_control=dict(real_recall_pp=a['real_recall_pct']-b['real_recall_pct'],fake_recall_pp=a['fake_recall_pct']-b['fake_recall_pct'],nll=a['nll']-b['nll']),
            ci95=transfer.paired_operating_ci(y,new['prob_fake'],old['prob_fake'],new['labels'],old['labels'],plan['protocol']['bootstrap']['n'],plan['protocol']['bootstrap']['seed']),
            labels_changed=int((new['labels']!=old['labels']).sum()))
    result.update(validation=summary,rank_from_frozen_logits=ranks,paired=paired,both_conditions_real_guard=real_guard(summary,plan['protocol']['diagnostic_real_guard']))
    same_tree(result,original)
    output.mkdir(parents=True,exist_ok=True);transfer.write(output/'comparison.json',result)
    diagnostic.csv_write(output/'group_metrics.csv',groups)
    if errors:diagnostic.csv_write(output/'errors.csv',errors)
    diagnostic.csv_write(output/'summary.csv',[dict(model=arm,condition=c,point=tag,real_recall_pct=v['real_recall_pct'],fake_recall_pct=v['fake_recall_pct'],fpr_pct=v['fpr_pct'],accuracy_pct=v['accuracy_pct'],nll=v['nll'],confidence_ece_argmax_pct=v['confidence_ece_argmax_pct'],score_auc_pct=v['auc_pct'],raw_margin_auc_pct=ranks[c]['raw_margin_auc_pct']) for c,rows in summary.items() for tag,v in rows.items()])
    diagnostic.csv_write(output/'paired_deltas.csv',[dict(condition=c,metric=k,delta=v,ci95_low=paired[c]['ci95'][k][0],ci95_high=paired[c]['ci95'][k][1]) for c,row in paired.items() for k,v in row['new_minus_control'].items()])
    for name in ['group_metrics.csv','errors.csv','summary.csv','paired_deltas.csv']:csv_equal(output/name,folder/'comparison'/name)
    return result,dict(independent_temperature=independent,cpu_refit_T=cpu_fit['T'],gpu_threshold=fitted['threshold'],cpu_refit_threshold=cpu_fit['threshold'],calibration_fp_reproduced=fp,
        validation_numerics={c+'/'+tag:v for (c,tag),v in numerics.items()},group_rows=len(groups),error_rows=len(errors),cached_raw_logits_bit_identical=cache_exact)


def audit_and_import(source,base=BASE):
    source,base=Path(source).resolve(),Path(base).resolve();snapshot=file_hashes(source)
    status=transfer.read(source/'status.json');identity=transfer.read(source/'identity.json');plan,cal,val=transfer.inputs()
    require(status['status']=='completed' and not status['training_started'] and not status['deployment_replaced'],'Run is incomplete or claims training/deployment')
    require(set(snapshot)==set(status['files'])|{'status.json'},'Missing/extra submitted files')
    for name,digest in status['files'].items():require(snapshot[name]==digest,'Submitted artifact hash differs: '+name)
    provenance=identity['source'];pinned=pinned_sources(ROOT,provenance['commit']);hashes={k:hashlib.sha256(v).hexdigest() for k,v in pinned.items()}
    require(hashes==provenance['files'],'Source proof differs from committed Git')
    require(hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()==provenance['tracked_files_fingerprint'],'Source fingerprint differs')
    require(identity['plan_sha256']==hashes['configs/phase7_calibration_transfer/plan.json']==transfer.sha(transfer.PLAN),'Frozen plan differs')
    same_tree(transfer.read(source/'plan.json'),plan)
    for name,expected in plan['diagnostic_reference']['files'].items():require(transfer.sha(source/'reference_validation'/name)==expected['sha256'],'Reference artifact differs')
    contract=transfer.read(source/'batch_contract.json');batch=contract['batch_size']
    require(contract['same_for_both_models_and_points'] is True and contract['automatic_midrun_batch_change'] is False and contract['probability_device']=='cuda','Batch contract differs')
    require(identity['precision']==contract['precision']=='fp16' and identity['tf32'] is False,'Numeric contract differs')
    mode=transfer.choose_validation_mode(transfer.read(source/'reference_validation/identity.json'),transfer.read(source/'reference_validation/batch_contract.json'),identity['gpu'],identity['torch'],identity['cuda'],batch,provenance['files'])
    require(mode==contract['validation_mode']=='reuse_frozen_logits','Unexpected validation mode for this submitted run')
    run_id=identity['run_id'];require(__import__('re').fullmatch(r'calib_transfer_\d{8}_\d{6}',run_id) is not None,'Invalid run ID')
    original_protection=transfer.read(BASE/'preparation/preservation.json')['files']
    for name,digest in original_protection.items():require(transfer.sha(ROOT/name)==digest,'Old evidence changed: '+name)
    notebooks={p.relative_to(ROOT).as_posix():transfer.sha(p) for p in (ROOT/'notebooks').rglob('*.ipynb')}
    target=base/'runs'/run_id;analysis=target/'analysis';analysis.mkdir(parents=True,exist_ok=True);models={};numerics={}
    with tempfile.TemporaryDirectory(prefix='transfer_audit_') as temporary:
        for arm in plan['models']:
            for name in ['canonical_load.json','probe.json']:
                report=transfer.read(source/arm/name);load=report['canonical_load'] if name=='probe.json' else report
                require(load['sha256']==plan['models'][arm]['sha256'] and load['missing']==load['unexpected']==0 and load['n_keys']==1200,'Canonical checkpoint chain differs')
            out=Path(temporary)/arm;models[arm],numerics[arm]=recompute_model(source,arm,plan,cal,val,batch,out)
        report=transfer.read(source/'report.json');recomputed=copy.deepcopy(report);recomputed['models']=models
        require(report['status']=='completed' and report['validation_mode']==mode and not report['deployment_replaced'] and not report['training_started'],'Invalid root result scope')
        require(report['schema']=='phase7-calibration-transfer-result/v1' and report['retained_package_sha256']==plan['retained_package_sha256'] and report['G72']=='unchanged: FAIL 6/8' and report['G73']=='unchanged: pending','Root result changes retained package/gates')
        same_tree(recomputed,report);copy_preserving(Path(temporary),analysis/'recomputed',file_hashes(Path(temporary)))
    require(snapshot==file_hashes(source),'Source changed during audit');copy_preserving(source,target/'raw',snapshot)
    for name,digest in notebooks.items():require(transfer.sha(ROOT/name)==digest,'Notebook changed during audit')
    notebook=ROOT/'notebooks/task/phase7_calibration_transfer/phase7_calibration_transfer_colab.ipynb';code=[c for c in transfer.read(notebook)['cells'] if c['cell_type']=='code']
    executed=all(c.get('execution_count') is not None for c in code)
    verification=dict(schema='phase7-calibration-transfer-audit/v1',passed=True,run_id=run_id,source_commit=provenance['commit'],source_files_checked=len(hashes),
        raw_files=len(snapshot),raw_bytes=sum((source/name).stat().st_size for name in snapshot),new_calibration_records=2160,frozen_validation_records=4320,scored_records=8640,
        validation_mode=mode,group_rows=sum(v['group_rows'] for v in numerics.values()),error_rows=sum(v['error_rows'] for v in numerics.values()),
        reports_and_paired_ci_reproduced=True,numerics=numerics,executed_notebook_available=executed,local_model_forward_performed=False,
        scope='Saved GPU scores/labels are authoritative; CPU scoring is diagnostic only. No weights/full image archive at local, and current notebook is a template if execution counts are null.')
    transfer.write(analysis/'verification.json',verification)
    transfer.write(analysis/'decision.json',dict(status='experimental_clean_only_calibration_not_accepted',deployment_replaced=False,
        reason='Both clean-fitted points fail the preregistered Q30 Real recall guard of 90%, despite improved fake recall and confidence metrics.',
        retained_package_sha256=plan['retained_package_sha256'],G72='unchanged FAIL 6/8',G73='unchanged pending best-by-condition reference',
        next_hypothesis='Separate condition-robust calibration on independent val_calib clean/Q30 with one shared conservative threshold, or controlled robustness training if ranking gates are the priority. No tuning from val_select or CFE evaluation.'))
    transfer.write(target/'import_manifest.json',dict(source=source.relative_to(ROOT).as_posix(),raw_sha256=snapshot,notebooks_unchanged=notebooks,prior_evidence_files_unchanged=len(original_protection),weights_included=False))
    transfer.write(base/'index.json',dict(current_run=run_id,path=target.relative_to(ROOT).as_posix(),status='completed_audited_guard_failed',deployment_replaced=False,
        retained_package_sha256=plan['retained_package_sha256'],new_weights_trained=False))
    return {k:v for k,v in verification.items() if k!='numerics'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    args=parser.parse_args();print(json.dumps(audit_and_import(args.source),ensure_ascii=False,indent=2))
