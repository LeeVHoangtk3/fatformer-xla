"""Audit submitted matched-evaluation arrays and preserve raw evidence unchanged."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import fnmatch
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

import numpy as np
from PIL import Image
import torch

from tools.audit_phase7_training_pilot import Checks
from tools.calibration_data import sha256_file
from tools.calibrate_and_select import compute_nll,fit_thresholds_by_fpr
from tools.phase7_corrected import CONDITIONS,INDIST_COUNTS,write_json
from tools.phase7_git_source import SOURCE_PATHS
from tools.run_phase7_corrected import metrics
from tools.run_locked_benchmark_phase7 import evaluate_master_gates

ROOT=Path(__file__).resolve().parents[1]
ARMS=('candidate_p0','incumbent_refit')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def near(a,b,tolerance=1e-8):return bool(np.allclose(a,b,rtol=0,atol=tolerance))


def pinned_sources(root,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Invalid source commit')
    names=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=root,text=True).splitlines()
    names=[name for name in names if any(name==spec or name.startswith(spec+'/') or fnmatch.fnmatchcase(name,spec) for spec in SOURCE_PATHS)]
    data=subprocess.check_output(['git','-c','core.autocrlf=false','archive','--format=tar',commit,'--',*names],cwd=root)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return {m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def metric_checks(check,actual,reported):
    for key,value in actual.items():
        if isinstance(value,dict):
            for child,number in value.items():check.need('metric_'+key,near(number,reported[key][child]),child)
        elif value is not None:check.need('metric_'+key,near(value,reported[key]),str((value,reported[key])))


def temperature_numpy(logits,targets):
    """Independent binary NLL derivative in inverse temperature, no GPU refit."""
    logits,targets=np.asarray(logits),np.asarray(targets)
    if targets.ndim!=1 or logits.shape!=(len(targets),2) or not np.isfinite(logits).all() or set(targets.tolist())!={0,1}:
        raise ValueError('Need finite binary logits and both target classes')
    z=logits[:,1].astype(np.float64)-logits[:,0].astype(np.float64)
    def derivative(beta):return float(np.mean(z*(np.exp(-np.logaddexp(0,-beta*z))-targets)))
    low,high=.1,20.
    if derivative(low)>=0:beta=low
    elif derivative(high)<=0:beta=high
    else:
        for _ in range(100):
            middle=(low+high)/2
            if derivative(middle)>0:high=middle
            else:low=middle
        beta=(low+high)/2
    return dict(T_float64=1/beta,T_float32=float(np.float32(1/beta)),
                nll=float(np.mean(np.logaddexp(0,beta*z)-targets*beta*z)),gradient_beta=derivative(beta))


def audit(source,notebook,repo=ROOT):
    source,notebook,repo=Path(source).resolve(),Path(notebook).resolve(),Path(repo).resolve()
    check=Checks();identity=read(source/'eval_identity.json');provenance=read(source/'source_identity.json')
    status=read(source/'status.json');commit=identity['source_commit'];pinned=pinned_sources(repo,commit)
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in pinned.items()}
    check.need('source_file_set',set(hashes)==set(provenance['files']))
    for name,expected in provenance['files'].items():check.need('source_file_hash',hashes.get(name)==expected,name)
    fingerprint=hashlib.sha256(json.dumps(provenance['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    check.need('source_fingerprint',fingerprint==identity['source_fingerprint']==provenance['tracked_files_fingerprint'])
    check.need('source_commit',commit==provenance['commit']==status['source_commit'])
    plan=json.loads(pinned['configs/phase7_pilot_eval/plan.json'])
    check.need('plan_hash',hashes['configs/phase7_pilot_eval/plan.json']==identity['plan_sha256'])
    check.need('staging_hash',sha256_file(source/'staging.json')==identity['staging_sha256'])
    check.need('staging_archives',read(source/'staging.json')['archives']==plan['archives'])
    check.need('completed_run',status['status']=='completed' and status['stage']=='full' and status['deployment_replaced'] is False)
    contract=read(source/'batch_contract.json')
    check.need('batch_contract',contract==dict(batch_size=16,precision='fp16',probability_device='cuda',same_for_both_models=True,automatic_midrun_batch_change=False))
    ids={};targets={}
    for tag in ['calibration','evaluation']:
        part=plan['partitions'][tag];payload=json.loads(pinned[part['file']])
        check.need('partition_hash',hashes[part['file']]==part['sha256'])
        ids[tag]=np.array([x['sample_id'] for x in payload['samples']]);targets[tag]=np.array([x['target'] for x in payload['samples']])
        check.need('partition_count_balance',len(ids[tag])==part['n'] and int((targets[tag]==0).sum())==part['real'] and int((targets[tag]==1).sum())==part['fake'])
    check.need('partition_isolation',not set(ids['calibration'])&set(ids['evaluation']))
    n=read(notebook);cells=[c for c in n['cells'] if c['cell_type']=='code']
    text='\n'.join(''.join(o.get('text',[])) for c in cells for o in c.get('outputs',[]))
    check.need('notebook_execution_order',[c.get('execution_count') for c in cells]==[1,2,3])
    check.need('notebook_no_errors',not any(o.get('output_type')=='error' for c in cells for o in c.get('outputs',[])))
    check.need('notebook_status',all(c.get('metadata',{}).get('executionInfo',{}).get('status')=='ok' for c in cells))
    check.need('notebook_commit',commit in text)
    run_ids=set(re.findall(r'p0_eval_\d{8}_\d{6}',text));check.need('notebook_single_run',len(run_ids)==1)
    if len(run_ids)!=1:raise ValueError('Missing/ambiguous evaluation run ID')
    run_id=next(iter(run_ids))
    log=(source/'commands.log').read_text(encoding='utf-8')
    commands=[json.loads(line[8:]) for line in log.splitlines() if line.startswith('COMMAND ')]
    requests=[]
    for command in commands:
        index=command.index('-m');request=dict(identity=identity,module=command[index+1],args=command[index+2:])
        requests.append(hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest())
    unavailable=[]
    for marker in sorted((source/'steps').glob('*.json')):
        saved=read(marker);check.need('step_signature_in_command_log',saved['signature'] in requests,marker.name)
        for name,expected in saved['files'].items():
            path=source/name
            if name.endswith('.pth') and not path.exists():unavailable.append(dict(path=name,sha256=expected));continue
            check.need('step_output_hash',path.is_file() and sha256_file(path)==expected,name)
    check.need('all_steps_recorded',len(list((source/'steps').glob('*.json')))==len(commands)==17)
    check.need('only_expected_weights_absent',{r['path'] for r in unavailable}=={arm+'/package/unified.pth' for arm in ARMS})
    image_ids=read(source/'verification_image_ids.json');images=sorted((source/'verification_images').glob('*'))
    check.need('verification_image_population',len(image_ids)==len(images)==20 and Counter(r['target'] for r in image_ids)=={0:10,1:10})
    check.need('verification_ids_from_calibration',all(x['sample_id'] in set(ids['calibration']) and x['source'].startswith('calibration') for x in image_ids))
    check.need('verification_images_unique',len({sha256_file(path) for path in images})==20)
    for path in images:
        with Image.open(path) as image:image.verify()
    reports={};recomputed={};calibrations={};latencies={};numerics={};gate_summary={};references={}
    for arm in ARMS:
        folder=source/arm;report=read(folder/'benchmark/final_report.json');reports[arm]=report
        calibration=read(folder/'calibration'/(arm+'.json'));calibrations[arm]=calibration
        expected_source=plan['models'][arm]['sha256'];package=report['package_sha256']
        verification=read(folder/'verification.json');metadata=read(folder/'package/metadata.json');reference=read(folder/'package_reference.json')
        check.need('report_complete',report['status']=='completed' and report['source_checkpoint_sha256']==expected_source)
        check.need('report_git_provenance',report['bundle']==provenance)
        check.need('source_hash_chain',metadata['source_checkpoint_sha256']==verification['reference_sha256']==calibration['checkpoint']['sha256']==reference['source_checkpoint_sha256']==expected_source)
        check.need('package_hash_chain',verification['checkpoint_sha256']==reference['sha256']==package)
        check.need('metadata_calibration',metadata['calibration']==calibration and metadata['router_enabled'] is False)
        check.need('gpu_equivalence_report',verification['tensors_equal'] is True and verification['n_tensors']==1200 and verification['n_images']==20 and verification['real_image_acceptance'] is True and verification['max_probability_error']==0)
        probe=read(folder/'gpu_probe.json');canonical=probe['canonical_load']
        check.need('canonical_load',canonical['sha256']==expected_source and canonical['missing']==canonical['unexpected']==0 and not canonical['layout']['partial'] and canonical['layout']['indices']==[7,15,23])
        check.need('cuda_batch_fit',probe['gpu_name']==identity['gpu'] and probe['selected_batch_size']==16)
        for name,expected in report['artifacts'].items():check.need('benchmark_artifact_hash',sha256_file(folder/'benchmark'/name)==expected,name)
        with np.load(folder/'calibration/logits_calib.npz',allow_pickle=False) as z:
            logits=z['logits'].copy();y=z['targets'].copy()
            check.need('calibration_ids_labels',np.array_equal(z['sample_ids'],ids['calibration']) and np.array_equal(y,targets['calibration']))
            check.need('calibration_npz_provenance',str(z['checkpoint_sha256'])==expected_source and str(z['evidence_level'])=='measured' and str(z['precision'])=='fp16' and int(z['inference_batch_size'])==16)
        check.need('calibration_source_hash',sha256_file(folder/'calibration/logits_calib.npz')==calibration['calibration_source_sha256'])
        fit=temperature_numpy(logits,y)
        check.need('temperature_independent_fit',abs(fit['T_float32']-calibration['T'])<5e-6)
        check.need('temperature_nll_objective',abs(fit['nll']-calibration['temperature_optimization']['nll_at_fitted_T'])<1e-7)
        check.need('temperature_converged',calibration['temperature_optimization']['converged'] is True)
        cpu_probs=torch.softmax(torch.tensor(logits).float()/float(calibration['T']),dim=-1)[:,1].numpy()
        cpu_thresholds,cpu_fpr=fit_thresholds_by_fpr(cpu_probs,y)
        check.need('threshold_cpu_gpu_tolerance',max(abs(cpu_thresholds[k]-calibration['T_star'][k]) for k in cpu_thresholds)<1e-6)
        check.need('calibration_nll_rounding',abs(compute_nll(cpu_probs,y)-calibration['nll_after'])<=.000051)
        check.need('runtime_calibration_fpr',read(folder/'calibration_preflight.json')['fpr_calibration']==calibration['fpr_measured']['0.05']==.05)
        numerics[arm]=dict(independent_temperature=fit,cpu_thresholds=cpu_thresholds,cpu_refitted_fpr=cpu_fpr,
            cpu_fpr_at_gpu_serialized_threshold=float((cpu_probs[y==0]>=calibration['T_star_default']).mean()),
            note='CPU softmax is diagnostic; stored GPU probabilities/labels are authoritative. ULP-scale threshold boundaries must not rewrite the measured GPU decision.')
        recomputed[arm]={}
        for name in [*['cfe_'+c for c in CONDITIONS],'indist']:
            path=folder/'benchmark'/('predictions_'+name+'.npz')
            with np.load(path,allow_pickle=False) as z:
                p=z['prob_fake'].astype(np.float64);y=z['targets'].copy();labels=z['labels'].copy();subsets=z['subsets'].astype(str);sample_ids=z['sample_ids'].astype(str);logits=z['logits'].copy()
                check.need('prediction_provenance',str(z['package_sha256'])==package and str(z['evidence_level'])=='measured' and str(z['precision'])=='fp16')
                check.need('fixed_fallback_predictions',np.all(z['w_deg']==0) and np.all(z['thresholds']==np.float32(calibration['T_star_default'])))
            expected_count=82003 if name=='indist' else 1600
            check.need('prediction_shapes',p.shape==y.shape==labels.shape==sample_ids.shape==(expected_count,) and logits.shape==(expected_count,2))
            check.need('prediction_values',np.isfinite(p).all() and np.isfinite(logits).all() and np.all((p>=0)&(p<=1)) and set(y.tolist())=={0,1})
            check.need('prediction_id_uniqueness',len(set(sample_ids))==expected_count)
            check.need('stored_gpu_labels',np.array_equal(labels,(p>=calibration['T_star_default']).astype(int)))
            if name!='indist':check.need('evaluation_ids_labels',np.array_equal(sample_ids,ids['evaluation']) and np.array_equal(y,targets['evaluation']))
            else:check.need('indist_subset_counts',Counter(subsets.tolist())==INDIST_COUNTS)
            cpu=torch.softmax(torch.tensor(logits).float()/float(calibration['T']),-1)[:,1].numpy()
            max_error=float(np.max(np.abs(cpu-p)));check.need('prediction_cpu_gpu_tolerance',max_error<1e-6)
            cpu_flips=int(((cpu>=calibration['T_star_default']).astype(int)!=labels).sum())
            actual=metrics(p,y,calibration['T_star_default']);groups={}
            reported=report['indist']['pooled'] if name=='indist' else report['cfe'][name[4:]]['metrics']
            metric_checks(check,actual,reported)
            recorded_groups=report['indist']['per_subset'] if name=='indist' else report['cfe'][name[4:]]['per_subset']
            check.need('subset_names',set(subsets)==set(recorded_groups))
            for subset in sorted(set(subsets)):
                mask=subsets==subset;value=metrics(p[mask],y[mask],calibration['T_star_default']);metric_checks(check,value,recorded_groups[subset]);groups[subset]=value
            actual.update(fixed_fpr_pct=100*float((labels[y==0]==1).mean()),false_positives=int((labels[y==0]==1).sum()),true_positives=int((labels[y==1]==1).sum()))
            recomputed[arm][name]=dict(metrics=actual,per_subset=groups,cpu_max_probability_error=max_error,cpu_label_disagreements=cpu_flips)
        subset_metrics=recomputed[arm]['indist']['per_subset']
        macro=float(np.mean([r['balanced_acc'] for r in subset_metrics.values()]))
        check.need('indist_macro_definition',near(macro,report['indist']['macro_acc']))
        independent_gate=evaluate_master_gates(dict(macro_acc=macro,real_acc=recomputed[arm]['indist']['metrics']['real_acc']),
            recomputed[arm]['cfe_clean']['metrics'],recomputed[arm]['cfe_q30']['metrics'],recomputed[arm]['cfe_clean']['per_subset'],0,{})['G7.2']
        acceptance=read(folder/'acceptance.json')
        check.need('g72_recomputed',independent_gate==acceptance['G72']==report['gates']['G7.2'])
        gate_summary[arm]=dict(passed=sum(v['pass'] for v in independent_gate['criteria'].values()),total=8,gate=independent_gate)
        latency=read(folder/'latency.json');latencies[arm]={}
        check.need('latency_protocol',latency['protocol']['warmup']==20 and latency['protocol']['steps']==200 and latency['allow_tf32'] is False)
        for batch,row in latency['batches'].items():
            b=int(batch);a=np.asarray(row['backbone_samples_ms']);p=np.asarray(row['package_samples_ms']);delta=(p-a)/b
            check.need('latency_pair_count',len(a)==len(p)==200 and np.all(a>0) and np.all(p>0))
            check.need('latency_raw_delta',near(delta,row['paired_delta_per_image_ms']))
            check.need('latency_median_delta',near(np.median(delta),row['overhead_delta_ms']))
            check.need('latency_p95_delta',near(np.quantile(delta,.95),row['overhead_delta_p95_ms']))
            check.need('latency_package_time',round(float(np.median(p)/b),2)==row['package_per_image_ms'])
            latencies[arm][batch]=dict(package_per_image_ms=row['package_per_image_ms'],overhead_ms=float(np.median(delta)),overhead_p95_ms=float(np.quantile(delta,.95)),pairs=200)
        check.need('g73_pending',acceptance['G73']['status']=='pending' and acceptance['G73']['condition_best_comparison']=='pending_verified_reference')
        check.need('same_package_acceptance',acceptance['package_sha256']==latency['checkpoint_sha256']==package)
        for path,expected in acceptance['source_reports'].items():
            relative=path.split('/'+run_id+'/')[-1];check.need('acceptance_source_hash',sha256_file(source/relative)==expected)
        references[arm]=reference
    for field in ['environment','preprocess','bootstrap','precision_contract']:
        check.need('matched_protocol_'+field,reports[ARMS[0]][field]==reports[ARMS[1]][field])
    check.finish()
    return dict(schema='phase7-p0-evaluation-local-audit/v1',passed=True,run_id=run_id,source_commit=commit,source_files_checked=len(hashes),
        checks=check.groups,data_checks=sum(v['checked'] for v in check.groups.values()),recomputed=recomputed,calibration_numerics=numerics,
        gate_summary=gate_summary,latency=latencies,model_references=references,unavailable_weight_files=unavailable,
        notebook=dict(path=notebook.relative_to(repo).as_posix(),sha256=sha256_file(notebook),execution_counts=[c['execution_count'] for c in cells],
            evaluation_wall_minutes=cells[-1]['metadata']['executionInfo']['elapsed']/60000),
        decision='Keep existing deployment package 24869052; P0 is not an improvement on the predefined acceptance goals. No automatic deployment.',
        scientific_scope='Post-hoc, single training seed; incumbent weights refit, not a binary replay of deployed package.')


def same_tree(actual,expected,path='root'):
    """Require the entire submitted comparison, including CIs, to reproduce."""
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('Comparison keys differ: '+path)
        for key in expected:same_tree(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(actual)!=len(expected):raise ValueError('Comparison shape differs: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):same_tree(a,b,path+f'[{i}]')
    elif isinstance(expected,(int,float)) and not isinstance(expected,bool):
        if not near(actual,expected):raise ValueError('Comparison numeric mismatch: '+path)
    elif actual!=expected:raise ValueError('Comparison mismatch: '+path)


def paired_verification(source):
    from tools.compare_pilot_checkpoints import compare
    with tempfile.TemporaryDirectory(prefix='phase7_paired_audit_') as temp:
        recomputed=compare(source,Path(temp)/'comparison.json',n_boot=1000)
    same_tree(recomputed,read(Path(source)/'comparison.json'))
    return dict(passed=True,comparison=recomputed,
        scope='All paired CFE image-bootstrap CIs (1000 per condition) and diagnostic macro subset CI (10000) recomputed. Per-arm marginal bootstrap CIs are source reports, not independently rerun here.')


def incumbent_reproduction(source,repo=ROOT):
    old=Path(repo)/'DATASET/logs/phase7/task_7.15/current/raw/benchmark'
    new=Path(source)/'incumbent_refit/benchmark';result={}
    for path in sorted(new.glob('predictions_*.npz')):
        with np.load(path,allow_pickle=False) as a,np.load(old/path.name,allow_pickle=False) as b:
            for key in ['sample_ids','targets','subsets']:
                if not np.array_equal(a[key],b[key]):raise ValueError('Old incumbent identity differs: '+path.name+'/'+key)
            result[path.stem]=dict(n=len(a['targets']),max_probability_error=float(np.max(np.abs(a['prob_fake']-b['prob_fake']))),
                label_disagreements=int((a['labels']!=b['labels']).sum()),logits_identical=bool(np.array_equal(a['logits'],b['logits'])))
    return dict(old_package_sha256=read(old/'final_report.json')['package_sha256'],
        refit_package_sha256=read(new/'final_report.json')['package_sha256'],populations=result,
        all_predictions_identical=all(r['max_probability_error']==0 and r['label_disagreements']==0 and r['logits_identical'] for r in result.values()),
        scope='Comparison of submitted predictions, not reloading either missing PTH package locally.')


def file_hashes(folder):
    return {p.relative_to(folder).as_posix():sha256_file(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def copy_preserving(source,destination,expected):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    for name,digest in expected.items():
        original=Path(source)/name;target=destination/name
        if sha256_file(original)!=digest:raise ValueError('Source changed during audit: '+name)
        if target.exists():
            if sha256_file(target)!=digest:raise ValueError('Refusing to overwrite different evidence: '+str(target))
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
    if file_hashes(destination)!=expected:raise ValueError('Imported evidence hash/file set differs')


def export_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def import_submission(source,notebook,base,repo=ROOT):
    source,notebook,base,repo=(Path(p).resolve() for p in [source,notebook,base,repo])
    before=file_hashes(source);notebook_hash=sha256_file(notebook)
    protected=read(repo/'DATASET/logs/phase7/task_7.16/documentation_manifest.json')['protected']
    # Preserve historical protected evidence, and all submitted notebooks.
    for name,digest in protected.items():
        if sha256_file(repo/name)!=digest:raise ValueError('Protected evidence changed before import: '+name)
    for p in (repo/'notebooks').rglob('*.ipynb'):protected.setdefault(p.relative_to(repo).as_posix(),sha256_file(p))
    verified=audit(source,notebook,repo)
    paired=paired_verification(source);reproduction=incumbent_reproduction(source,repo)
    if before!=file_hashes(source) or sha256_file(notebook)!=notebook_hash:raise ValueError('Submission changed during audit')
    target=base/'runs'/verified['run_id'];analysis=target/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    copy_preserving(source,target/'raw',before)
    copy_preserving(notebook.parent,target/'evidence', {notebook.name:notebook_hash})
    recomputed=verified.pop('recomputed')
    write_json(analysis/'verification.json',verified);write_json(analysis/'recomputed_metrics.json',recomputed)
    write_json(analysis/'paired_verification.json',paired);write_json(analysis/'incumbent_reproduction.json',reproduction)
    rows=[]
    for condition,record in paired['comparison']['CFE'].items():
        for metric in ['auc','fake_at_real90']:
            ci=record['paired_ci95_pp'][metric]
            rows.append(dict(population='cfe_'+condition,metric=metric,candidate_pct=record['candidate'][metric],
                reference_pct=record['reference'][metric],delta_pp=record['candidate_minus_reference_pp'][metric],ci95_low_pp=ci[0],ci95_high_pp=ci[1]))
    export_csv(analysis/'comparison.csv',rows)
    export_csv(analysis/'gates.csv',[dict(arm=arm,criterion=key,**value) for arm,data in verified['gate_summary'].items()
        for key,value in data['gate']['criteria'].items()])
    export_csv(analysis/'latency.csv',[dict(arm=arm,batch_size=batch,**value) for arm,data in verified['latency'].items() for batch,value in data.items()])
    decision=dict(schema='phase7-pilot-evaluation-decision/v1',evaluation_run=verified['run_id'],
        selected_training_checkpoint='p0_legacy_full',candidate_accepted=False,deployment_replaced=False,
        retained_package_sha256=reproduction['old_package_sha256'],G72={arm:dict(pass_gate=False,criteria_passed=data['passed'],criteria_total=8) for arm,data in verified['gate_summary'].items()},
        G73='pending_verified_best_by_condition_reference',
        reason='P0 misses four predefined G7.2 criteria; refitted incumbent misses two. No demonstrated improvement sufficient to replace the incumbent.',
        next_step='Close round1 as a negative result. Before new GPU training, define a new hypothesis on training/validation data; do not tune to the inspected CFE test or lower acceptance gates.')
    write_json(analysis/'decision.json',decision)
    for name,digest in protected.items():
        if sha256_file(repo/name)!=digest:raise ValueError('Protected evidence changed: '+name)
    manifest=dict(schema='phase7-pilot-evaluation-import/v1',imported_utc=datetime.now(timezone.utc).isoformat(),
        source=source.relative_to(repo).as_posix(),destination=target.relative_to(repo).as_posix(),raw_count=len(before),
        raw_bytes=sum((source/n).stat().st_size for n in before),raw_sha256=before,
        notebook=dict(source=notebook.relative_to(repo).as_posix(),copy=(target/'evidence'/notebook.name).relative_to(repo).as_posix(),sha256=notebook_hash),
        protected_files_unchanged=protected,source_unchanged=True,weights_included=False,local_model_forward_performed=False)
    write_json(target/'import_manifest.json',manifest)
    write_json(base/'index.json',dict(schema='phase7-pilot-evaluation-index/v1',current_run=verified['run_id'],
        path=target.relative_to(repo).as_posix(),state='completed_and_audited_negative_result',deployment_replaced=False,
        candidate_G72='FAIL 4/8',incumbent_G72='FAIL 6/8',G73='pending_verified_best_by_condition_reference'))
    return dict(path=str(target),raw_files=len(before),data_checks=verified['data_checks'],protected_files=len(protected),paired_comparison_reproduced=True,incumbent_predictions_identical=reproduction['all_predictions_identical'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--notebook',required=True)
    parser.add_argument('--base',default='DATASET/logs/phase7/improvement_round1/evaluation')
    parser.add_argument('--import_results',action='store_true')
    args=parser.parse_args()
    if args.import_results:result=import_submission(args.source,args.notebook,args.base)
    else:
        result=audit(args.source,args.notebook)
        result.pop('recomputed');result['paired_verification']=paired_verification(args.source);result['incumbent_reproduction']=incumbent_reproduction(args.source)
    print(json.dumps(result,ensure_ascii=False,indent=2))
