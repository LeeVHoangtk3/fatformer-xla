"""Diagnose frozen train/validation evidence without touching CFE or training."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/phase7_validation_diagnostics/plan.json'
DEFAULT_PREPARATION=ROOT/'DATASET/logs/phase7/improvement_round2/diagnostics/preparation'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_write(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validate_records(rows):
    """Check metadata, identities and split isolation; never infer semantic labels."""
    if not rows:raise ValueError('Empty manifest')
    for row in rows:
        path=Path(row['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in row['path'] or ':' in row['path']:
            raise ValueError('Unsafe relative image path')
        if row['split'] not in {'train','val_calib','val_select'} or row['label'] not in (0,1):raise ValueError('Invalid split/label')
        if (row['family']=='real')!=(row['label']==0):raise ValueError('Family/label metadata disagree')
        if not re.fullmatch('[0-9a-f]{64}',row['sha256']):raise ValueError('Invalid image SHA256')
    for key in ['path','sha256']:
        if len({r[key] for r in rows})!=len(rows):raise ValueError('Duplicate manifest '+key+' across records/splits')


def historical_q30_indices(records,max_samples=300,seed=42):
    if max_samples<2 or max_samples>300 or max_samples%2:raise ValueError('Need even historical Q30 count in [2,300]')
    rng=random.Random(seed);selected=[]
    for label in [0,1]:
        indices=[i for i,r in enumerate(records) if r['label']==label]
        if len(indices)<max_samples//2:raise ValueError('Insufficient historical Q30 class population')
        rng.shuffle(indices);selected.extend(indices[:max_samples//2])
    return selected


def validation_tensor(path,condition,geometry,normalize):
    from PIL import Image
    if condition not in {'clean','q30_post_crop'}:raise ValueError('Validation condition must be clean or post-crop Q30')
    with Image.open(path) as image:rgb=geometry(image.convert('RGB'))
    if condition=='q30_post_crop':
        buffer=io.BytesIO();rgb.save(buffer,format='JPEG',quality=30);buffer.seek(0)
        with Image.open(buffer) as decoded:rgb=decoded.convert('RGB')
    return normalize(rgb)


def batch_chunks(indices,batch_size,historical_boundary=None):
    """Keep historical Q30's last short batch separate from the added images."""
    boundaries=[0]+([historical_boundary] if historical_boundary is not None else [])+[len(indices)]
    for start,end in zip(boundaries,boundaries[1:]):
        for offset in range(start,end,batch_size):yield indices[offset:min(offset+batch_size,end)]


def inputs(repo=ROOT):
    repo=Path(repo);plan=read(repo/PLAN.relative_to(ROOT))
    if plan['schema']!='phase7-validation-diagnostics-plan/v1':raise ValueError('Unknown diagnostic plan')
    for tag in ['manifest','val_manifest','operating_point']:
        spec=plan[tag]
        if sha(repo/spec['file'])!=spec['sha256']:raise ValueError('Frozen input changed: '+tag)
    manifest=read(repo/plan['manifest']['file']);rows=manifest['records'];validate_records(rows)
    val=[r for r in rows if r['split']=='val_select'];protocol=plan['protocol']
    if manifest['metadata']['n_records']!=len(rows):raise ValueError('Manifest count metadata differs')
    if val!=read(repo/plan['val_manifest']['file'])['records']:raise ValueError('Validation subset differs from full manifest')
    if len(val)!=protocol['n'] or Counter(r['label'] for r in val)!={0:protocol['n_real'],1:protocol['n_fake']}:raise ValueError('Validation counts differ')
    point=read(repo/plan['operating_point']['file'])
    if point['evidence_level']!='measured' or point['T']!=plan['operating_point']['T'] or point['T_star_default']!=plan['operating_point']['threshold']:
        raise ValueError('Frozen historical operating point differs')
    return plan,rows,val


def prepare(output=DEFAULT_PREPARATION):
    plan,rows,val=inputs();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    isolation_path=ROOT/'DATASET/logs/phase7/data_isolation_report.json';isolation=read(isolation_path)
    if isolation['metadata']['manifest_sha256']!=plan['manifest']['sha256'] or isolation['pass'] is not True or isolation['violations']!=0:
        raise ValueError('Historical data-isolation report does not match passing manifest')
    # Preserve both historical evidence and the user's newly executed notebooks.
    protected={p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'notebooks').rglob('*.ipynb')}
    for folder in ['task_7.14','task_7.15/current','improvement_round1']:
        protected.update({p.relative_to(ROOT).as_posix():sha(p) for p in (ROOT/'DATASET/logs/phase7'/folder).rglob('*') if p.is_file()})
    index=historical_q30_indices(val);chosen=[val[i] for i in index]
    distribution=Counter((r['split'],r['family'],r['generator'],r['label']) for r in rows)
    csv_write(output/'generator_coverage.csv',[dict(split=k[0],family=k[1],generator=k[2],label=k[3],n=v) for k,v in sorted(distribution.items())])
    family=Counter((r['split'],r['family'],r['label']) for r in rows)
    csv_write(output/'family_coverage.csv',[dict(split=k[0],family=k[1],label=k[2],n=v) for k,v in sorted(family.items())])
    q30=Counter((r['family'],r['generator'],r['label']) for r in chosen)
    csv_write(output/'historical_q30_coverage.csv',[dict(family=k[0],generator=k[1],label=k[2],n=v) for k,v in sorted(q30.items())])
    histories={}
    pilot=ROOT/'DATASET/logs/phase7/improvement_round1/runs/pilot_20261109_182359/raw'
    for arm in ['p0_legacy_full','p1_clip_full']:
        path=pilot/arm/'val_groups.csv'
        with path.open(encoding='utf-8',newline='') as stream:data=list(csv.DictReader(stream))
        last=data[-1];history=[]
        for row in data:
            recalls={k:float(row[k]) for k in ['clean_r','clean_f','q30_r','q30_f']}
            if not math.isclose(min(recalls.values()),float(row['worst']),abs_tol=1e-12):raise ValueError('Historical worst-group differs')
            if (float(row['threshold']),float(row['temperature']))!=(plan['operating_point']['threshold'],plan['operating_point']['T']):raise ValueError('Historical operating point differs')
            history.append(dict(epoch=int(row['epoch']),**recalls,worst=float(row['worst'])))
        histories[arm]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),history=history,
            final_clean_false_positives=round((1-float(last['clean_r']))*590),final_clean_true_positives=round(float(last['clean_f'])*490),
            final_q30_false_positives=round((1-float(last['q30_r']))*150),final_q30_true_positives=round(float(last['q30_f'])*150))
    split_counts={s:dict(n=sum(r['split']==s for r in rows),real=sum(r['split']==s and r['label']==0 for r in rows),fake=sum(r['split']==s and r['label']==1 for r in rows)) for s in ['train','val_calib','val_select']}
    fake_families=sorted({r['family'] for r in rows if r['label']==1})
    train_gen={r['generator'] for r in rows if r['split']=='train'}
    report=dict(schema='phase7-validation-diagnostic-preparation/v1',status='metadata_audited_per_image_validation_pending',
        plan_sha256=sha(PLAN),manifest_sha256=plan['manifest']['sha256'],counts=split_counts,
        unique_fake_generators_by_family={f:sorted({r['generator'] for r in rows if r['family']==f}) for f in fake_families},
        validation_generators_unseen_in_training=sorted({r['generator'] for r in val}-train_gen),
        historical_q30=dict(n=300,real=150,fake=150,seed=42,by_family=dict(Counter(r['family'] for r in chosen)),
            ids_sha256=hashlib.sha256('\n'.join(val[i]['sha256'] for i in index).encode()).hexdigest()),
        historical_validation=histories,
        baseline_required_logit_margin=plan['operating_point']['T']*math.log(plan['operating_point']['threshold']/(1-plan['operating_point']['threshold'])),
        local_images_available=sum((ROOT/r['path']).is_file() for r in rows),
        historical_isolation=dict(path=isolation_path.relative_to(ROOT).as_posix(),sha256=sha(isolation_path),pass_reported=True,violations_reported=0,
            local_image_content_audit_repeated=False),
        per_image_validation_predictions_available=False,local_model_forward_performed=False,
        pending=['Per-image val_select logits and scores from both frozen checkpoints.','Semantic label/error review using real images.','New training hypothesis and control fixed only after diagnosis.'],
        limits=['Metadata label consistency does not establish semantic correctness.','Source isolation audit is historical; local image content is unavailable.',
            'All validation generators are seen in training; no unseen-generator generalization claim.',
            'No training or deployment change is authorized by a successful diagnostic forward.'])
    write(output/'audit.json',report)
    write(output/'historical_q30_ids.json',dict(split='val_select',ids=[val[i]['sha256'] for i in index]))
    for name,digest in protected.items():
        if sha(ROOT/name)!=digest:raise ValueError('Protected evidence changed: '+name)
    write(output/'preservation.json',dict(files=protected,all_unchanged=True))
    return dict(status=report['status'],output=str(output),protected_files=len(protected),historical_q30=report['historical_q30']['by_family'])


def summarize_scores(probabilities,targets,labels):
    from sklearn.metrics import roc_auc_score,average_precision_score
    p=np.asarray(probabilities,dtype=float);y=np.asarray(targets);labels=np.asarray(labels)
    if p.ndim!=1 or p.shape!=y.shape or p.shape!=labels.shape or not len(p) or not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or not np.isin(labels,[0,1]).all() or np.any((p<0)|(p>1)):
        raise ValueError('Invalid validation scores/labels')
    real=y==0;fake=y==1
    result=dict(n=len(y),n_real=int(real.sum()),n_fake=int(fake.sum()),accuracy_pct=100*float((labels==y).mean()),
        real_recall_pct=100*float((labels[real]==0).mean()) if real.any() else None,
        fake_recall_pct=100*float((labels[fake]==1).mean()) if fake.any() else None,
        probability_quantiles=np.quantile(p,[.1,.5,.9]).tolist())
    if real.any() and fake.any():
        result.update(auc_pct=100*float(roc_auc_score(y,p)),ap_pct=100*float(average_precision_score(y,p)),
            fake_at_real90_diagnostic_pct=100*float((p[fake]>=np.quantile(p[real],.90)).mean()),
            diagnostic_only='Real90 is estimated on this already-used validation split; not an exported deployment threshold.')
    return result


def load_predictions(path,val,plan,arm,condition):
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    for field,expected in [('split','val_select'),('evidence_level','measured'),('checkpoint_sha256',plan['models'][arm]['sha256']),
                           ('manifest_sha256',plan['manifest']['sha256']),('plan_sha256',sha(PLAN)),('condition',condition),('precision','fp16')]:
        if str(data[field])!=expected:raise ValueError('Prediction provenance differs: '+field)
    ids=data['sample_ids'].astype(str).tolist();expected={r['sha256']:r for r in val}
    if len(ids)!=len(val) or len(set(ids))!=len(ids) or set(ids)!=set(expected):raise ValueError('Validation IDs missing/duplicated/extra')
    for field,key in [('targets','label'),('families','family'),('generators','generator'),('paths','path')]:
        if data[field].tolist()!=[expected[sid][key] for sid in ids]:raise ValueError('Validation identity metadata differs: '+field)
    p=data['prob_fake'];labels=data['labels'];logits=data['logits']
    summarize_scores(p,data['targets'],labels)
    if logits.shape!=(len(val),2) or not np.isfinite(logits).all():raise ValueError('Invalid validation logits')
    if not np.array_equal(labels,(p>=plan['operating_point']['threshold']).astype(int)):raise ValueError('Stored GPU labels disagree with threshold')
    difference=(logits[:,1].astype(float)-logits[:,0].astype(float))/plan['operating_point']['T']
    cpu=np.exp(-np.logaddexp(0,-difference))
    if np.max(np.abs(cpu-p))>1e-6:raise ValueError('Logits/probabilities disagree')
    hist={val[i]['sha256'] for i in historical_q30_indices(val)}
    if not np.array_equal(data['historical_q30'],[sid in hist for sid in ids]):raise ValueError('Historical Q30 membership differs')
    return data


def analyze(run_root,output=None):
    plan,_,val=inputs();run_root=Path(run_root);output=Path(output or run_root/'analysis')
    if output.exists() and any(output.iterdir()):raise ValueError('Analysis output already exists; choose a new directory')
    output.mkdir(parents=True,exist_ok=True);result={};groups=[];errors=[];populations={}
    for arm in plan['models']:
        result[arm]={}
        for condition in plan['protocol']['conditions']:
            data=load_predictions(run_root/arm/('predictions_'+condition+'.npz'),val,plan,arm,condition)
            populations[arm,condition]=data;summary=summarize_scores(data['prob_fake'],data['targets'],data['labels'])
            if condition=='q30_post_crop':
                mask=data['historical_q30'];summary['historical_300']=summarize_scores(data['prob_fake'][mask],data['targets'][mask],data['labels'][mask])
            result[arm][condition]=summary
            for field in ['families','generators']:
                for group in sorted(set(data[field])):
                    mask=data[field]==group;value=summarize_scores(data['prob_fake'][mask],data['targets'][mask],data['labels'][mask])
                    shared=None
                    if value['n_fake'] and not value['n_real']:
                        reference=mask|(data['targets']==0)
                        shared=summarize_scores(data['prob_fake'][reference],data['targets'][reference],data['labels'][reference])
                    groups.append(dict(arm=arm,condition=condition,group_type=field,group=group,n=value['n'],real_recall_pct=value['real_recall_pct'],fake_recall_pct=value['fake_recall_pct'],
                        auc_vs_shared_real_pct=shared['auc_pct'] if shared else None,
                        fake_at_real90_diagnostic_pct=shared['fake_at_real90_diagnostic_pct'] if shared else None,
                        shared_real_reference_n=shared['n_real'] if shared else 0,
                        p10=value['probability_quantiles'][0],p50=value['probability_quantiles'][1],p90=value['probability_quantiles'][2]))
            for i in np.flatnonzero(data['targets']!=data['labels']):
                errors.append(dict(arm=arm,condition=condition,sample_id=str(data['sample_ids'][i]),path=str(data['paths'][i]),family=str(data['families'][i]),generator=str(data['generators'][i]),target=int(data['targets'][i]),prediction=int(data['labels'][i]),prob_fake=float(data['prob_fake'][i])))
    matched={}
    for condition in plan['protocol']['conditions']:
        a,b=(populations[arm,condition] for arm in plan['models'])
        if not np.array_equal(a['sample_ids'],b['sample_ids']):raise ValueError('Model validation order differs')
        ca=a['labels']==a['targets'];cb=b['labels']==b['targets']
        matched[condition]=dict(both_correct=int((ca&cb).sum()),candidate_only_correct=int((ca&~cb).sum()),incumbent_only_correct=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum()))
    historical=dict(p0_clean_real_expected=586/590,p0_clean_fake_expected=296/490,p0_q30_real_expected=147/150,p0_q30_fake_expected=63/150)
    p0=result['candidate_p0'];observed=[p0['clean']['real_recall_pct']/100,p0['clean']['fake_recall_pct']/100,p0['q30_post_crop']['historical_300']['real_recall_pct']/100,p0['q30_post_crop']['historical_300']['fake_recall_pct']/100]
    history_comparison=dict(expected=historical,observed=observed,absolute_differences=[abs(a-b) for a,b in zip(observed,historical.values())],
        note='Different GPU/batch/library versions can change FP16 decisions. Investigate discrepancies; do not replace historical CSVs.')
    report=dict(schema='phase7-per-image-validation-diagnostics/v1',status='completed',metrics=result,matched_correctness=matched,historical_p0_comparison=history_comparison,
        interpretation='Diagnostic only. Group recalls do not prove causal effects; ranking and fixed-threshold performance are reported separately. No threshold, checkpoint or training arm selected for deployment.',
        training_started=False,deployment_replaced=False,scientific_limits=plan['scientific_limits'])
    write(output/'diagnosis.json',report);csv_write(output/'group_metrics.csv',groups)
    if errors:csv_write(output/'errors.csv',errors)
    return report


def collect(drive_root,data_root,run_root,backup_root,max_batch_size=16):
    import torch
    from types import SimpleNamespace
    from PIL import Image
    from src.datasets.preprocessing import preprocessing_parts
    from src.models.canonical_loader import load_canonical
    from tools.phase7_git_source import git_source_identity
    from tools.phase7_training_pilot import command
    plan,_,val=inputs();drive_root,data_root,run_root,backup_root=(Path(p).resolve() for p in [drive_root,data_root,run_root,backup_root])
    if not torch.cuda.is_available():raise RuntimeError('Real validation requires a compatible CUDA GPU')
    if max_batch_size<1 or max_batch_size>16:raise ValueError('Set max batch in [1,16] before running')
    if run_root==backup_root or run_root.is_relative_to(backup_root) or backup_root.is_relative_to(run_root):raise ValueError('Separate local runtime and Drive backup required')
    for folder in [run_root,backup_root]:
        if folder.exists() and any(folder.iterdir()):raise ValueError('Preserve previous results; choose a new diagnostic run ID')
    source=git_source_identity(ROOT)
    if sha(data_root/plan['manifest']['file'])!=plan['manifest']['sha256']:raise ValueError('Staged manifest differs')
    if (data_root/'.extracted_sha256').read_text().strip()!=plan['archive']['sha256']:raise ValueError('Staging archive identity differs')
    # Decode and hash all 1,080 real validation images; no fallback or image skipping.
    for row in val:
        path=(data_root/row['path']).resolve()
        if not path.is_relative_to(data_root) or not path.is_file() or sha(path)!=row['sha256']:raise ValueError('Validation image missing/changed: '+row['path'])
        with Image.open(path) as image:image.convert('RGB').load()
    checkpoints={}
    for arm,spec in plan['models'].items():
        path=drive_root/spec['drive_path']
        if sha(path)!=spec['sha256']:raise ValueError('Wrong diagnostic checkpoint: '+arm)
        checkpoints[arm]=path
    run_root.mkdir(parents=True,exist_ok=True);backup_root.mkdir(parents=True,exist_ok=True)
    identity=dict(schema='phase7-validation-diagnostic-run/v1',source=source,plan_sha256=sha(PLAN),manifest_sha256=plan['manifest']['sha256'],
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,max_batch_size=max_batch_size,precision='fp16',tf32=False,
        validated_image_count=len(val),created_utc=datetime.now(timezone.utc).isoformat())
    write(run_root/'identity.json',identity);write(run_root/'plan.json',plan)
    status=dict(status='running',training_started=False,deployment_replaced=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        for arm in plan['models']:
            folder=run_root/arm;folder.mkdir()
            # Keep source weights on Colab SSD outside the exported result folder.
            cache=run_root.parent/'checkpoint_cache';cache.mkdir(exist_ok=True)
            local_checkpoint=cache/(plan['models'][arm]['sha256']+'.pth')
            if not local_checkpoint.exists():shutil.copy2(checkpoints[arm],local_checkpoint)
            if sha(local_checkpoint)!=plan['models'][arm]['sha256']:raise ValueError('Incomplete/changed checkpoint cache; preserve it and use a fresh runtime')
            checkpoints[arm]=local_checkpoint
            command([sys.executable,'-u','-m','tools.probe_phase7_gpu','--checkpoint',checkpoints[arm],
                '--output',folder/'probe.json','--max_batch_size',max_batch_size,'--expected_checkpoint_sha',plan['models'][arm]['sha256']],run_root/'commands.log')
        batch=min(read(run_root/arm/'probe.json')['selected_batch_size'] for arm in plan['models'])
        write(run_root/'batch_contract.json',dict(batch_size=batch,precision='fp16',same_for_both_models=True,automatic_midrun_batch_change=False))
        geometry,normalize,_,_=preprocessing_parts('clip',224,224)
        historical=historical_q30_indices(val);historical_set=set(historical)
        orders={'clean':list(range(len(val))),'q30_post_crop':historical+[i for i in range(len(val)) if i not in historical_set]}
        for arm in plan['models']:
            args=SimpleNamespace(backbone='CLIP:ViT-L/14',clip_path=None,num_classes=2,use_srm=True,use_gating=True,num_vit_adapter=3,num_context_embedding=8)
            model,load_report=load_canonical(checkpoints[arm],args)
            if load_report['missing'] or load_report['unexpected']:raise ValueError('Canonical load has missing/unexpected keys')
            write(run_root/arm/'canonical_load.json',load_report);model=model.to('cuda').eval()
            for condition,indices in orders.items():
                logits_parts=[];prob_parts=[];label_parts=[]
                seen=0
                for chunk in batch_chunks(indices,batch,len(historical) if condition=='q30_post_crop' else None):
                    tensors=[validation_tensor(data_root/val[i]['path'],condition,geometry,normalize) for i in chunk]
                    with torch.inference_mode():
                        with torch.amp.autocast('cuda',dtype=torch.float16):output=model(torch.stack(tensors).to('cuda'))
                        if isinstance(output,tuple):output=output[0]
                        if output.shape!=(len(chunk),2) or not torch.isfinite(output).all():raise ValueError('Invalid measured logits')
                        probability=torch.softmax(output.float()/plan['operating_point']['T'],-1)[:,1]
                        logits_parts.append(output.float().cpu().numpy());prob_parts.append(probability.cpu().numpy())
                        label_parts.append((probability>=plan['operating_point']['threshold']).long().cpu().numpy())
                    seen+=len(chunk)
                    if seen%(batch*20)==0 or seen==len(indices):print(f'{arm} {condition}: {seen}/{len(indices)}',flush=True)
                selected=[val[i] for i in indices]
                payload=dict(logits=np.concatenate(logits_parts),prob_fake=np.concatenate(prob_parts),labels=np.concatenate(label_parts),
                    targets=np.array([r['label'] for r in selected]),sample_ids=np.array([r['sha256'] for r in selected]),
                    families=np.array([r['family'] for r in selected]),generators=np.array([r['generator'] for r in selected]),paths=np.array([r['path'] for r in selected]),
                    historical_q30=np.array([i in historical_set for i in indices]),split='val_select',evidence_level='measured',condition=condition,
                    checkpoint_sha256=plan['models'][arm]['sha256'],manifest_sha256=plan['manifest']['sha256'],plan_sha256=sha(PLAN),precision='fp16')
                np.savez_compressed(run_root/arm/('predictions_'+condition+'.npz'),**payload)
            del model;torch.cuda.empty_cache()
        analyze(run_root)
        status.update(status='completed',files={p.relative_to(run_root).as_posix():sha(p) for p in run_root.rglob('*') if p.is_file() and p.name!='status.json'})
    except Exception as error:
        status.update(status='failed',error=str(error));raise
    finally:
        write(run_root/'status.json',status)
        for path in run_root.rglob('*'):
            if path.is_file():
                target=backup_root/path.relative_to(run_root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        with zipfile.ZipFile(backup_root/'validation_diagnostics_without_weights.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(run_root.rglob('*')):
                if path.is_file():archive.write(path,path.relative_to(run_root))
    return dict(status='completed',batch_size=batch,backup=str(backup_root),training_started=False,deployment_replaced=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,default=DEFAULT_PREPARATION)
    p=sub.add_parser('collect')
    for name in ['drive_root','data_root','run_root','backup_root']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max_batch_size',type=int,default=16)
    p=sub.add_parser('analyze');p.add_argument('--run_root',type=Path,required=True);p.add_argument('--output',type=Path)
    args=vars(parser.parse_args());action=args.pop('action')
    print(json.dumps(globals()[action](**args),ensure_ascii=False,indent=2))
