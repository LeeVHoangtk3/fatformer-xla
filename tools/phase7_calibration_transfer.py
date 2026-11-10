"""Controlled calibration transfer on frozen checkpoints; never train or deploy."""
import argparse
from datetime import datetime,timezone
import gc
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score

from tools import phase7_validation_diagnostics as diagnostic
from tools.calibrate_and_select import ModelWithTemperature,calibration_probabilities,compute_ece,compute_nll,fit_thresholds_by_fpr

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/phase7_calibration_transfer/plan.json'
read,write,sha=diagnostic.read,diagnostic.write,diagnostic.sha


def require(condition,message):
    if not condition:raise ValueError(message)


def inputs():
    plan=read(PLAN);_,records,validation=diagnostic.inputs()
    require(plan['schema']=='phase7-calibration-transfer-plan/v1','Unknown transfer plan')
    for tag in ['manifest','calibration_manifest','validation_manifest']:
        require(sha(ROOT/plan[tag]['file'])==plan[tag]['sha256'],'Frozen partition changed: '+tag)
    calibration=[r for r in records if r['split']=='val_calib']
    require(calibration==read(ROOT/plan['calibration_manifest']['file'])['records'],'Calibration subset differs')
    for subset,tag in [(calibration,'calibration_count'),(validation,'evaluation_count')]:
        require(len(subset)==plan['protocol'][tag],'Wrong split count')
        require(sum(r['label']==0 for r in subset)==plan['protocol']['real'] and sum(r['label']==1 for r in subset)==plan['protocol']['fake'],'Wrong class counts')
    require(not {r['sha256'] for r in calibration}&{r['sha256'] for r in validation},'Calibration/validation overlap')
    for arm,spec in plan['controls'].items():
        path=ROOT/spec['file'];control=read(path)
        require(sha(path)==spec['sha256'] and control['evidence_level']=='measured','Control file changed/not measured')
        require(control['checkpoint']['sha256']==plan['models'][arm]['sha256'],'Control/checkpoint mismatch')
    return plan,calibration,validation


def stage_reference(drive_root,destination,plan):
    """Copy only the sixteen audited files, from their Drive folder or result ZIP."""
    spec=plan['diagnostic_reference'];folder=Path(drive_root)/spec['drive_path'];destination=Path(destination)
    require(not destination.exists(),'Reference staging already exists; preserve it and choose a new run')
    destination.mkdir(parents=True)
    archive_path=folder/'validation_diagnostics_without_weights.zip'
    for name,expected in spec['files'].items():
        require('..' not in Path(name).parts and not Path(name).is_absolute(),'Unsafe reference path')
        original=folder/name
        if original.is_file():
            require(original.stat().st_size==expected['bytes'],'Reference size differs: '+name)
            content=original.read_bytes()
        else:
            require(archive_path.is_file(),'Missing audited reference on Drive: '+str(original))
            with zipfile.ZipFile(archive_path) as archive:
                require(archive.namelist().count(name)==1,'Missing/duplicate reference ZIP entry: '+name)
                require(archive.getinfo(name).file_size==expected['bytes'],'Reference ZIP size differs')
                content=archive.read(name)
        import hashlib
        require(hashlib.sha256(content).hexdigest()==expected['sha256'],'Reference SHA256 differs: '+name)
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    status=read(destination/'status.json');identity=read(destination/'identity.json')
    require(status['status']=='completed' and not status['training_started'] and not status['deployment_replaced'],'Reference run incomplete')
    require(identity['source']['commit']==spec['source_commit'],'Reference source commit differs')
    for name,digest in status['files'].items():require(sha(destination/name)==digest,'Reference status hash differs')
    _,_,validation=diagnostic.inputs();old_plan=read(diagnostic.PLAN)
    for arm in plan['models']:
        for condition in plan['protocol']['conditions']:
            diagnostic.load_predictions(destination/arm/f'predictions_{condition}.npz',validation,old_plan,arm,condition)
    return identity,read(destination/'batch_contract.json')


def choose_validation_mode(reference_identity,reference_contract,gpu,torch_version,cuda_version,batch,current_source_files=None):
    same=(reference_identity['gpu']==gpu and reference_identity['torch']==torch_version and reference_identity['cuda']==cuda_version
          and reference_contract['batch_size']==batch)
    if current_source_files is not None:
        def inference_files(files):return {name:digest for name,digest in files.items() if name.startswith('src/') or name=='tools/phase7_validation_diagnostics.py'}
        same=bool(same and inference_files(current_source_files)==inference_files(reference_identity.get('source',{}).get('files',{})))
    return 'reuse_frozen_logits' if same else 'remeasure_both_models'


def payload(logits,records,split,condition,arm,plan,batch,origin):
    return dict(logits=np.asarray(logits,dtype=np.float32),targets=np.array([r['label'] for r in records],dtype=np.int64),
        sample_ids=np.array([r['sha256'] for r in records]),families=np.array([r['family'] for r in records]),
        generators=np.array([r['generator'] for r in records]),paths=np.array([r['path'] for r in records]),
        split=split,condition=condition,evidence_level='measured',precision='fp16',batch_size=batch,
        checkpoint_sha256=plan['models'][arm]['sha256'],manifest_sha256=plan['manifest']['sha256'],plan_sha256=sha(PLAN),origin=origin)


def validate_logits(data,records,split,condition,arm,plan,batch):
    for field,value in [('split',split),('condition',condition),('evidence_level','measured'),('precision','fp16'),
                        ('checkpoint_sha256',plan['models'][arm]['sha256']),('manifest_sha256',plan['manifest']['sha256']),('plan_sha256',sha(PLAN))]:
        require(str(data[field])==value,'Logit provenance mismatch: '+field)
    ids=data['sample_ids'].tolist();expected={r['sha256']:r for r in records}
    require(len(ids)==len(set(ids))==len(records) and set(ids)==set(expected),'Wrong/missing/duplicate split IDs')
    for field,key in [('targets','label'),('families','family'),('generators','generator'),('paths','path')]:
        require(data[field].tolist()==[expected[sid][key] for sid in ids],'Logit identity metadata mismatch: '+field)
    z=data['logits'];require(z.shape==(len(records),2) and np.isfinite(z).all(),'Invalid logits')
    require(int(data['batch_size'])==batch,'Calibration/validation batch differs')
    require(set(data['targets'].tolist())=={0,1},'Both classes required')
    return data


def fit_point(calibration,records,arm,plan,batch,device='cuda'):
    """Only clean val_calib is accepted; validation is not an argument to the fit."""
    validate_logits(calibration,records,'val_calib','clean',arm,plan,batch)
    logits=torch.tensor(calibration['logits']).float();targets=torch.tensor(calibration['targets']).long()
    fitter=ModelWithTemperature();temperature,_=fitter.fit(logits,targets,max_iter=plan['protocol']['temperature_max_iter'])
    require(fitter.fit_diagnostics['converged'] is True,'Temperature did not converge')
    probabilities=calibration_probabilities(logits,temperature,device,batch)
    y=targets.numpy();thresholds,_=fit_thresholds_by_fpr(probabilities,y,[plan['protocol']['fpr_target']])
    threshold=thresholds['0.05'];pred=probabilities>=np.float32(threshold);n_real=int((y==0).sum());false_positives=int(pred[y==0].sum())
    allowed=math.floor(plan['protocol']['fpr_target']*n_real)
    require(false_positives<=allowed,'Measured calibration FPR exceeds target')
    return dict(schema='phase7-calibration-transfer-fit/v1',deployable=False,evidence_level='measured',split='val_calib',condition='clean',
        checkpoint_sha256=plan['models'][arm]['sha256'],manifest_sha256=plan['manifest']['sha256'],plan_sha256=sha(PLAN),
        n=len(y),n_real=n_real,n_fake=int((y==1).sum()),calibration_ids_sha256=__import__('hashlib').sha256('\n'.join(calibration['sample_ids']).encode()).hexdigest(),
        T=temperature,threshold=threshold,comparison_operator='>=',probability_device=device,probability_dtype='float32',batch_size=batch,
        false_positive_budget=allowed,false_positives=false_positives,empirical_fpr=false_positives/n_real,
        temperature_optimization=fitter.fit_diagnostics,fit_only_metrics=dict(nll=compute_nll(probabilities,y),confidence_ece_pct=compute_ece(probabilities,y)),
        scope='Research schema, not calibration/v1. One clean-fitted threshold used for both conditions; no validation labels used to fit parameters.')


def metrics(probability,targets,labels):
    result=diagnostic.summarize_scores(probability,targets,labels)
    result.update(fpr_pct=100*float(labels[targets==0].mean()),nll=compute_nll(probability,targets),
        confidence_ece_argmax_pct=compute_ece(probability,targets))
    return result


def paired_operating_ci(y,new_scores,old_scores,new_labels,old_labels,n_boot=1000,seed=42):
    """Conditional on the two fitted points; same image draws in each comparison."""
    rng=np.random.default_rng(seed);real=np.flatnonzero(y==0);fake=np.flatnonzero(y==1)
    require(len(real)>0 and len(fake)>0 and n_boot>=50,'Need both labels and >=50 bootstrap draws')
    r=real[rng.integers(0,len(real),size=(n_boot,len(real)))];f=fake[rng.integers(0,len(fake),size=(n_boot,len(fake)))]
    real_delta=(old_labels.astype(float)-new_labels.astype(float))
    fake_delta=-real_delta
    def loss(p):
        p=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
        return -(y*np.log(p)+(1-y)*np.log1p(-p))
    nll_delta=loss(new_scores)-loss(old_scores)
    draws=dict(real_recall_pp=100*real_delta[r].mean(axis=1),fake_recall_pp=100*fake_delta[f].mean(axis=1),
        nll=(nll_delta[r].sum(axis=1)+nll_delta[f].sum(axis=1))/len(y))
    return {key:np.quantile(value,[.025,.975]).tolist() for key,value in draws.items()}


def compare_points(arm,calibration,validation,calibration_records,validation_records,plan,batch,output,device='cuda'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fitted=fit_point(calibration,calibration_records,arm,plan,batch,device);write(output/'v2_calibration.json',fitted)
    control=read(ROOT/plan['controls'][arm]['file'])
    points={'cfe_control':dict(T=control['T'],threshold=control['T_star_default'],source_sha256=plan['controls'][arm]['sha256']),
            'v2_clean_fit':dict(T=fitted['T'],threshold=fitted['threshold'],source_sha256=sha(output/'v2_calibration.json'))}
    summary={};groups=[];errors=[];paired={};rank={}
    for condition,data in validation.items():
        validate_logits(data,validation_records,'val_select',condition,arm,plan,batch)
        logits=torch.tensor(data['logits']).float();y=data['targets'];margin=data['logits'][:,1].astype(float)-data['logits'][:,0].astype(float)
        rank[condition]=dict(raw_margin_auc_pct=100*roc_auc_score(y,margin),raw_margin_ap_pct=100*average_precision_score(y,margin),
            scope='Identical raw logits for both points; temperature cannot create a learned ranking gain. Score AUC can differ through float32 ties.')
        scores={};labels={};summary[condition]={}
        for tag,point in points.items():
            scores[tag]=calibration_probabilities(logits,point['T'],device,batch);labels[tag]=(scores[tag]>=np.float32(point['threshold'])).astype(np.int64)
            summary[condition][tag]=metrics(scores[tag],y,labels[tag])
            np.savez_compressed(output/f'{tag}_{condition}.npz',prob_fake=scores[tag],labels=labels[tag],targets=y,
                sample_ids=data['sample_ids'],families=data['families'],generators=data['generators'],paths=data['paths'],
                T=point['T'],threshold=point['threshold'],point_sha256=point['source_sha256'],checkpoint_sha256=plan['models'][arm]['sha256'],
                split='val_select',condition=condition,precision='fp16',probability_device=device,batch_size=batch,plan_sha256=sha(PLAN))
            for field in ['families','generators']:
                for group in sorted(set(data[field])):
                    mask=data[field]==group;value=diagnostic.summarize_scores(scores[tag][mask],y[mask],labels[tag][mask])
                    groups.append(dict(model=arm,condition=condition,point=tag,group_type=field,group=group,n=value['n'],
                        real_recall_pct=value['real_recall_pct'],fake_recall_pct=value['fake_recall_pct'],p10=value['probability_quantiles'][0],p50=value['probability_quantiles'][1],p90=value['probability_quantiles'][2]))
            for i in np.flatnonzero(labels[tag]!=y):
                errors.append(dict(model=arm,condition=condition,point=tag,sample_id=str(data['sample_ids'][i]),path=str(data['paths'][i]),
                    target=int(y[i]),label=int(labels[tag][i]),prob_fake=float(scores[tag][i])))
        a,b=summary[condition]['v2_clean_fit'],summary[condition]['cfe_control']
        paired[condition]=dict(new_minus_control=dict(real_recall_pp=a['real_recall_pct']-b['real_recall_pct'],fake_recall_pp=a['fake_recall_pct']-b['fake_recall_pct'],nll=a['nll']-b['nll']),
            ci95=paired_operating_ci(y,scores['v2_clean_fit'],scores['cfe_control'],labels['v2_clean_fit'],labels['cfe_control'],n_boot=plan['protocol']['bootstrap']['n'],seed=plan['protocol']['bootstrap']['seed']),
            labels_changed=int((labels['v2_clean_fit']!=labels['cfe_control']).sum()))
    diagnostic.csv_write(output/'group_metrics.csv',groups)
    diagnostic.csv_write(output/'summary.csv',[dict(model=arm,condition=condition,point=tag,
        real_recall_pct=value['real_recall_pct'],fake_recall_pct=value['fake_recall_pct'],fpr_pct=value['fpr_pct'],accuracy_pct=value['accuracy_pct'],
        nll=value['nll'],confidence_ece_argmax_pct=value['confidence_ece_argmax_pct'],score_auc_pct=value['auc_pct'],
        raw_margin_auc_pct=rank[condition]['raw_margin_auc_pct']) for condition,rows in summary.items() for tag,value in rows.items()])
    diagnostic.csv_write(output/'paired_deltas.csv',[dict(condition=condition,metric=metric,delta=value,
        ci95_low=paired[condition]['ci95'][metric][0],ci95_high=paired[condition]['ci95'][metric][1])
        for condition,row in paired.items() for metric,value in row['new_minus_control'].items()])
    if errors:diagnostic.csv_write(output/'errors.csv',errors)
    guards={tag:all(summary[c][tag]['real_recall_pct']>=100*plan['protocol']['diagnostic_real_guard'] for c in summary) for tag in points}
    report=dict(schema='phase7-calibration-transfer-comparison/v1',model=arm,points=points,validation=summary,rank_from_frozen_logits=rank,
        paired=paired,both_conditions_real_guard=guards,experimental_point_only=True,calibration_fit_split='val_calib',validation_split='val_select',
        training_started=False,deployment_replaced=False,G72='unchanged',G73='unchanged pending',limits=plan['limits'])
    write(output/'comparison.json',report);return report


def forward_logits(model,records,indices,condition,data_root,batch):
    from src.datasets.preprocessing import preprocessing_parts
    geometry,normalize,_,_=preprocessing_parts('clip',224,224);parts=[];selected=[]
    boundary=300 if condition=='q30_post_crop' and len(indices)>=300 else None
    for chunk in diagnostic.batch_chunks(indices,batch,boundary):
        images=torch.stack([diagnostic.validation_tensor(data_root/records[i]['path'],condition,geometry,normalize) for i in chunk]).to('cuda')
        with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.float16):z=model(images)
        if isinstance(z,tuple):z=z[0]
        require(z.shape==(len(chunk),2) and torch.isfinite(z).all(),'Invalid forward logits')
        parts.append(z.float().cpu().numpy());selected.extend(records[i] for i in chunk)
        if len(selected)%(batch*20)==0 or len(selected)==len(indices):print(f'{condition}: {len(selected)}/{len(indices)} real images',flush=True)
    return np.concatenate(parts),selected


def export_run(run,backup,status):
    write(run/'status.json',status)
    files=[p for p in run.rglob('*') if p.is_file()]
    require(not any(p.suffix in {'.pth','.pt','.tar'} for p in files),'Weights/data archive must stay outside results')
    for path in files:
        target=backup/path.relative_to(run);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    with zipfile.ZipFile(backup/'calibration_transfer_without_weights.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):archive.write(path,path.relative_to(run))


def run(drive_root,data_root,run_root,backup_root,max_batch_size=16):
    from types import SimpleNamespace
    from PIL import Image
    from src.models.canonical_loader import load_canonical
    from tools.phase7_git_source import git_source_identity
    from tools.phase7_training_pilot import command
    require(torch.cuda.is_available(),'Compatible CUDA GPU required; CPU tests cannot complete this experiment')
    plan,calibration_records,validation_records=inputs()
    drive_root,data_root,run_root,backup_root=(Path(p).resolve() for p in [drive_root,data_root,run_root,backup_root])
    require(1<=max_batch_size<=16,'Set max batch in [1,16] before starting')
    require(run_root!=backup_root and not run_root.is_relative_to(backup_root) and not backup_root.is_relative_to(run_root),'Runtime and backup must be separate')
    for folder in [run_root,backup_root]:require(not folder.exists() or not any(folder.iterdir()),'Preserve old results and choose a new run ID')
    source=git_source_identity(ROOT)
    require(sha(data_root/plan['manifest']['file'])==plan['manifest']['sha256'],'Staged manifest differs')
    require((data_root/'.extracted_sha256').read_text().strip()==plan['archive']['sha256'],'Staging archive differs')
    run_root.mkdir(parents=True,exist_ok=True);backup_root.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',training_started=False,deployment_replaced=False)
    identity=dict(run_id=run_root.name,source=source,plan_sha256=sha(PLAN),gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        precision='fp16',tf32=False,max_batch_size=max_batch_size,created_utc=datetime.now(timezone.utc).isoformat())
    write(run_root/'identity.json',identity);write(run_root/'plan.json',plan)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        reference=run_root/'reference_validation';ref_identity,ref_contract=stage_reference(drive_root,reference,plan)
        checkpoints={};cache=run_root.parent/'checkpoint_cache';cache.mkdir(exist_ok=True)
        for arm,spec in plan['models'].items():
            original=drive_root/spec['drive_path'];require(original.is_file() and sha(original)==spec['sha256'],'Wrong checkpoint: '+str(original))
            target=cache/(spec['sha256']+'.pth')
            if not target.exists():shutil.copy2(original,target)
            require(sha(target)==spec['sha256'],'Checkpoint cache changed; use a fresh runtime');checkpoints[arm]=target
            folder=run_root/arm;folder.mkdir()
            command([sys.executable,'-u','-m','tools.probe_phase7_gpu','--checkpoint',target,'--output',folder/'probe.json',
                '--max_batch_size',max_batch_size,'--expected_checkpoint_sha',spec['sha256']],run_root/'commands.log')
        probes=[read(run_root/arm/'probe.json') for arm in plan['models']]
        require(all(p['status']=='completed' for p in probes),'Incomplete GPU probe')
        batch=min(p['selected_batch_size'] for p in probes)
        mode=choose_validation_mode(ref_identity,ref_contract,identity['gpu'],identity['torch'],identity['cuda'],batch,source['files'])
        write(run_root/'batch_contract.json',dict(batch_size=batch,precision='fp16',probability_device='cuda',same_for_both_models_and_points=True,automatic_midrun_batch_change=False,validation_mode=mode))
        print('Common batch:',batch,'Validation mode:',mode,flush=True)
        for row in calibration_records+(validation_records if mode=='remeasure_both_models' else []):
            path=(data_root/row['path']).resolve();require(path.is_relative_to(data_root) and path.is_file() and sha(path)==row['sha256'],'Missing/changed image: '+row['path'])
            with Image.open(path) as image:image.convert('RGB').load()
        reports={}
        for arm in plan['models']:
            folder=run_root/arm;args=SimpleNamespace(backbone='CLIP:ViT-L/14',clip_path=None,num_classes=2,use_srm=True,use_gating=True,num_vit_adapter=3,num_context_embedding=8)
            model,load_report=load_canonical(checkpoints[arm],args)
            require(load_report['missing']==load_report['unexpected']==0 and load_report['sha256']==plan['models'][arm]['sha256'] and load_report['n_keys']==1200,'Canonical load incomplete/mismatched')
            write(folder/'canonical_load.json',load_report);model=model.to('cuda').eval()
            print('Extracting independent clean val_calib:',arm,flush=True)
            z,rows=forward_logits(model,calibration_records,list(range(len(calibration_records))),'clean',data_root,batch)
            cal=payload(z,rows,'val_calib','clean',arm,plan,batch,'new_forward');np.savez_compressed(folder/'calibration_logits.npz',**cal)
            validation={}
            for condition in plan['protocol']['conditions']:
                if mode=='reuse_frozen_logits':
                    with np.load(reference/arm/f'predictions_{condition}.npz',allow_pickle=False) as old:
                        mapping={r['sha256']:r for r in validation_records};rows=[mapping[sid] for sid in old['sample_ids']];z=old['logits'].copy()
                        baseline=read(diagnostic.PLAN)['operating_point']
                        replay=calibration_probabilities(torch.tensor(z),baseline['T'],'cuda',batch)
                        error=float(np.max(np.abs(replay-old['prob_fake'])))
                        require(error<=1e-6,'Frozen logit replay differs beyond numeric tolerance')
                        write(folder/f'cache_replay_{condition}.json',dict(max_probability_error=error,
                            label_disagreements=int(((replay>=np.float32(baseline['threshold']))!=old['labels']).sum()),
                            scope='Replay of historical baseline scores from frozen logits, not a new backbone forward or deployment acceptance.'))
                    data=payload(z,rows,'val_select',condition,arm,plan,batch,'frozen_diagnostic_logits')
                else:
                    print('Remeasuring validation:',arm,condition,flush=True)
                    indices=list(range(len(validation_records)))
                    if condition=='q30_post_crop':
                        historical=diagnostic.historical_q30_indices(validation_records);indices=historical+[i for i in indices if i not in set(historical)]
                    z,rows=forward_logits(model,validation_records,indices,condition,data_root,batch)
                    data=payload(z,rows,'val_select',condition,arm,plan,batch,'new_forward')
                np.savez_compressed(folder/f'validation_logits_{condition}.npz',**data);validation[condition]=data
            del model;gc.collect();torch.cuda.empty_cache()
            print('Fitting val_calib only and comparing fixed points:',arm,flush=True)
            reports[arm]=compare_points(arm,cal,validation,calibration_records,validation_records,plan,batch,folder/'comparison')
        report=dict(schema='phase7-calibration-transfer-result/v1',status='completed',validation_mode=mode,models=reports,
            training_started=False,deployment_replaced=False,retained_package_sha256=plan['retained_package_sha256'],G72='unchanged: FAIL 6/8',G73='unchanged: pending',limits=plan['limits'])
        write(run_root/'report.json',report)
        status.update(status='completed',files={p.relative_to(run_root).as_posix():sha(p) for p in run_root.rglob('*') if p.is_file() and p!=run_root/'status.json'})
    except Exception as error:
        status.update(status='failed',error=str(error));raise
    finally:export_run(run_root,backup_root,status)
    return dict(status='completed',batch_size=batch,validation_mode=mode,backup=str(backup_root),training_started=False,deployment_replaced=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['drive_root','data_root','run_root','backup_root']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--max_batch_size',type=int,default=16)
    print(run(**vars(parser.parse_args())))
