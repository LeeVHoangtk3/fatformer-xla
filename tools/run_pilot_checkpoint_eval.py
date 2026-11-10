"""Calibrate and evaluate frozen P0 vs independently recalibrated incumbent weights."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import zipfile

import torch

from tools.calibration_data import read_partition,sha256_file
from tools.phase7_corrected import resolve_cfe,resolve_indist,write_json,validate_inputs
from tools.phase7_git_source import git_source_identity
from tools.phase7_training_pilot import command

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'configs/phase7_pilot_eval/plan.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_extract(archive,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:*') as source:
        for member in source.getmembers():
            target=(destination/member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not (member.isdir() or member.isfile()):
                raise ValueError('Unsafe archive member: '+member.name)
            if member.isfile() and target.exists():raise ValueError('Duplicate staged file: '+member.name)
        source.extractall(destination,filter='data')


def stage_datasets(drive_root,run_root,plan):
    drive_root,run_root=Path(drive_root),Path(run_root)
    record=dict(schema='phase7-eval-staging/v1',archives={})
    for name,expected in plan['archives'].items():
        path=drive_root/'datasets'/name
        print('Hashing dataset archive:',path,flush=True)
        if not path.is_file() or sha256_file(path)!=expected:raise ValueError('Dataset archive differs from audited historical input: '+name)
        record['archives'][name]=expected
    marker=run_root/'staging.json'
    if marker.exists():
        if read(marker)!=record:raise ValueError('Staged datasets bound to another input; use a new run ID')
        return
    data=run_root/'data'
    if data.exists() and any(data.iterdir()):raise ValueError('Partial dataset staging; preserve it and use a new run ID')
    for name in plan['archives']:
        destination=data/('cfe' if name=='test_communityforensics.tar' else 'indist')
        print('Staging dataset:',name,flush=True)
        safe_extract(drive_root/'datasets'/name,destination)
    write_json(marker,record)


def validate_plan(plan,root=ROOT):
    if plan.get('schema')!='phase7-pilot-evaluation-plan/v1' or list(plan['models'])!=['candidate_p0','incumbent_refit']:
        raise ValueError('Unsupported or changed evaluation plan')
    labels={}
    for tag,counts in [('calibration',(400,200)),('evaluation',(1600,800))]:
        record=plan['partitions'][tag];path=Path(root)/record['file']
        if sha256_file(path)!=record['sha256']:raise ValueError('Frozen partition file changed')
        ids,y=read_partition(path,tag)
        if len(ids)!=counts[0] or int((y==0).sum())!=counts[1] or int((y==1).sum())!=counts[1]:
            raise ValueError('Wrong frozen partition count/class balance')
        labels[tag]=set(ids)
    if labels['calibration']&labels['evaluation']:raise ValueError('Calibration/evaluation overlap')
    for model in plan['models'].values():
        if len(model['sha256'])!=64 or any(x not in '0123456789abcdef' for x in model['sha256']):
            raise ValueError('Invalid frozen source model hash')
        if Path(model['drive_path']).is_absolute() or '..' in Path(model['drive_path']).parts:
            raise ValueError('Model path must stay inside the named Drive root')
    return labels


def common_batch(probes):
    if any(p.get('status')!='completed' or int(p.get('selected_batch_size',0))<1 for p in probes):
        raise ValueError('Every source model needs a completed CUDA memory probe')
    return min(int(p['selected_batch_size']) for p in probes)


class Steps:
    """Reuse completed, byte-verified steps; never infer success from file existence."""
    def __init__(self,root,identity):self.root=Path(root);self.identity=identity
    def run(self,name,module,args,outputs):
        marker=self.root/'steps'/(name+'.json')
        request=dict(identity=self.identity,module=module,args=[str(x) for x in args])
        signature=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        if marker.exists():
            saved=read(marker)
            if saved['signature']!=signature:raise ValueError('Step identity changed; use a new eval run ID')
            for path,expected in saved['files'].items():
                if not (self.root/path).is_file() or sha256_file(self.root/path)!=expected:
                    raise ValueError('Completed step output changed; preserve it and use a new run ID')
            print('Reusing verified step:',name,flush=True);return
        if any(Path(p).exists() and (Path(p).is_file() or any(Path(p).iterdir())) for p in outputs):
            raise ValueError(f'Partial/unbound outputs for {name}; preserve this run and start a new eval ID')
        command([sys.executable,'-u','-m',module,*args],self.root/'commands.log')
        files={}
        for path in map(Path,outputs):
            paths=[path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
            if not paths:raise ValueError('Step produced no required outputs: '+str(path))
            for p in paths:files[p.relative_to(self.root).as_posix()]=sha256_file(p)
        write_json(marker,dict(signature=signature,files=files))


def sync_reports(run,backup):
    """Copy reports, NPZ predictions and 20 verification images, excluding datasets/source weights."""
    for path in run.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(run)
        if rel.parts[0] in {'data','references'} or path.suffix not in {'.json','.jsonl','.csv','.npz','.png','.log'}:continue
        target=backup/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)


def evaluate(args):
    if not torch.cuda.is_available():raise RuntimeError('Real calibration/evaluation requires CUDA')
    if args.run_root==args.backup_root or args.run_root.is_relative_to(args.backup_root) or args.backup_root.is_relative_to(args.run_root):
        raise ValueError('Runtime and Drive backup roots must be separate')
    plan=read(PLAN);validate_plan(plan);source=git_source_identity(ROOT)
    identity=dict(source_commit=source['commit'],source_fingerprint=source['tracked_files_fingerprint'],plan_sha256=sha256_file(PLAN),
        gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=torch.version.cuda,
        precision='fp16',max_batch_size=args.max_batch_size,
        staging_sha256=sha256_file(args.run_root/'staging.json'))
    args.run_root.mkdir(parents=True,exist_ok=True);args.backup_root.mkdir(parents=True,exist_ok=True)
    for folder in [args.run_root,args.backup_root]:
        path=folder/'eval_identity.json'
        if path.exists() and read(path)!=identity:raise ValueError('Evaluation identity changed; use the original source/runtime or a new run ID')
        write_json(path,identity)
    write_json(args.run_root/'source_identity.json',source)
    status=dict(schema='phase7-p0-evaluation-status/v1',status='running',stage=args.stage,source_commit=source['commit'])
    steps=Steps(args.run_root,identity)
    calibration_split=ROOT/plan['partitions']['calibration']['file'];evaluation_split=ROOT/plan['partitions']['evaluation']['file']
    try:
        calibration_records,cfe=resolve_cfe(args.cfe_root,calibration_split,evaluation_split)
        indist=resolve_indist(args.indist_root)
        if (len(calibration_records),len(cfe),len(indist))!=(400,1600,82003):raise ValueError('Dataset count mismatch')
        write_json(args.run_root/'input_counts.json',dict(calibration=400,evaluation=1600,indist_records=82003,
            note='Identity/count check; no claim that all in-dist records are independent image contents.'))
        models={};probes=[]
        for arm,spec in plan['models'].items():
            original=args.drive_root/spec['drive_path'];local=args.run_root/'references'/(arm+'.pth')
            print('Checking frozen weights:',original,flush=True)
            if not original.is_file() or sha256_file(original)!=spec['sha256']:raise ValueError('Missing/mismatched checkpoint: '+str(original))
            if local.exists() and sha256_file(local)!=spec['sha256']:raise ValueError('Staged source model changed')
            local.parent.mkdir(parents=True,exist_ok=True)
            if not local.exists():shutil.copy2(original,local)
            if sha256_file(local)!=spec['sha256']:raise IOError('Staged model checksum mismatch')
            folder=args.run_root/arm;folder.mkdir(exist_ok=True)
            probe=folder/'gpu_probe.json'
            steps.run(arm+'_probe','tools.probe_phase7_gpu',['--checkpoint',local,'--output',probe,
                '--max_batch_size',str(args.max_batch_size),'--expected_checkpoint_sha',spec['sha256']],[probe])
            probes.append(read(probe));models[arm]=dict(source=local,folder=folder,sha256=spec['sha256'])
        batch=common_batch(probes);write_json(args.run_root/'batch_contract.json',dict(batch_size=batch,precision='fp16',
            probability_device='cuda',same_for_both_models=True,automatic_midrun_batch_change=False))
        verify_images=args.run_root/'verification_images';verify_images.mkdir(exist_ok=True)
        chosen=[]
        for subset in sorted({x[2].split('/')[0] for x in calibration_records}):
            for label in [0,1]:
                images=[r for r in calibration_records if r[1]==label and r[2].split('/')[0]==subset][:2]
                if len(images)!=2:raise ValueError('Verification images missing from frozen calibration')
                chosen.extend(images)
        if len(chosen)!=20:raise ValueError('Require 20 calibration verification images')
        for i,(path,label,sid) in enumerate(chosen):
            target=verify_images/(f'{i:02d}_'+Path(path).name)
            if target.exists() and sha256_file(target)!=sha256_file(path):raise ValueError('Verification image changed')
            if not target.exists():shutil.copy2(path,target)
        write_json(args.run_root/'verification_image_ids.json',[dict(sample_id=sid,target=y,source='calibration; functional equivalence only') for path,y,sid in chosen])
        # Complete short calibration/package checks for both models before long benchmarks.
        for arm,model in models.items():
            folder=model['folder'];calib=folder/'calibration';calib.mkdir(exist_ok=True)
            logits=calib/'logits_calib.npz';calibration=calib/(arm+'.json')
            steps.run(arm+'_logits','tools.extract_calibration_logits',['--checkpoint',model['source'],'--data_dir',args.cfe_root,
                '--split_file',calibration_split,'--evaluation_split',evaluation_split,'--output_npz',logits,
                '--device','cuda','--precision','fp16','--batch_size',str(batch)],[logits])
            steps.run(arm+'_calibration','tools.calibrate_and_select',['--checkpoint',model['source'],'--logits_npz',logits,
                '--calibration_split',calibration_split,'--evaluation_split',evaluation_split,'--model_name',arm,
                '--output_dir',calib,'--fpr_target','0.05','--probability_device','cuda'],[calibration,calib/(arm+'_reliability.png')])
            fitted=read(calibration)
            if fitted.get('temperature_optimization',{}).get('converged') is not True:
                raise ValueError('Temperature optimizer has no convergence evidence')
            calibration_proof=validate_inputs(model['source'],calibration,logits,calibration_split,evaluation_split,
                expected_checkpoint=model['sha256'])
            if (calibration_proof['inference_precision']!='fp16' or calibration_proof['extraction_batch_size']!=batch
                    or calibration_proof['probability_device']!='cuda' or calibration_proof['probability_batch_size']!=batch):
                raise ValueError('Calibration precision/batch differs from this paired run')
            write_json(folder/'calibration_preflight.json',calibration_proof)
            package_dir=folder/'package';package_dir.mkdir(exist_ok=True);package=package_dir/'unified.pth'
            steps.run(arm+'_package','tools.package_unified_fatformer',['--checkpoint',model['source'],
                '--calibration',calibration,'--output',package,'--metadata_out',package_dir/'metadata.json'],[package,package_dir/'metadata.json'])
            verification=folder/'verification.json'
            steps.run(arm+'_verification','tools.verify_unified_package',['--checkpoint',package,'--reference',model['source'],
                '--output',verification,'--device','cuda','--precision','fp16','--images',verify_images],[verification])
            model.update(calibration=calibration,logits=logits,package=package,verification=verification)
            target=args.backup_root/arm/'package/unified.pth';target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists() or sha256_file(target)!=sha256_file(package):shutil.copy2(package,target)
            if sha256_file(target)!=sha256_file(package):raise IOError('Drive package copy mismatch')
            write_json(folder/'package_reference.json',dict(path=str(target),sha256=sha256_file(package),source_checkpoint_sha256=model['sha256']))
            sync_reports(args.run_root,args.backup_root)
        if args.stage=='full':
            if git_source_identity(ROOT)!=source:raise ValueError('Code changed after calibration; evaluation stopped')
            for arm,model in models.items():
                folder=model['folder'];benchmark=folder/'benchmark'
                steps.run(arm+'_benchmark','tools.run_phase7_corrected',['--git_source','--bundle_root',ROOT,
                    '--expected_checkpoint_sha',model['sha256'],'--package',model['package'],'--reference_checkpoint',model['source'],
                    '--calibration',model['calibration'],'--calibration_logits',model['logits'],'--calibration_split',calibration_split,
                    '--evaluation_split',evaluation_split,'--cfe_root',args.cfe_root,'--indist_root',args.indist_root,
                    '--verification',model['verification'],'--output_dir',benchmark,'--device','cuda','--precision','fp16',
                    '--batch_size',str(batch),'--num_workers','2'],[benchmark])
                latency=folder/'latency.json'
                steps.run(arm+'_latency','tools.infer_unified',['--checkpoint',model['package'],'--device','cuda',
                    '--benchmark_latency','--latency_out',latency,'--latency_batches',*map(str,sorted({1,8,32,batch}))],[latency])
                acceptance=folder/'acceptance.json'
                steps.run(arm+'_acceptance','tools.finalize_phase7_corrected',['--benchmark',benchmark/'final_report.json',
                    '--verification',model['verification'],'--latency',latency,'--output',acceptance],[acceptance])
                sync_reports(args.run_root,args.backup_root)
            steps.run('matched_comparison','tools.compare_pilot_checkpoints',['--run_root',args.run_root,
                '--output',args.run_root/'comparison.json'],[args.run_root/'comparison.json'])
        status['status']='completed' if args.stage=='full' else 'calibration_and_packaging_completed_test_not_run'
        status['deployment_replaced']=False
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),error_type=type(error).__name__)
        raise
    finally:
        write_json(args.run_root/'status.json',status);sync_reports(args.run_root,args.backup_root)
        archive=args.run_root/'evaluation_results_without_weights.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for path in args.run_root.rglob('*'):
                if not path.is_file() or path==archive:continue
                rel=path.relative_to(args.run_root)
                if rel.parts[0] not in {'data','references'} and path.suffix in {'.json','.csv','.npz','.png','.log'}:
                    z.write(path,rel.as_posix())
        shutil.copy2(archive,args.backup_root/archive.name)
        print('Results/weights on Drive:',args.backup_root,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['drive_root','run_root','backup_root','cfe_root','indist_root']:
        parser.add_argument('--'+name,type=lambda p:Path(p).resolve(),required=True)
    parser.add_argument('--stage',choices=['calibrate','full'],default='full')
    parser.add_argument('--max_batch_size',type=int,default=16)
    args=parser.parse_args()
    if args.max_batch_size<1:parser.error('Batch size must be positive')
    evaluate(args)


if __name__=='__main__':main()
