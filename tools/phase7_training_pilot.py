"""Two fresh training arms, selected only on fixed CLIP val_select, never on CFE."""
import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src.training.runtime_state import sha256_file, source_sha256
from tools.phase7_git_source import git_source_identity

ROOT = Path(__file__).resolve().parents[1]
ARMS = {'p0_legacy_full': 'legacy_imagenet', 'p1_clip_full': 'clip'}
MAX_SKIP_FRACTION = .05  # A preregistered engineering policy, not a G7.2 quality gate.


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def runtime_config(arm, data_root, assets, folder, backup, batch_size=16, accum=2):
    config = yaml.safe_load((ROOT/'configs/ablation_v2/a4_full.yaml').read_text(encoding='utf-8'))
    config['experiment'].update(name='phase7_improvement_round1_'+arm, seed=42)
    config['model'].update(clip_path=str(assets/'ViT-L-14.pt'),
        checkpoint_init=str(assets/'fatformer_4class_ckpt.pth'),
        teacher_checkpoint=str(assets/'fatformer_4class_ckpt.pth'))
    config['dataset'].update(data_root=str(data_root),
        manifest=str(data_root/'DATASET/manifests/data_manifest_v2.json'),
        isolation_report=str(data_root/'DATASET/logs/phase7/data_isolation_report.json'),
        audit_config=str(data_root/'configs/phase7_data_v2.yaml'),
        degradation_config=str(ROOT/'configs/degradation_v2.yaml'),
        preprocessing=ARMS[arm], validation_preprocessing='clip',
        train_dir='', val_dir='', val_split='val_select')
    config['training'].update(device='cuda', epochs=5, batch_size=batch_size, grad_accum_steps=accum,
        preflight_min_free_gib=0.0,  # Fit is established by actual 100-step backward smoke, not GPU name/size.
        save_dir=str(folder), backup_dir=str(backup), save_name='best.pth',
        history_csv=str(folder/'train_history.csv'),
        calibration_file=str(ROOT/'DATASET/logs/phase7/calibration/baseline.json'))
    config['training'].pop('resume_from', None)
    return config


def check_stability(metrics, max_skip_fraction=MAX_SKIP_FRACTION):
    keys = ['optimizer_steps_expected', 'optimizer_steps_attempted', 'optimizer_steps_applied',
            'optimizer_steps_skipped', 'nonfinite_grad_events', 'grad_norm_finite_count']
    if any(k not in metrics for k in keys):
        raise ValueError('Missing actual optimizer-update evidence; old averages cannot establish stability')
    values = {k: int(metrics[k]) for k in keys}
    if any(float(metrics[k]) != values[k] or values[k] < 0 for k in keys):
        raise ValueError('Invalid update counts')
    n, attempted, applied, skipped, nonfinite, finite = (values[k] for k in keys)
    if n != attempted or applied + skipped != attempted or applied < 1 or finite + nonfinite != attempted:
        raise ValueError('Inconsistent or unusable optimizer updates')
    if not math.isfinite(float(metrics['grad_norm'])) or not math.isfinite(float(metrics['loss_total'])):
        raise ValueError('Non-finite loss/gradient evidence')
    if skipped/attempted > max_skip_fraction:
        raise ValueError(f'Optimizer skip fraction {skipped/attempted:.3%} exceeds policy {max_skip_fraction:.1%}')
    if nonfinite > skipped:
        raise ValueError('A non-finite norm may have been applied; inspect per-update trace')
    return {**values, 'skip_fraction': skipped/attempted, 'policy_max_skip_fraction':max_skip_fraction}


def check_trace(path, phase, epoch=None, expected=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = [r for r in rows if r['phase']==phase and (epoch is None or r['epoch']==epoch)]
    if not rows or any(r['unrecoverable'] or (not r['grad_norm_finite'] and r['applied']) for r in rows):
        raise ValueError('Trace has missing/unrecoverable or applied non-finite updates')
    counts = dict(optimizer_steps_attempted=len(rows), optimizer_steps_applied=sum(r['applied'] for r in rows),
                  optimizer_steps_skipped=sum(not r['applied'] for r in rows),
                  nonfinite_grad_events=sum(not r['grad_norm_finite'] for r in rows))
    if expected and any(int(expected[k])!=v for k,v in counts.items()):
        raise ValueError('Per-update trace does not match reported epoch counters')
    return counts


def command(args, log):
    with Path(log).open('a', encoding='utf-8') as stream:
        stream.write('COMMAND '+json.dumps([str(x) for x in args])+'\n'); stream.flush()
        with subprocess.Popen([str(x) for x in args], cwd=ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONIOENCODING':'utf-8'}) as process:
            for line in process.stdout:
                print(line,end='',flush=True);stream.write(line);stream.flush()
            code=process.wait()
        if code: raise subprocess.CalledProcessError(code,args)


def sync_small(folder, backup):
    backup.mkdir(parents=True,exist_ok=True)
    for path in folder.rglob('*'):
        if path.is_file() and path.suffix in {'.json','.jsonl','.yaml','.csv','.log'}:
            target=backup/path.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)


def summarize(run_root):
    from tools.run_ablation_v2 import verify_run
    output={}
    for arm in ARMS:
        folder=run_root/arm
        identity=read_json(folder/'identity.json')
        verify_run(folder,identity['context'],identity['source_commit'])
        history=list(csv.DictReader((folder/'train_history.csv').open(encoding='utf-8')))
        for row in history:
            check_stability(row)
            check_trace(folder/'optimizer_updates.jsonl','training',int(row['epoch']),row)
        best=torch.load(folder/'best.pth',map_location='cpu',weights_only=False,mmap=True)['metadata']
        output[arm]={k:best[k] for k in ['epoch','clean_r','clean_f','q30_r','q30_f','worst_group']}
        output[arm]['sha256']=sha256_file(folder/'best.pth')
    # Fixed tie-break favors the control; no CFE test values are available here.
    selected=max(ARMS,key=lambda arm:output[arm]['worst_group'])
    report=dict(schema='phase7-training-pilot-selection/v1',selection_split='val_select',
        validation_preprocessing='clip',arms=output,selected=selected,
        tie_break='p0_legacy_full',quality_claim='validation-only; G7.2/G7.3 not evaluated or passed',
        next_step='Freeze selection before new calibration and locked evaluation; use a new unseen test for a generalization claim.')
    write_json(run_root/'selection.json',report)
    return report


def run_pilot(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the real pilot; CPU unit tests do not satisfy this gate')
    if args.batch_size*args.grad_accum_steps != 32:
        raise ValueError('This round fixes effective batch=32; adjust batch and accumulation together before running')
    if args.run_root==args.backup_root or args.run_root.is_relative_to(args.backup_root) or args.backup_root.is_relative_to(args.run_root):
        raise ValueError('Runtime and backup directories must be separate')
    source=git_source_identity(ROOT)
    from tools.run_ablation_v2 import context_for, restore_latest
    args.run_root.mkdir(parents=True,exist_ok=True)
    args.backup_root.mkdir(parents=True,exist_ok=True)
    for directory in [args.run_root,args.backup_root]:
        marker=directory/'source_identity.json'
        if marker.exists() and read_json(marker)!=source:
            raise ValueError('Git source differs from this run; use its original commit or a new run ID')
        write_json(marker,source)
    summary_status={'schema':'phase7-training-pilot-status/v1','status':'running','stage':args.stage,
                    'GPU':torch.cuda.get_device_name(),'quality_gate_pass':None,'source_commit':source['commit']}
    prepared=[]
    try:
        for arm in ARMS:
            folder,backup=args.run_root/arm,args.backup_root/arm
            folder.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
            config=runtime_config(arm,args.data_root,args.assets,folder,backup,args.batch_size,args.grad_accum_steps)
            config_path=folder/'run_config.yaml'
            if config_path.exists() and yaml.safe_load(config_path.read_text(encoding='utf-8'))!=config:
                raise ValueError('Immutable pilot config changed; use a new run ID')
            config_path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
            context=context_for(config_path)
            identity=dict(context=context,source_commit=source['commit'],source_branch='hoang',
                          runner_sha256=sha256_file(__file__),
                          source_fingerprint=source['tracked_files_fingerprint'])
            for directory in [folder,backup]:
                path=directory/'identity.json'
                if path.exists() and read_json(path)!=identity:
                    raise ValueError('Pilot identity changed; do not rebind old logs/checkpoints to new code')
                if not path.exists() and any(directory.glob('*.pth')):
                    raise ValueError('Unidentified checkpoints must not be adopted into a new experiment')
                write_json(path,identity)
            preflight=folder/'preflight.json'
            command([sys.executable,'-m','tools.preflight_phase7_train','--config',config_path,
                     '--output',preflight],folder/'commands.log')
            if not read_json(preflight).get('ready_to_train'):raise ValueError('Preflight did not pass')
            benchmark=folder/'stability_smoke.json'
            # Each attempt has a new trace directory, so old successful traces cannot mask failure.
            from datetime import datetime, timezone
            scratch=folder/('smoke_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
            command([sys.executable,'-m','src.training.train_crossdomain','--config',config_path,
                '--benchmark_steps','100','--benchmark_warmup','5','--output',benchmark,
                '--save_dir',scratch],folder/'commands.log')
            result=read_json(benchmark)
            if result['evidence_level']!='measured' or result['context']!=context:
                raise ValueError('Smoke context differs from training')
            proof=check_stability(result['metrics'])
            check_trace(scratch/'optimizer_updates.jsonl','benchmark',0,result['metrics'])
            write_json(folder/'stability_decision.json',dict(passed=True,policy=proof,
                smoke_sha256=sha256_file(benchmark),trace_sha256=sha256_file(scratch/'optimizer_updates.jsonl'),
                warmup_excluded=True,quality_claim=None))
            sync_small(folder,backup)
            prepared.append((folder,backup,context,config_path,scratch))
        # Both arms must fit and satisfy the declared stability policy before any full training.
        if args.stage=='train':
            if git_source_identity(ROOT)!=source:
                raise ValueError('Source changed after smoke; training has not started')
            for folder,backup,context,config_path,scratch in prepared:
                latest=restore_latest(folder,backup,context)
                # Resume traces must be available too; an epoch average is not enough.
                if latest and not (folder/'optimizer_updates.jsonl').exists() and (backup/'optimizer_updates.jsonl').exists():
                    shutil.copy2(backup/'optimizer_updates.jsonl',folder/'optimizer_updates.jsonl')
                if latest:
                    completed=int(torch.load(latest,map_location='cpu',weights_only=False,mmap=True)['metadata']['epoch'])
                    trace=folder/'optimizer_updates.jsonl'
                    if trace.exists():
                        text=trace.read_text(encoding='utf-8')
                        kept=[line for line in text.splitlines() if json.loads(line)['epoch']<=completed]
                        if len(kept)!=len(text.splitlines()):
                            evidence=folder/'interrupted_traces';evidence.mkdir(exist_ok=True)
                            shutil.copy2(trace,evidence/(scratch.name+'.jsonl'))
                            trace.write_text(''.join(line+'\n' for line in kept),encoding='utf-8')
                cmd=[sys.executable,'-m','src.training.train_crossdomain','--config',config_path]
                if latest:cmd+=['--resume_from',latest]
                command(cmd,folder/'commands.log')
                sync_small(folder,backup)
            summarize(args.run_root)
        summary_status['status']='smoke_passed' if args.stage=='smoke' else 'training_completed_validation_selected'
    except BaseException as exc:
        summary_status.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                              error=str(exc),error_type=type(exc).__name__)
        raise
    finally:
        write_json(args.run_root/'status.json',summary_status)
        sync_small(args.run_root,args.backup_root)
        import zipfile
        archive=args.backup_root/'pilot_results_without_weights.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
            for path in args.run_root.rglob('*'):
                if path.is_file() and path.suffix in {'.json','.jsonl','.yaml','.csv','.log'}:
                    output.write(path,path.relative_to(args.run_root).as_posix())
        print(f'Results: {archive}; selected weights remain in the same Drive run folder.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['smoke','train','summarize'],default='smoke')
    for name in ['data_root','assets','run_root','backup_root']:
        parser.add_argument('--'+name,type=lambda p:Path(p).resolve(),required=True)
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--grad_accum_steps',type=int,default=2)
    args=parser.parse_args()
    if args.stage=='summarize':print(json.dumps(summarize(args.run_root),indent=2))
    else:run_pilot(args)


if __name__=='__main__':main()
