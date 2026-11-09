import copy
import math
from pathlib import Path

import pytest

from tools.phase7_training_pilot import runtime_config,check_stability,check_trace,run_pilot


def good_metrics():
    return dict(optimizer_steps_expected=50,optimizer_steps_attempted=50,optimizer_steps_applied=49,
        optimizer_steps_skipped=1,nonfinite_grad_events=1,grad_norm_finite_count=49,
        loss_total=.7,grad_norm=3.)


def test_configs_change_only_train_profile_and_arm_identity(tmp_path):
    base=runtime_config('p0_legacy_full',tmp_path,tmp_path,tmp_path/'run',tmp_path/'backup')
    candidate=runtime_config('p1_clip_full',tmp_path,tmp_path,tmp_path/'run',tmp_path/'backup')
    assert base['dataset']['validation_preprocessing']==candidate['dataset']['validation_preprocessing']=='clip'
    assert base['dataset']['val_split']=='val_select'
    assert base['dataset']['preprocessing']=='legacy_imagenet' and candidate['dataset']['preprocessing']=='clip'
    for config in [base,candidate]:
        config['experiment'].pop('name');config['dataset'].pop('preprocessing')
    assert base==candidate
    assert base['training']['epochs']==5 and base['experiment']['seed']==42


@pytest.mark.parametrize('change',['old_log','all_skipped','too_many_skips','inconsistent','applied_nonfinite','nan_loss'])
def test_smoke_rejects_unproven_or_unstable_updates(change):
    metrics=good_metrics()
    if change=='old_log':metrics.pop('optimizer_steps_applied')
    elif change=='all_skipped':metrics.update(optimizer_steps_applied=0,optimizer_steps_skipped=50)
    elif change=='too_many_skips':metrics.update(optimizer_steps_applied=45,optimizer_steps_skipped=5)
    elif change=='inconsistent':metrics['optimizer_steps_attempted']=49
    elif change=='applied_nonfinite':metrics.update(nonfinite_grad_events=2,grad_norm_finite_count=48)
    else:metrics['loss_total']=float('nan')
    with pytest.raises(ValueError):check_stability(metrics)


def test_small_logged_amp_recovery_is_not_automatically_a_quality_failure():
    proof=check_stability(good_metrics())
    assert proof['skip_fraction']==.02


def test_cpu_never_claims_actual_pilot_pass(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'):
        run_pilot(SimpleNamespace(run_root=tmp_path))


def test_trace_detects_nonfinite_applied_even_when_averages_look_good(tmp_path):
    import json
    path=tmp_path/'trace.jsonl'
    path.write_text(json.dumps(dict(phase='benchmark',epoch=0,unrecoverable=False,applied=True,
                                   grad_norm_finite=False))+'\n')
    with pytest.raises(ValueError,match='non-finite'):check_trace(path,'benchmark')


def test_second_arm_smoke_failure_prevents_all_full_training(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    import tools.phase7_training_pilot as pilot
    import tools.run_ablation_v2 as legacy
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'unit-test-fixture')
    monkeypatch.setattr(pilot,'git_source_identity',lambda root:dict(commit='unit-test',tracked_files_fingerprint='fixture'))
    monkeypatch.setattr(legacy,'context_for',lambda p:{'test':'orchestration-only'})
    full_train=[]
    def simulated_command(args,log):
        args=[str(a) for a in args]
        if 'tools.preflight_phase7_train' in args:
            pilot.write_json(args[args.index('--output')+1],{'ready_to_train':True})
        elif '--benchmark_steps' in args:
            if 'p1_clip_full' in str(log):raise RuntimeError('simulated OOM in second arm')
            scratch=Path(args[args.index('--save_dir')+1]);scratch.mkdir(parents=True)
            events=[dict(phase='benchmark',epoch=0,unrecoverable=False,applied=i!=0,
                         grad_norm_finite=i!=0) for i in range(50)]
            (scratch/'optimizer_updates.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            pilot.write_json(args[args.index('--output')+1],dict(evidence_level='measured',
                context={'test':'orchestration-only'},metrics=good_metrics()))
        else:full_train.append(args)
    monkeypatch.setattr(pilot,'command',simulated_command)
    args=SimpleNamespace(run_root=tmp_path/'run',backup_root=tmp_path/'backup',
        data_root=tmp_path/'data',assets=tmp_path/'assets',batch_size=16,grad_accum_steps=2,stage='train')
    with pytest.raises(RuntimeError,match='simulated OOM'):pilot.run_pilot(args)
    assert full_train==[]
    assert pilot.read_json(args.run_root/'status.json')['status']=='failed'
    assert (args.backup_root/'pilot_results_without_weights.zip').exists()
