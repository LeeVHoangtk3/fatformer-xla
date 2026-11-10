import copy
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from tools.run_pilot_checkpoint_eval import validate_plan,common_batch,Steps,safe_extract,stage_datasets,ROOT,PLAN
from tools.compare_pilot_checkpoints import paired_arrays,paired_ci
from tools.calibration_data import sha256_file


def test_frozen_plan_uses_exact_selected_source_and_disjoint_partitions():
    plan=json.loads(PLAN.read_text());partitions=validate_plan(plan)
    assert len(partitions['calibration'])==400 and len(partitions['evaluation'])==1600
    assert not partitions['calibration']&partitions['evaluation']
    assert plan['models']['candidate_p0']['sha256']=='9d97613c5f271801a30ff514326557f262a372abeb4d52ce864c78f20fc1069b'
    assert plan['models']['incumbent_refit']['sha256']=='5cd49252ae94189734cee324f9543aa93f462028c879ac259cc16864b609bd3c'
    assert 'not an exact replay' in plan['reference_scope']


def test_partition_hashes_survive_git_windows_to_linux_checkout():
    plan=json.loads(PLAN.read_text())
    for record in plan['partitions'].values():
        payload=(ROOT/record['file']).read_bytes()
        assert b'\r\n' not in payload
        assert sha256_file(ROOT/record['file'])==record['sha256']


def test_common_batch_fits_both_not_just_the_candidate():
    assert common_batch([dict(status='completed',selected_batch_size=16),dict(status='completed',selected_batch_size=4)])==4
    with pytest.raises(ValueError):common_batch([dict(status='failed',selected_batch_size=16)])


def test_completed_step_reused_only_if_bytes_unchanged(tmp_path,monkeypatch):
    import tools.run_pilot_checkpoint_eval as module
    output=tmp_path/'result.json';calls=[]
    def command(args,log):calls.append(args);output.write_text('{"measured":true}')
    monkeypatch.setattr(module,'command',command)
    step=Steps(tmp_path,{'commit':'fixture'})
    step.run('one','fake.module',['--output',output],[output])
    step.run('one','fake.module',['--output',output],[output])
    assert len(calls)==1
    output.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='output changed'):step.run('one','fake.module',['--output',output],[output])
    assert len(calls)==1


def test_partial_outputs_are_preserved_and_not_adopted(tmp_path):
    output=tmp_path/'partial.json';output.write_text('{}')
    with pytest.raises(ValueError,match='Partial/unbound'):
        Steps(tmp_path,{}).run('one','fake',[],[output])
    assert output.read_text()=='{}'


def archive(path,name,content=b'fixture'):
    with tarfile.open(path,'w') as output:
        info=tarfile.TarInfo(name);info.size=len(content);output.addfile(info,io.BytesIO(content))


def test_unsafe_tar_path_cannot_write_outside_staging(tmp_path):
    p=tmp_path/'bad.tar';archive(p,'../escaped.txt')
    with pytest.raises(ValueError,match='Unsafe'):safe_extract(p,tmp_path/'stage')
    assert not (tmp_path/'escaped.txt').exists()


def test_dataset_archive_hash_is_checked_before_extraction(tmp_path):
    drive=tmp_path/'drive';(drive/'datasets').mkdir(parents=True)
    p=drive/'datasets/test_communityforensics.tar';archive(p,'image.png')
    with pytest.raises(ValueError,match='differs from audited'):
        stage_datasets(drive,tmp_path/'run',{'archives':{p.name:'0'*64}})
    assert not (tmp_path/'run/staging.json').exists()


def predictions():
    return dict(sample_ids=np.array(['a','b','c','d']),targets=np.array([0,0,1,1]),
                subsets=np.array(['s']*4),prob_fake=np.array([.1,.4,.6,.9]))


@pytest.mark.parametrize('problem',['ids','labels','duplicates','nan'])
def test_unmatched_predictions_cannot_produce_comparison(problem):
    a,b=predictions(),predictions()
    if problem=='ids':b['sample_ids']=b['sample_ids'][::-1]
    elif problem=='labels':b['targets']=1-b['targets']
    elif problem=='duplicates':a['sample_ids']=b['sample_ids']=np.array(['a','a','c','d'])
    else:a['prob_fake'][0]=np.nan
    with pytest.raises(ValueError):paired_arrays(a,b)


def test_identical_frozen_models_have_zero_paired_delta_interval():
    p=predictions();y,a,b=paired_arrays(p,p)
    result=paired_ci(y,a,b,n_boot=50)
    assert result=={'auc':[0.,0.],'fake_at_real90':[0.,0.]}


@pytest.mark.parametrize('git_mode',[True,False])
def test_benchmark_explicit_source_hash_and_historical_default(tmp_path,monkeypatch,git_mode):
    import tools.run_phase7_corrected as module
    from tools.phase7_corrected import TRAINING_SHA
    seen=[]
    monkeypatch.setattr(module,'git_source_identity',lambda root:{'delivery':'git'})
    monkeypatch.setattr(module,'verify_bundle',lambda root:{'delivery':'old_bundle'})
    class StopAfterInputCheck(Exception):pass
    def validate(*args,**kwargs):seen.append(kwargs['expected_checkpoint']);raise StopAfterInputCheck
    monkeypatch.setattr(module,'validate_inputs',validate)
    args=SimpleNamespace(output_dir=str(tmp_path/'out'),bundle_root=str(tmp_path),reference_checkpoint='source',
        calibration='cal',calibration_logits='logits',calibration_split='split',evaluation_split='eval')
    if git_mode:args.git_source=True;args.expected_checkpoint_sha='a'*64
    with pytest.raises(StopAfterInputCheck):module.run(args)
    assert seen==['a'*64 if git_mode else TRAINING_SHA]


def test_calibrate_then_full_uses_same_batch_and_verified_step_cache(tmp_path,monkeypatch):
    """CPU orchestration fixture only; does not claim CUDA quality or model verification."""
    import tools.run_pilot_checkpoint_eval as module
    import torch
    drive=tmp_path/'drive';drive.mkdir();run=tmp_path/'run';run.mkdir()
    (run/'staging.json').write_text('{}')
    plan=json.loads(PLAN.read_text())
    for arm,spec in plan['models'].items():
        path=drive/spec['drive_path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(arm.encode())
        spec['sha256']=sha256_file(path)
    local_plan=tmp_path/'plan.json';local_plan.write_text(json.dumps(plan))
    monkeypatch.setattr(module,'PLAN',local_plan)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU-orchestration-fixture')
    monkeypatch.setattr(module,'git_source_identity',lambda root:dict(commit='a'*40,tracked_files_fingerprint='b'*64))
    image=tmp_path/'image.png';image.write_bytes(b'image fixture; never decoded by mock subprocess')
    calib=[(str(image),label,f's{subset}/{label}/{i}.png') for subset in range(5) for label in [0,1] for i in range(40)]
    monkeypatch.setattr(module,'resolve_cfe',lambda *a:(calib,[{}]*1600))
    monkeypatch.setattr(module,'resolve_indist',lambda *a:[{}]*82003)
    monkeypatch.setattr(module,'validate_inputs',lambda *a,**kw:dict(inference_precision='fp16',extraction_batch_size=8,
        probability_device='cuda',probability_batch_size=8))
    calls=[]
    def fake_command(args,log):
        args=[str(x) for x in args];calls.append(args);name=args[args.index('-m')+1]
        def value(flag):return Path(args[args.index(flag)+1])
        def output(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
        if name=='tools.probe_phase7_gpu':
            batch=16 if 'candidate_p0' in str(value('--output')) else 8
            output(value('--output'),dict(status='completed',selected_batch_size=batch))
        elif name=='tools.extract_calibration_logits':value('--output_npz').write_bytes(b'fixture npz')
        elif name=='tools.calibrate_and_select':
            folder=value('--output_dir');arm=args[args.index('--model_name')+1]
            output(folder/(arm+'.json'),dict(temperature_optimization=dict(converged=True)))
            (folder/(arm+'_reliability.png')).write_bytes(b'fixture png')
        elif name=='tools.package_unified_fatformer':
            value('--output').write_bytes(b'fixture package');output(value('--metadata_out'),{})
        elif name=='tools.verify_unified_package':output(value('--output'),{'fixture':'no real model assertion'})
        elif name=='tools.run_phase7_corrected':output(value('--output_dir')/'final_report.json',{'fixture':'not a quality result'})
        elif name=='tools.infer_unified':output(value('--latency_out'),{'fixture':'not a latency measurement'})
        elif name in ['tools.finalize_phase7_corrected','tools.compare_pilot_checkpoints']:output(value('--output'),{'fixture':True})
        else:raise AssertionError(name)
    monkeypatch.setattr(module,'command',fake_command)
    args=SimpleNamespace(drive_root=drive,run_root=run,backup_root=drive/'eval',cfe_root=run/'data/cfe',
        indist_root=run/'data/indist',max_batch_size=16,stage='calibrate')
    module.evaluate(args)
    assert not any('tools.run_phase7_corrected' in call for call in calls)
    args.stage='full';module.evaluate(args)
    benchmarks=[call for call in calls if 'tools.run_phase7_corrected' in call]
    assert len(benchmarks)==2
    assert all(call[call.index('--batch_size')+1]=='8' and '--git_source' in call for call in benchmarks)
    assert {call[call.index('--expected_checkpoint_sha')+1] for call in benchmarks}=={m['sha256'] for m in plan['models'].values()}
    count=len(calls);module.evaluate(args)
    assert len(calls)==count
    assert (drive/'eval/evaluation_results_without_weights.zip').exists()
