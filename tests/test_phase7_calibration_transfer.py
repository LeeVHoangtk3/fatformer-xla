"""Calib/select separation, conservative FPR, paired scoring and safe export."""
import copy
from contextlib import nullcontext
import hashlib
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from tools import phase7_calibration_transfer as transfer
from tools import phase7_validation_diagnostics as diagnostic


def records(split,real=2,fake=2):
    _,rows,_=diagnostic.inputs()
    return [r for r in rows if r['split']==split and r['label']==0][:real]+[r for r in rows if r['split']==split and r['label']==1][:fake]


def fixture_data(rows,split,plan,condition='clean',reverse=False):
    y=np.array([r['label'] for r in rows]);difference=(2*y-1)*(-2 if reverse else 2)
    logits=np.column_stack([np.zeros(len(rows)),difference]).astype(np.float32)
    return transfer.payload(logits,rows,split,condition,'candidate_p0',plan,16,'unit_fixture_not_real_forward')


def test_frozen_calibration_control_and_partition_contract():
    plan,cal,val=transfer.inputs()
    assert len(cal)==len(val)==1080 and sum(r['label']==0 for r in cal)==590
    assert not {r['sha256'] for r in cal}&{r['sha256'] for r in val}
    assert plan['protocol']['false_positive_budget']==29
    assert plan['controls']['candidate_p0']['sha256']==transfer.sha(transfer.ROOT/plan['controls']['candidate_p0']['file'])


@pytest.mark.parametrize('bad',['validation_split','q30_fit','duplicate_id','wrong_checkpoint','synthetic','nan'])
def test_fit_rejects_misidentified_or_noncalibration_inputs(bad):
    plan,_,_=transfer.inputs();rows=records('val_calib');data=fixture_data(rows,'val_calib',plan)
    if bad=='validation_split':data['split']='val_select'
    elif bad=='q30_fit':data['condition']='q30_post_crop'
    elif bad=='duplicate_id':data['sample_ids'][1]=data['sample_ids'][0]
    elif bad=='wrong_checkpoint':data['checkpoint_sha256']='f'*64
    elif bad=='synthetic':data['evidence_level']='synthetic'
    else:data['logits'][0,0]=np.nan
    with pytest.raises(ValueError):transfer.fit_point(data,rows,'candidate_p0',plan,16,device='cpu')


def test_real_calibration_budget_is_29_not_30_and_fit_is_research_only():
    plan,cal,_=transfer.inputs();data=fixture_data(cal,'val_calib',plan)
    y=data['targets'];data['logits'][:,1]=np.where(y==0,np.linspace(-4,1.5,len(y)),np.linspace(-1,4,len(y)))
    original=data['logits'].copy();result=transfer.fit_point(data,cal,'candidate_p0',plan,16,'cpu')
    assert result['false_positive_budget']==29 and result['false_positives']<=29
    assert result['empirical_fpr']<=.05 and result['deployable'] is False
    assert result['schema']!='calibration/v1' and result['temperature_optimization']['converged']
    np.testing.assert_array_equal(data['logits'],original)


def test_validation_scores_cannot_change_the_fit(tmp_path):
    plan,_,_=transfer.inputs();cal=records('val_calib');val=records('val_select');data=fixture_data(cal,'val_calib',plan)
    plan=copy.deepcopy(plan);plan['protocol']['bootstrap']['n']=50
    first=transfer.compare_points('candidate_p0',data,{'clean':fixture_data(val,'val_select',plan)},cal,val,plan,16,tmp_path/'a','cpu')
    second=transfer.compare_points('candidate_p0',data,{'clean':fixture_data(val,'val_select',plan,reverse=True)},cal,val,plan,16,tmp_path/'b','cpu')
    assert first['points']['v2_clean_fit']['T']==second['points']['v2_clean_fit']['T']
    assert first['points']['v2_clean_fit']['threshold']==second['points']['v2_clean_fit']['threshold']
    assert first['rank_from_frozen_logits']['clean']['raw_margin_auc_pct']==100
    assert second['rank_from_frozen_logits']['clean']['raw_margin_auc_pct']==0
    assert not first['deployment_replaced'] and not first['training_started']


def test_paired_bootstrap_has_correct_direction_and_zero_for_identical_points():
    y=np.array([0,0,1,1]);old=np.array([1,1,0,0]);new=1-old
    result=transfer.paired_operating_ci(y,np.array([.1,.1,.9,.9]),np.array([.9,.9,.1,.1]),new,old,50)
    assert result['real_recall_pp']==[100,100] and result['fake_recall_pp']==[100,100]
    assert result['nll'][1]<0
    zero=transfer.paired_operating_ci(y,np.full(4,.5),np.full(4,.5),old,old,50)
    assert all(value==[0,0] for value in zero.values())


@pytest.mark.parametrize('difference',[None,'gpu','torch','cuda','batch'])
def test_cache_reuse_requires_same_environment_and_batch(difference):
    ref={'gpu':'L4','torch':'torch_a','cuda':'13'};contract={'batch_size':16}
    args=dict(gpu='L4',torch_version='torch_a',cuda_version='13',batch=16)
    if difference=='gpu':args['gpu']='another compatible GPU'
    elif difference=='torch':args['torch_version']='other'
    elif difference=='cuda':args['cuda_version']='other'
    elif difference=='batch':args['batch']=4
    assert transfer.choose_validation_mode(ref,contract,**args)==('reuse_frozen_logits' if difference is None else 'remeasure_both_models')


def test_reference_content_hash_is_checked_before_it_can_be_used(tmp_path):
    folder=tmp_path/'source';folder.mkdir();(folder/'data.json').write_bytes(b'bad')
    plan={'diagnostic_reference':{'drive_path':'source','files':{'data.json':{'sha256':hashlib.sha256(b'old').hexdigest(),'bytes':3}}}}
    with pytest.raises(ValueError,match='SHA256'):transfer.stage_reference(tmp_path,tmp_path/'copy',plan)


def test_changed_inference_source_forces_new_validation_forward():
    ref={'gpu':'L4','torch':'same','cuda':'same','source':{'files':{'src/model.py':'original'}}}
    args=(ref,{'batch_size':16},'L4','same','same',16)
    assert transfer.choose_validation_mode(*args,current_source_files={'src/model.py':'original','tools/new_runner.py':'new'})=='reuse_frozen_logits'
    assert transfer.choose_validation_mode(*args,current_source_files={'src/model.py':'changed'})=='remeasure_both_models'


def test_reference_zip_duplicate_member_is_rejected(tmp_path):
    folder=tmp_path/'source';folder.mkdir()
    with zipfile.ZipFile(folder/'validation_diagnostics_without_weights.zip','w') as archive:
        archive.writestr('identity.json',b'{}')
        with pytest.warns(UserWarning):archive.writestr('identity.json',b'{}')
    plan={'diagnostic_reference':{'drive_path':'source','files':{'identity.json':{'sha256':hashlib.sha256(b'{}').hexdigest(),'bytes':2}}}}
    with pytest.raises(ValueError,match='duplicate'):transfer.stage_reference(tmp_path,tmp_path/'copy',plan)


def test_cpu_does_not_create_a_measured_run(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='CUDA'):transfer.run(tmp_path,tmp_path,tmp_path/'run',tmp_path/'backup')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('mode,fails',[('reuse_frozen_logits',False),('remeasure_both_models',False),('reuse_frozen_logits',True)])
def test_runtime_success_failure_and_export_without_weights(tmp_path,monkeypatch,mode,fails):
    """CPU fixtures mock CUDA APIs; only tests orchestration, not real GPU behavior."""
    import tools.phase7_git_source as git_source
    import tools.phase7_training_pilot as pilot
    import src.models.canonical_loader as canonical
    plan=copy.deepcopy(transfer.read(transfer.PLAN));plan['protocol']['bootstrap']['n']=50
    cal=copy.deepcopy(records('val_calib'));val=copy.deepcopy(records('val_select'))
    data_root=tmp_path/'data';data_root.mkdir();drive=tmp_path/'drive'
    for i,row in enumerate(cal+val):
        path=data_root/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(230,240),(20*i,40,100)).save(path);row['sha256']=transfer.sha(path)
    transfer.write(data_root/plan['manifest']['file'],{'records':cal+val});plan['manifest']['sha256']=transfer.sha(data_root/plan['manifest']['file'])
    (data_root/'.extracted_sha256').write_text(plan['archive']['sha256'])
    for arm,spec in plan['models'].items():
        path=drive/spec['drive_path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(arm.encode());spec['sha256']=transfer.sha(path)
    before={str(p):transfer.sha(p) for p in drive.rglob('*.pth')}
    monkeypatch.setattr(transfer,'inputs',lambda:(plan,cal,val))
    monkeypatch.setattr(git_source,'git_source_identity',lambda root:{'commit':'test-fixture-only','files':{}})
    monkeypatch.setattr(diagnostic,'historical_q30_indices',lambda records:[0,2])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU fixture')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.amp,'autocast',lambda *a,**k:nullcontext())
    original_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        if args and str(args[0]).startswith('cuda'):args=('cpu',*args[1:])
        return original_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    def reference(drive_root,destination,plan):
        destination.mkdir(parents=True);transfer.write(destination/'status.json',{'status':'fixture-only'})
        for arm in plan['models']:
            (destination/arm).mkdir()
            for condition in plan['protocol']['conditions']:
                old=transfer.payload(np.zeros((len(val),2),np.float32),val,'val_select',condition,arm,plan,16,'unit_fixture')
                np.savez_compressed(destination/arm/f'predictions_{condition}.npz',**old,prob_fake=np.full(len(val),.5,np.float32),labels=np.zeros(len(val),dtype=int))
        return {'gpu':'CPU fixture' if mode=='reuse_frozen_logits' else 'old GPU','torch':str(torch.__version__),'cuda':torch.version.cuda},{'batch_size':16}
    monkeypatch.setattr(transfer,'stage_reference',reference)
    def command(args,log):transfer.write(args[args.index('--output')+1],{'status':'completed','selected_batch_size':16})
    monkeypatch.setattr(pilot,'command',command)
    calls=[]
    class Model:
        def to(self,device):return self
        def eval(self):return self
        def __call__(self,images):
            calls.append(len(images));return torch.full((len(images),2),float('nan') if fails else 0.)
    monkeypatch.setattr(canonical,'load_canonical',lambda path,args:(Model(),{'missing':0,'unexpected':0,'sha256':transfer.sha(path),'n_keys':1200}))
    run=tmp_path/'runtime/run1';backup=drive/'results/run1'
    if fails:
        with pytest.raises(ValueError,match='forward logits'):transfer.run(drive,data_root,run,backup)
    else:
        result=transfer.run(drive,data_root,run,backup)
        assert result['validation_mode']==mode
        assert len(calls)==(2 if mode=='reuse_frozen_logits' else 6)
        report=transfer.read(backup/'report.json');assert not report['deployment_replaced'] and not report['training_started']
        status=transfer.read(backup/'status.json');assert 'reference_validation/status.json' in status['files']
        assert all(transfer.sha(backup/name)==digest for name,digest in status['files'].items())
    assert transfer.read(backup/'status.json')['status']==('failed' if fails else 'completed')
    with zipfile.ZipFile(backup/'calibration_transfer_without_weights.zip') as archive:
        assert not any(name.endswith(('.pth','.pt','.tar')) for name in archive.namelist())
    assert before=={str(p):transfer.sha(p) for p in drive.rglob('*.pth')}
