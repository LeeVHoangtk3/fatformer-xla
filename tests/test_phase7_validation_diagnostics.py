"""Reject stale/misidentified validation evidence; verify JPEG parity with training."""
import copy
from contextlib import nullcontext
import json

import numpy as np
from PIL import Image
import pytest
import torch

from tools import phase7_validation_diagnostics as diagnostic


def test_frozen_split_and_historical_q30_are_reconstructed():
    plan,rows,val=diagnostic.inputs()
    assert len(rows)==14400 and len(val)==1080
    assert sum(r['label']==0 for r in val)==590
    indices=diagnostic.historical_q30_indices(val)
    assert len(indices)==len(set(indices))==300
    assert all(val[i]['label']==0 for i in indices[:150])
    assert all(val[i]['label']==1 for i in indices[150:])
    assert sum(val[i]['family']=='autoregressive' for i in indices)==33
    assert plan['operating_point']['threshold']==.9127


@pytest.mark.parametrize('bad',['duplicate_content','label','path','split'])
def test_invalid_manifest_evidence_is_rejected(bad):
    _,rows,_=diagnostic.inputs();rows=copy.deepcopy(rows[:2])
    if bad=='duplicate_content':rows[1]['sha256']=rows[0]['sha256'];rows[1]['split']='val_select'
    elif bad=='label':rows[0]['label']=0
    elif bad=='path':rows[0]['path']='../escape.png'
    else:rows[0]['split']='evaluation'
    with pytest.raises(ValueError):diagnostic.validate_records(rows)


def test_q30_matches_training_resize_encode_normalize_order(tmp_path):
    from src.datasets.preprocessing import preprocessing_parts
    from src.training.selection import build_q30_cache
    from types import SimpleNamespace
    geometry,normalize,mean,std=preprocessing_parts('clip',224,224)
    records=[]
    for i,label in enumerate([0,1]):
        path=tmp_path/f'{i}.png'
        pixels=np.random.default_rng(i).integers(0,256,(300,410,3),dtype=np.uint8)
        Image.fromarray(pixels).save(path);records.append(dict(path=str(path),label=label))
    dataset=SimpleNamespace(records=records,eval_geometry=geometry,normalization_mean=mean,normalization_std=std,allow_dummy=False)
    # build_q30_cache calls len(dataset), _source reads its records.
    class Dataset:
        def __len__(self):return len(self.records)
    original=Dataset();original.__dict__.update(dataset.__dict__)
    cache=build_q30_cache(original,max_samples=2)
    for record,(expected,label) in zip(records,cache):
        actual=diagnostic.validation_tensor(record['path'],'q30_post_crop',geometry,normalize)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        assert label==record['label']


def test_q30_historical_boundary_is_not_mixed_with_extra_validation_images():
    chunks=list(diagnostic.batch_chunks(list(range(1080)),16,300))
    assert sum(len(c) for c in chunks)==1080
    assert chunks[18]==list(range(288,300))
    assert chunks[19]==list(range(300,316))


def test_rank_and_operating_point_metrics_are_distinct():
    # Perfect ordering can still miss every fake at a conservative fixed threshold.
    result=diagnostic.summarize_scores([.1,.2,.7,.8],[0,0,1,1],[0,0,0,0])
    assert result['auc_pct']==100 and result['fake_recall_pct']==0
    assert result['real_recall_pct']==100
    one_class=diagnostic.summarize_scores([.7,.8],[1,1],[0,0])
    assert 'auc_pct' not in one_class and one_class['real_recall_pct'] is None


def prediction_fixture(val,plan,arm,condition):
    indices=list(range(len(val)))
    if condition=='q30_post_crop':
        chosen=diagnostic.historical_q30_indices(val);indices=chosen+[i for i in indices if i not in set(chosen)]
    selected=[val[i] for i in indices];targets=np.array([r['label'] for r in selected])
    logits=np.stack([np.zeros(len(val)),targets*3.-1.],axis=1).astype(np.float32)
    p=torch.softmax(torch.tensor(logits)/plan['operating_point']['T'],-1)[:,1].numpy()
    chosen={val[i]['sha256'] for i in diagnostic.historical_q30_indices(val)}
    return dict(logits=logits,prob_fake=p,labels=(p>=plan['operating_point']['threshold']).astype(int),targets=targets,
        sample_ids=np.array([r['sha256'] for r in selected]),families=np.array([r['family'] for r in selected]),
        generators=np.array([r['generator'] for r in selected]),paths=np.array([r['path'] for r in selected]),
        historical_q30=np.array([r['sha256'] in chosen for r in selected]),split='val_select',evidence_level='measured',
        condition=condition,checkpoint_sha256=plan['models'][arm]['sha256'],manifest_sha256=plan['manifest']['sha256'],
        plan_sha256=diagnostic.sha(diagnostic.PLAN),precision='fp16')


@pytest.mark.parametrize('bad',['checkpoint','label','image_id','logit','synthetic','membership'])
def test_corrupt_per_image_results_are_rejected(tmp_path,bad):
    plan,_,val=diagnostic.inputs();data=prediction_fixture(val,plan,'candidate_p0','clean')
    if bad=='checkpoint':data['checkpoint_sha256']='f'*64
    elif bad=='label':data['labels'][0]=1-data['labels'][0]
    elif bad=='image_id':data['sample_ids'][0]=data['sample_ids'][1]
    elif bad=='logit':data['logits'][0,0]+=10
    elif bad=='synthetic':data['evidence_level']='synthetic'
    else:data['historical_q30'][0]=not data['historical_q30'][0]
    path=tmp_path/'fixture.npz';np.savez_compressed(path,**data)
    with pytest.raises(ValueError):diagnostic.load_predictions(path,val,plan,'candidate_p0','clean')


def test_analysis_produces_group_and_error_evidence_without_changing_predictions(tmp_path):
    plan,_,val=diagnostic.inputs()
    for arm in plan['models']:
        (tmp_path/arm).mkdir()
        for condition in plan['protocol']['conditions']:
            np.savez_compressed(tmp_path/arm/f'predictions_{condition}.npz',**prediction_fixture(val,plan,arm,condition))
    before={str(p):diagnostic.sha(p) for p in tmp_path.rglob('*.npz')}
    result=diagnostic.analyze(tmp_path)
    assert result['metrics']['candidate_p0']['clean']['auc_pct']==100
    assert result['metrics']['candidate_p0']['q30_post_crop']['historical_300']['n']==300
    assert (tmp_path/'analysis/errors.csv').is_file()
    assert (tmp_path/'analysis/group_metrics.csv').is_file()
    with (tmp_path/'analysis/group_metrics.csv').open(encoding='utf-8') as stream:
        import csv
        rows=list(csv.DictReader(stream))
    ar=next(r for r in rows if r['arm']=='candidate_p0' and r['condition']=='clean' and r['group']=='autoregressive')
    assert float(ar['auc_vs_shared_real_pct'])==100 and int(ar['shared_real_reference_n'])==590
    assert result['training_started'] is result['deployment_replaced'] is False
    assert before=={str(p):diagnostic.sha(p) for p in tmp_path.rglob('*.npz')}


def test_cpu_cannot_be_reported_as_real_gpu_validation(tmp_path,monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'):
        diagnostic.collect(tmp_path,tmp_path,tmp_path/'run',tmp_path/'backup')
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('forward_fails',[False,True])
def test_collect_orchestration_keeps_weights_out_of_export_and_preserves_failed_run(tmp_path,monkeypatch,forward_fails):
    """Tiny CPU fixture mocks GPU calls; this does not validate real CUDA execution."""
    import zipfile
    import tools.phase7_git_source as git_source
    import tools.phase7_training_pilot as pilot
    import src.models.canonical_loader as canonical
    plan=copy.deepcopy(diagnostic.read(diagnostic.PLAN));drive=tmp_path/'drive';data=tmp_path/'data';data.mkdir()
    val=[]
    for i in range(4):
        relative=f'DATASET/raw_v2/test/{i}.png';path=data/relative;path.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(240,280),(i*40,50,100)).save(path)
        val.append(dict(path=relative,sha256=diagnostic.sha(path),label=i//2,family='real' if i<2 else 'autoregressive',generator='real' if i<2 else 'fixture'))
    manifest=data/plan['manifest']['file'];diagnostic.write(manifest,{'records':val});plan['manifest']['sha256']=diagnostic.sha(manifest)
    (data/'.extracted_sha256').write_text(plan['archive']['sha256'])
    for arm,spec in plan['models'].items():
        path=drive/spec['drive_path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(arm.encode())
        spec['sha256']=diagnostic.sha(path)
    original_weights={str(p):diagnostic.sha(p) for p in drive.rglob('*.pth')}
    monkeypatch.setattr(diagnostic,'inputs',lambda:(plan,val,val))
    monkeypatch.setattr(diagnostic,'historical_q30_indices',lambda records:[0,2])
    monkeypatch.setattr(git_source,'git_source_identity',lambda root:{'commit':'test-fixture-only'})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda:'CPU fixture pretending GPU for orchestration test')
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.amp,'autocast',lambda *a,**k:nullcontext())
    tensor_to=torch.Tensor.to
    def cpu_to(self,*args,**kwargs):
        if args and str(args[0]).startswith('cuda'):args=('cpu',*args[1:])
        return tensor_to(self,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    def command(args,log):
        target=args[args.index('--output')+1]
        diagnostic.write(target,{'selected_batch_size':2 if 'candidate_p0' in str(target) else 1})
    monkeypatch.setattr(pilot,'command',command)
    class Model:
        def to(self,device):return self
        def eval(self):return self
        def __call__(self,images):return torch.full((len(images),2),float('nan') if forward_fails else 0.)
    monkeypatch.setattr(canonical,'load_canonical',lambda *args:(Model(),{'missing':0,'unexpected':0}))
    run=tmp_path/'runtime/run1';backup=drive/'diagnostics/run1'
    if forward_fails:
        with pytest.raises(ValueError,match='Invalid measured logits'):diagnostic.collect(drive,data,run,backup,max_batch_size=2)
    else:
        result=diagnostic.collect(drive,data,run,backup,max_batch_size=2)
        assert result['status']=='completed' and result['batch_size']==1
        assert (backup/'analysis/diagnosis.json').is_file()
    status=diagnostic.read(backup/'status.json')
    assert status['status']==('failed' if forward_fails else 'completed')
    assert not status['training_started'] and not status['deployment_replaced']
    with zipfile.ZipFile(backup/'validation_diagnostics_without_weights.zip') as archive:
        assert 'status.json' in archive.namelist()
        assert not any(name.endswith('.pth') or 'checkpoint_cache' in name for name in archive.namelist())
    assert original_weights=={str(p):diagnostic.sha(p) for p in drive.rglob('*.pth')}
