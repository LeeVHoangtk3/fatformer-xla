"""Exercise the exported Colab workflow without a GPU or a multi-GB backbone."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.unified_fatformer import UnifiedFatFormerXLA
from tools.calibration_data import sha256_file, validate_partitions
from tools.phase7_corrected import (CONTRACT, StrictImageDataset, resolve_indist,
                                   validate_inputs, verify_bundle, write_json)
from tools.run_phase7_corrected import predict_records, metrics, summarize
from tools.finalize_phase7_corrected import finalize


def test_notebook_cells_are_valid_python_and_do_not_clone_old_head():
    notebook=json.loads(Path('notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb').read_text(encoding='utf-8'))
    sources=[]
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            source=''.join(cell['source']);ast.parse(source);sources.append(source)
            assert not cell['outputs'] and cell['execution_count'] is None
    joined='\n'.join(sources)
    assert 'git clone' not in joined and 'git pull' not in joined
    assert 'REQUIRE_T4' not in joined and 'phase7-corrected-20261109-v2' in joined
    assert 'tools.probe_phase7_gpu' in joined and 'MAX_BATCH_SIZE = 16' in joined


def test_missing_indist_or_wrong_count_does_not_become_dummy(tmp_path):
    for label in ['0_real','1_fake']:
        p=tmp_path/'progan'/label;p.mkdir(parents=True)
        Image.new('RGB',(20,30)).save(p/'sample.png')
    records=resolve_indist(tmp_path,expected_counts={'progan':2})
    assert len(records)==2 and {r['label'] for r in records}=={0,1}
    with pytest.raises(ValueError,match='Wrong/missing'):
        resolve_indist(tmp_path,expected_counts={'progan':3})
    with pytest.raises(ValueError,match='Wrong/missing'):
        resolve_indist(tmp_path,expected_counts={'progan':2,'guided':2})


def test_corrupt_image_fails_and_preserves_aspect_ratio(tmp_path):
    path=tmp_path/'image.png'
    Image.new('RGB',(300,100),color='red').save(path)
    dataset=StrictImageDataset([{'path':str(path),'label':1}])
    x,label,index=dataset[0]
    assert x.shape==(3,224,224) and label==1 and index==0
    assert dataset.spatial.transforms[0].size==224
    path.write_bytes(b'corrupt')
    with pytest.raises(Exception):dataset[0]


class Toy(torch.nn.Module):
    def forward(self,x):
        z=x.mean(dim=(1,2,3))
        return torch.stack([-z,z],dim=1)


def test_actual_predict_is_captured_once_and_metrics_are_finite(tmp_path):
    records=[]
    for index in range(4):
        path=tmp_path/f'{index}.png'
        Image.new('RGB',(250,270),color='white' if index%2 else 'black').save(path)
        records.append(dict(path=str(path),label=index%2,sample_id=f's/{index%2}/{index}.png',subset='s'))
    model=UnifiedFatFormerXLA(Toy(),calibration={'schema':'calibration/v1','evidence_level':'measured',
        'checkpoint':{'sha256':'a'*64},'T':1.,'T_star':.5},checkpoint_sha256='a'*64).eval()
    result=predict_records(model,records,'q30',torch.device('cpu'),batch_size=2,workers=0,precision='fp32')
    assert np.array_equal(result['targets'],[0,1,0,1])
    assert result['logits'].shape==(4,2) and len(model.backbone._forward_hooks)==0
    summary,groups=summarize(result,.5,n_boot=50)
    assert summary['accuracy']==100 and summary['balanced_acc']==100 and groups['s']['n_real']==2
    extremes=metrics(np.array([0.,1.]),np.array([0,1]),.5)
    assert np.isfinite(extremes['nll'])


def test_more_than_one_backbone_forward_rejected_and_hook_removed(tmp_path):
    class Bad(torch.nn.Module):
        router_enabled=False
        def __init__(self):super().__init__();self.backbone=Toy()
        def predict(self,x):self.backbone(x);self.backbone(x);return {}
    path=tmp_path/'image.png';Image.new('RGB',(224,224)).save(path)
    model=Bad().eval()
    with pytest.raises(ValueError,match='exactly one'):
        predict_records(model,[dict(path=str(path),label=0,sample_id='s/0/image.png',subset='s')],
                        'clean',torch.device('cpu'),workers=0)
    assert len(model.backbone._forward_hooks)==0


def test_preflight_rejects_old_calibration_checkpoint_and_fpr(tmp_path):
    source=tmp_path/'source.pth';source.write_bytes(b'checkpoint')
    checkpoint=sha256_file(source)
    calib_ids=[f's/{y}/{i}.png' for y in [0,1] for i in range(10)]
    eval_ids=[f's/{y}/{i}.png' for y in [0,1] for i in range(10,12)]
    cal_split,eval_split=tmp_path/'cal.npz',tmp_path/'eval.npz'
    labels=np.array([0]*10+[1]*10)
    np.savez(cal_split,sample_ids=np.array(calib_ids),targets=labels,split='calibration')
    np.savez(eval_split,sample_ids=np.array(eval_ids),targets=np.array([0,0,1,1]),split='evaluation')
    logits=np.tile([[2.,-2.]],(20,1)).astype('float32');logits[10:]=[-2,2]
    cache=tmp_path/'cache.npz'
    np.savez(cache,logits=logits,targets=labels,sample_ids=np.array(calib_ids),split='calibration',
             evidence_level='measured',checkpoint_sha256=checkpoint)
    proof=validate_partitions(calib_ids,labels,cal_split,eval_split)
    cal=dict(schema='calibration/v1',evidence_level='measured',checkpoint={'sha256':checkpoint},
             T=1.,T_star_default=.5,T_star={'0.05':.5},fpr_target=.05,fpr_measured={'0.05':0.},
             partition_validation=proof,n_calib=20,calibration_source_sha256=sha256_file(cache))
    path=tmp_path/'cal.json';write_json(path,cal)
    args=(source,path,cache,cal_split,eval_split)
    assert validate_inputs(*args,expected_checkpoint=checkpoint,expected_counts=(20,4))['fpr_calibration']==0
    with pytest.raises(ValueError,match='Wrong training'):
        validate_inputs(*args,expected_checkpoint='b'*64,expected_counts=(20,4))
    cal['T_star_default']=0.;write_json(path,cal)
    with pytest.raises(ValueError,match='exceeds'):
        validate_inputs(*args,expected_checkpoint=checkpoint,expected_counts=(20,4))
    cal['checkpoint']['sha256']='b'*64;write_json(path,cal)
    with pytest.raises(ValueError,match='Calibration'):
        validate_inputs(*args,expected_checkpoint=checkpoint,expected_counts=(20,4))


def test_bundle_detects_old_or_changed_source(tmp_path):
    names=['tools/run_phase7_corrected.py','tools/phase7_corrected.py','tools/calibration_data.py',
           'tools/infer_unified.py','tools/package_unified_fatformer.py','src/models/unified_fatformer.py',
           'inputs/calibration_400.npz','inputs/calibration_split.npz','inputs/evaluation_split.npz','inputs/anchored_v2.json']
    for name in names:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    manifest={'contract':CONTRACT,'files':{n:sha256_file(tmp_path/n) for n in names}}
    write_json(tmp_path/'BUNDLE_MANIFEST.json',manifest)
    assert verify_bundle(tmp_path)['contract']==CONTRACT
    (tmp_path/names[0]).write_bytes(b'old code')
    with pytest.raises(ValueError,match='changed'):
        verify_bundle(tmp_path)


def test_finalize_accepts_reported_gpu_without_inventing_t4_or_best_condition_pass(tmp_path):
    benchmark,verification,latency,output=[tmp_path/n for n in ['b.json','v.json','l.json','out.json']]
    b=dict(schema='phase7_corrected_benchmark/v1',status='completed',package_sha256='a'*64,
           artifacts={},gates={'G7.2':{'pass':False}})
    v=dict(checkpoint_sha256='a'*64,real_image_acceptance=True,tensors_equal=True)
    l=dict(schema='unified_latency/v3',device='cuda',checkpoint_sha256='a'*64,gpu_name='NVIDIA L4',batches={'1':{'overhead_delta_ms':.2}})
    for p,d in zip([benchmark,verification,latency],[b,v,l]):write_json(p,d)
    result=finalize(benchmark,verification,latency,output)
    assert result['G73']['status']=='pending' and result['G73']['allocated_gpu_overhead']=='pass'
    assert result['G73']['original_t4_reference']=='not_measured_on_T4'
    l['gpu_name']='Tesla T4';write_json(latency,l)
    assert finalize(benchmark,verification,latency,output)['G73']['original_t4_reference']=='pass'
    assert finalize(benchmark,verification,latency,output)['G73']['status']=='pending'
    l['checkpoint_sha256']='b'*64;write_json(latency,l)
    with pytest.raises(ValueError,match='same actual package'):
        finalize(benchmark,verification,latency,output)


def test_memory_probe_falls_back_only_on_oom_and_rejects_unrelated_errors():
    from tools.probe_phase7_gpu import select_fitting_batch
    def attempt(batch):
        if batch>4:raise torch.OutOfMemoryError('fixture only')
        return dict(peak_allocated_gib=3.)
    selected,records=select_fitting_batch(attempt,[16,8,4,2,1])
    assert selected==4 and [r['status'] for r in records]==['out_of_memory','out_of_memory','fits']
    with pytest.raises(RuntimeError,match='No batch fits'):
        select_fitting_batch(attempt,[16,8])
    def broken(batch):raise ValueError('unrelated failure')
    with pytest.raises(ValueError,match='unrelated'):
        select_fitting_batch(broken,[16,1])


def test_latency_large_batch_oom_keeps_smaller_gpu_measurements(monkeypatch):
    import contextlib,itertools
    import tools.infer_unified as inference
    original_randn=torch.randn
    def allocation(*shape,**kwargs):
        if shape[0]>8:raise torch.OutOfMemoryError('fixture allocation')
        return original_randn(*shape)
    monkeypatch.setattr(inference.torch,'randn',allocation)
    monkeypatch.setattr(inference.torch.amp,'autocast',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *_:'NVIDIA L4')
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *_:(20*1024**3,24*1024**3))
    for name in ['reset_peak_memory_stats','synchronize','empty_cache']:
        monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *_:1024**3)
    ticks=itertools.count(step=.001)
    monkeypatch.setattr(inference,'time',SimpleNamespace(perf_counter=lambda:next(ticks)))
    class Fake:
        def backbone(self,x):return x
        def __call__(self,x):return x
    result=inference.benchmark_latency(Fake(),torch.device('cuda'),[1,8,32],n_warmup=1,n_steps=2)
    assert result['batches']['32']['status']=='out_of_memory'
    assert 'overhead_delta_ms' not in result['batches']['32']
    assert result['batches']['1']['status']=='measured' and result['batches']['8']['status']=='measured'
    assert result['overhead_meets_target'] is True and result['original_t4_reference_meets_target'] is None


def test_precision_guards_do_not_silently_use_cpu(tmp_path):
    from tools.extract_calibration_logits import extract_logits
    from tools.verify_unified_package import verify
    with pytest.raises(ValueError,match='CUDA'):
        extract_logits('unused.pth',str(tmp_path),str(tmp_path/'out.npz'),precision='fp16',device='cpu')
    with pytest.raises(ValueError,match='CUDA'):
        verify('unused.pth','unused.pth',str(tmp_path/'proof.json'),precision='fp16',device='cpu')


def test_scoring_backend_preserves_batch_results_and_extreme_nll():
    from tools.calibrate_and_select import calibration_probabilities, compute_nll
    logits=torch.tensor([[2.,-2.],[-3.,3.],[.4,.2]])
    a=calibration_probabilities(logits,1.4,'cpu',1)
    b=calibration_probabilities(logits,1.4,'cpu',8)
    np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-7)
    assert np.isfinite(compute_nll(np.array([0.,1.],dtype='float32'),np.array([0,1])))


def test_packaging_without_git_records_unknown_state(tmp_path,monkeypatch):
    import subprocess
    import tools.package_unified_fatformer as packaging
    sha='a'*64
    state={f'clip_model.visual.transformer.resblocks.{i}.forgery_aware_adapter.w':torch.ones(1) for i in [7,15,23]}
    source=tmp_path/'source.pth'
    torch.save({'model':state,'metadata':{'schema':'unified_fatformer/v1','source_checkpoint_sha256':sha}},source)
    calibration=dict(schema='calibration/v1',evidence_level='measured',checkpoint={'sha256':sha},T=1.,T_star_default=.8,
        n_calib=20,calibration_source_sha256='b'*64,partition_validation=dict(status='verified_disjoint_ids',overlap=0,
        n_calibration=20,calibration_split_sha256='c'*64,evaluation_split_sha256='d'*64,sample_ids_sha256='e'*64))
    cal=tmp_path/'cal.json';write_json(cal,calibration)
    monkeypatch.setattr(packaging,'get_git_commit',lambda:'unknown_commit')
    def no_git(*args,**kwargs):raise subprocess.CalledProcessError(128,args[0])
    monkeypatch.setattr(packaging.subprocess,'check_output',no_git)
    meta=packaging.package_checkpoint(str(source),str(tmp_path/'new.pth'),str(cal),repackage=True)
    assert meta['build']['dirty'] is None
