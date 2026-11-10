"""Evaluate the actual calibrated package on frozen CFE and all 18 in-dist sets.

Post-hoc confirmation, not a new blind test. No training, refitting, synthetic
fallback, skipped images, cached predictions, or automatic best-reference claim.
"""
import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.calibration_data import sha256_file
from tools.eval_communityforensics import calculate_metrics_v3, compute_bootstrap_ci
from tools.infer_unified import load_unified_model
from tools.phase7_corrected import (CONDITIONS, StrictImageDataset, resolve_cfe, resolve_indist,
                                   validate_inputs, verify_bundle, write_json)
from tools.run_locked_benchmark_phase7 import evaluate_master_gates
from tools.phase7_corrected import TRAINING_SHA
from tools.phase7_git_source import git_source_identity


def metrics(probabilities, targets, threshold):
    # Float64 clipping avoids log(0) when a float32 score is exactly 1.
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets)
    result = calculate_metrics_v3(probabilities, targets, threshold)
    predictions = probabilities >= threshold
    real, fake = targets == 0, targets == 1
    if not real.any() or not fake.any():
        raise ValueError("Every reported dataset/subset must contain real and fake images")
    result['accuracy'] = float((predictions == targets).mean() * 100)
    result['balanced_acc'] = float(((~predictions[real]).mean() + predictions[fake].mean()) * 50)
    result['n_real'], result['n_fake'] = int(real.sum()), int(fake.sum())
    return result


def predict_records(model, records, condition, device, batch_size=8, workers=2, precision='fp16'):
    if model.router_enabled:
        raise ValueError("This corrected fixed-fallback protocol does not approve a fitted router")
    dataset = StrictImageDataset(records, condition)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == 'cuda', drop_last=False)
    parts = {k: [] for k in ['logits','prob_fake','labels','thresholds','w_deg','targets']}
    captured, seen = [], []
    def capture(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    handle = model.backbone.register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for step,(images,targets,indices) in enumerate(loader):
                captured.clear()
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, dtype=torch.float16,
                                        enabled=device.type == 'cuda' and precision == 'fp16'):
                    outputs = model.predict(images)
                if len(captured) != 1:
                    raise ValueError("predict() must make exactly one backbone forward per batch")
                logits = captured[0].detach().float()
                temperature = model.clean_temperature if model.fixed_branch == 'clean' else model.forensic_temperature
                reference = torch.softmax(logits / temperature, -1)[:,1]
                torch.testing.assert_close(outputs['prob_fake'], reference, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(outputs['threshold'],torch.full_like(reference,model.fixed_threshold),rtol=0,atol=0)
                torch.testing.assert_close(outputs['labels'] if 'labels' in outputs else outputs['label'],
                                           (reference >= model.fixed_threshold).long())
                values = dict(logits=logits, prob_fake=outputs['prob_fake'], labels=outputs['label'],
                              thresholds=outputs['threshold'], w_deg=outputs['w_deg'], targets=targets)
                for key,value in values.items():
                    if not torch.isfinite(value).all():
                        raise ValueError(f"Nonfinite prediction: {key}")
                    parts[key].append(value.detach().cpu().numpy())
                seen.extend(indices.tolist())
                if step % 50 == 0:
                    print(f'{condition}: {len(seen)}/{len(records)} images through predict()', flush=True)
    finally:
        handle.remove()
    if seen != list(range(len(records))):
        raise ValueError("Prediction count/order changed; no missing samples are allowed")
    result = {k: np.concatenate(v) for k,v in parts.items()}
    result.update(sample_ids=np.array([r['sample_id'] for r in records]),
                  paths=np.array([r['path'] for r in records]),
                  subsets=np.array([r['subset'] for r in records]))
    return result


def summarize(predictions, threshold, n_boot=1000):
    scores, targets = predictions['prob_fake'],predictions['targets']
    result = metrics(scores,targets,threshold)
    result['ci95'] = compute_bootstrap_ci(scores.astype(np.float64),targets,n_boot=n_boot,seed=42)
    groups = {}
    for name in sorted(set(predictions['subsets'].tolist())):
        mask = predictions['subsets'] == name
        groups[name] = metrics(scores[mask],targets[mask],threshold)
    return result,groups


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory; do not mix or overwrite runs")
    output.mkdir(parents=True,exist_ok=True)
    bundle = git_source_identity(Path(__file__).resolve().parents[1]) if getattr(args,'git_source',False) else verify_bundle(args.bundle_root)
    preflight = validate_inputs(args.reference_checkpoint,args.calibration,args.calibration_logits,
                                args.calibration_split,args.evaluation_split,
                                expected_checkpoint=getattr(args,'expected_checkpoint_sha',None) or TRAINING_SHA)
    if preflight['inference_precision']!=args.precision or preflight['extraction_batch_size']!=args.batch_size:
        raise ValueError('Calibration extraction precision/batch must match the benchmark protocol')
    if preflight['probability_device']!=torch.device(args.device).type or preflight['probability_batch_size']!=args.batch_size:
        raise ValueError('Calibration softmax backend/batch must match the benchmark')
    calib_records,cfe = resolve_cfe(args.cfe_root,args.calibration_split,args.evaluation_split)
    indist = resolve_indist(args.indist_root)
    if len(cfe) != 1600 or len(calib_records) != 400 or len(indist) != 82003:
        raise ValueError("Expected 400/1600 CFE and 82003 in-dist samples")
    source = json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    package = torch.load(args.package,map_location='cpu',weights_only=True,mmap=True)
    if package['metadata']['calibration'] != source:
        raise ValueError("Package and explicit calibration disagree")
    del package
    package_sha = sha256_file(args.package)
    proof = json.loads(Path(args.verification).read_text(encoding='utf-8'))
    if (proof.get('checkpoint_sha256') != package_sha or proof.get('reference_sha256') != preflight['checkpoint_sha256']
            or proof.get('tensors_equal') is not True or proof.get('real_image_acceptance') is not True
            or proof.get('n_images',0) < 20 or str(proof.get('precision','')).lower()!=args.precision):
        raise ValueError("Require matching tensor and >=20 real-image verification before the expensive benchmark")
    write_json(output/'preflight.json',dict(bundle=bundle,inputs=preflight,
               n_cfe=len(cfe),n_indist=len(indist),verification_sha256=sha256_file(args.verification)))
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    model,_ = load_unified_model(args.package,clip_path='NON_EXISTENT_CLIP.pt',device=str(device))
    report = dict(schema='phase7_corrected_benchmark/v1',status='running',
                  scientific_status='post_hoc_confirmation; evaluation previously inspected',
                  package_sha256=package_sha,source_checkpoint_sha256=preflight['checkpoint_sha256'],
                  calibration=preflight,bundle=bundle,
                  environment=dict(python=platform.python_version(),torch=str(torch.__version__),
                    device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                    precision=args.precision if device.type=='cuda' else 'fp32'),
                  preprocess='Resize short side 224 bicubic -> CenterCrop 224 -> degradation -> CLIP normalize',
                  precision_contract='fp16 means backbone autocast; logits/softmax are float32; loaded weights are not converted',
                  protocol_notes=['Actual package.predict; exactly one captured backbone forward per batch.',
                    'Canonical CLIP preprocessing also used on in-dist. Old runner stretched in-dist to 224x224; direct before/after attribution is invalid.',
                    'Identity disjointness is subset/class/filename, not an image-content duplicate audit.',
                    'No generator-family inference from filename keywords. Per-subset results are provided.',
                    'No training or calibration refitting uses evaluation outcomes.'],
                  bootstrap=dict(n=1000,seed=42,unit='image',confidence_level=.95),
                  cfe={},artifacts={})
    write_json(output/'final_report.json',report)
    rows=[]
    for condition in CONDITIONS:
        predictions = predict_records(model,cfe,condition,device,args.batch_size,args.num_workers,args.precision)
        result,groups = summarize(predictions,model.fixed_threshold)
        path = output/f'predictions_cfe_{condition}.npz'
        np.savez_compressed(path,**predictions,condition=condition,package_sha256=package_sha,
                            evidence_level='measured',precision=report['environment']['precision'])
        report['artifacts'][path.name] = sha256_file(path)
        report['cfe'][condition] = dict(metrics=result,per_subset=groups)
        for name,value in groups.items():
            rows.append(dict(condition=condition,subset=name,**{k:v for k,v in value.items() if not isinstance(v,dict)}))
        write_json(output/'final_report.json',report)
        print(condition,result,flush=True)
    predictions = predict_records(model,indist,'clean',device,args.batch_size,args.num_workers,args.precision)
    result,groups = summarize(predictions,model.fixed_threshold)
    path = output/'predictions_indist.npz'
    np.savez_compressed(path,**predictions,condition='clean',package_sha256=package_sha,
                        evidence_level='measured',precision=report['environment']['precision'])
    report['artifacts'][path.name] = sha256_file(path)
    report['indist'] = dict(pooled=result,per_subset=groups,
       macro_acc=float(np.mean([m['balanced_acc'] for m in groups.values()])),
       pooled_real_acc=result['real_acc'],mean_subset_real_acc=float(np.mean([m['real_acc'] for m in groups.values()])))
    report['gates'] = {'G7.2':evaluate_master_gates(
        dict(macro_acc=report['indist']['macro_acc'],real_acc=result['real_acc']),
        report['cfe']['clean']['metrics'],report['cfe']['q30']['metrics'],
        report['cfe']['clean']['per_subset'],0,{})['G7.2'],
        'G7.3':dict(status='pending',single_checkpoint_forward=True,
                   condition_best_delta=None,allocated_gpu_overhead=None,
                   hardware_protocol='any_compatible_cuda_gpu_with_sufficient_memory',
                   note='A source/package equivalence check is not a comparison against the best model by condition.')}
    report['accuracy_definitions']={'cfe_macro_acc':'legacy evaluator name: per-image accuracy',
       'indist_macro_acc':'mean of 18 class-balanced subset accuracies',
       'indist_real_gate':'pooled real accuracy; mean-subset real accuracy reported separately'}
    with open(output/'per_subset_cfe.csv','w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report['artifacts']['per_subset_cfe.csv'] = sha256_file(output/'per_subset_cfe.csv')
    plot_roc(output)
    report['artifacts']['roc.png'] = sha256_file(output/'roc.png')
    report.update(status='completed',finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'final_report.json',report)
    return report


def plot_roc(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,roc_auc_score
    fig,axis=plt.subplots(figsize=(7,6))
    for name in ['cfe_clean','cfe_q30','indist']:
        with np.load(output/f'predictions_{name}.npz',allow_pickle=False) as z:
            fpr,tpr,_=roc_curve(z['targets'],z['prob_fake'])
            auc=roc_auc_score(z['targets'],z['prob_fake'])
        axis.plot(fpr,tpr,label=f'{name} AUC={auc:.4f}')
    axis.plot([0,1],[0,1],linestyle='--',color='grey')
    axis.set(xlabel='False positive rate',ylabel='True positive rate',title='Measured package predictions')
    axis.legend();fig.tight_layout();fig.savefig(output/'roc.png',dpi=160);plt.close(fig)


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['package','reference_checkpoint','calibration','calibration_logits','calibration_split',
                 'evaluation_split','cfe_root','indist_root','verification','output_dir','bundle_root']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--precision',choices=['fp16','fp32'],default='fp16')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--num_workers',type=int,default=2)
    parser.add_argument('--git_source',action='store_true',help='Verify clean hoang Git checkout instead of the historical ZIP manifest')
    parser.add_argument('--expected_checkpoint_sha',default=TRAINING_SHA,help='Explicit expected source hash; historical default is preserved')
    args=parser.parse_args()
    if args.batch_size<1 or args.num_workers<0:parser.error('Invalid batch/workers')
    if Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        parser.error('Use a new empty output directory; existing runs are preserved')
    try:
        run(args)
    except Exception as error:
        path=Path(args.output_dir)
        if path.exists():write_json(path/'failure.json',dict(status='failed',error=str(error)))
        raise


if __name__=='__main__':main()
