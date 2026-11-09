"""Prepare audit and pinned input metadata for the hoang Git training workflow."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import torch
import yaml

from src.training.runtime_state import sha256_file, source_sha256
from tools.phase7_training_pilot import ARMS, MAX_SKIP_FRACTION, write_json

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'DATASET/logs/phase7/improvement_round1/preparation'


def prepare(output=DEFAULT_OUTPUT):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    notebooks={p.relative_to(ROOT).as_posix():sha256_file(p) for p in (ROOT/'notebooks').rglob('*.ipynb')}
    manifest_path=ROOT/'DATASET/manifests/data_manifest_v2.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    isolation=json.loads((ROOT/'DATASET/logs/phase7/data_isolation_report.json').read_text(encoding='utf-8'))
    if isolation['metadata']['manifest_sha256']!=sha256_file(manifest_path) or isolation['pass'] is not True:
        raise ValueError('Manifest does not match the passing source isolation audit')
    rows=manifest['records']
    grouped=Counter((r['split'],r['family'],r['generator'],r['source']) for r in rows)
    history=list(csv.DictReader((ROOT/'DATASET/logs/phase7/anchored_v2_run1/train_history.csv').open(encoding='utf-8')))
    audit=dict(schema='phase7-improvement-preparation/v1',
        source_manifest_sha256=sha256_file(manifest_path),records=len(rows),
        by_split=dict(Counter(r['split'] for r in rows)),
        autoregressive=[dict(split=k[0],family=k[1],generator=k[2],source=k[3],n=v)
                       for k,v in sorted(grouped.items()) if k[1]=='autoregressive'],
        findings=[
            'Autoregressive is present, but all 1,200 manifest records use tamingTransformers. This is narrow generator coverage, not evidence that the whole family is missing.',
            'No verified one-to-one mapping from the CFE 5_other_autoregressive subset to these train generators; do not infer generator identity from filenames.',
            'Legacy train/validation uses square stretch + ImageNet normalization; deployed benchmark uses short-edge bicubic + center crop + CLIP normalization.',
            'The preprocessing mismatch is a testable hypothesis, not a proven explanation for the quality gap.',
            'Old NaN averages cannot reveal skipped optimizer updates retrospectively. New trace instrumentation cannot reconstruct missing historical observations.'
        ],
        old_nonfinite_grad_norm_epochs=[int(r['epoch']) for r in history if not math.isfinite(float(r['grad_norm']))],
        local=dict(torch=torch.__version__,cuda_available=torch.cuda.is_available(),
                   sample_image_available=(ROOT/rows[0]['path']).is_file()),
        readiness='local preparation only; staged-image audit, GPU smoke and training have not run')
    write_json(output/'audit.json',audit)
    protocol=dict(schema='phase7-improvement-protocol/v1',round='improvement_round1',
        source_delivery=dict(method='git',branch='hoang',commit='recorded at runtime in source_identity.json'),
        hypothesis='Training with deployed CLIP preprocessing improves held-out val_select worst-group recall relative to legacy training preprocessing.',
        stage='two-arm engineering/validation pilot; not the four-arm Task 7.12 ablation',
        arms=ARMS,validation_preprocessing='clip',seed=42,epochs_each=5,
        default_batch_size=16,default_accumulation=2,effective_batch_size=32,
        frozen_loss_weights=dict(ce_clean=.3,ce_degraded=.3,kd=.7,consistency=.2,feature=.3,guard=.5),
        selection=dict(split='val_select',metric='worst_group',clean_real_min=.9,tie_break='p0_legacy_full',
            frozen_operating_point='DATASET/logs/phase7/calibration/baseline.json'),
        smoke=dict(steps=100,warmup=5,actual_backward=True,max_skip_fraction=MAX_SKIP_FRACTION,
                   applied_nonfinite_gradient_allowed=False,all_skipped_allowed=False,
                   policy_note='Engineering thresholds declared before the new run; these do not modify G7.2.'),
        prohibitions=['No CFE evaluation during pilot or model selection.','No reusing old A4 as the control.',
                      'No changed data/loss/LR/epoch budget between arms.','No overwrite of historical notebooks/results.',
                      'No selecting a successful training seed from CFE test scores.'],
        future='Freeze the validation selection; then calibrate and evaluate it. An unseen test and multiple seeds are needed for stronger generalization claims.')
    write_json(output/'protocol.json',protocol)
    config_folder=output/'planned_configs';config_folder.mkdir(exist_ok=True)
    for arm,profile in ARMS.items():
        config=yaml.safe_load((ROOT/'configs/ablation_v2/a4_full.yaml').read_text(encoding='utf-8'))
        config['experiment'].update(name='phase7_improvement_round1_'+arm,seed=42)
        config['dataset'].update(preprocessing=profile,validation_preprocessing='clip')
        config['training'].update(save_dir=f'DATASET/checkpoints/phase7/improvement_round1/RUN_ID/{arm}',
                                   preflight_min_free_gib=0.0)
        (config_folder/(arm+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    assets={name:sha256_file(ROOT/'DATASET/pretrained'/name) for name in ['ViT-L-14.pt','fatformer_4class_ckpt.pth']}
    baseline=json.loads((ROOT/'DATASET/logs/phase7/calibration/baseline.json').read_text(encoding='utf-8'))
    if assets['fatformer_4class_ckpt.pth']!=baseline['checkpoint']['sha256']:
        raise ValueError('Baseline file differs from the frozen selection operating point')
    inputs={'expected_assets_sha256':assets,
        'training_archive_sha256':(ROOT/'DATASET/logs/phase7/train_crossdomain_v2.tar.sha256').read_text().split()[0],
        'drive_root':'MyDrive/Fatformer','training_archive':'phase7/train_crossdomain_v2.tar',
        'assets':'pretrained/','notes':'Data and large weights are supplied by Drive, not bundled.'}
    write_json(output/'expected_inputs.json',inputs)
    if any(sha256_file(ROOT/name)!=value for name,value in notebooks.items()):
        raise RuntimeError('A protected notebook changed during preparation')
    report=dict(status='prepared_not_gpu_run',source_delivery='git',branch='hoang',
        commit='record actual HEAD at runtime in source_identity.json',
        protected_notebooks=len(notebooks),all_notebooks_unchanged=True,
        gpu_smoke='not_run',training='not_run',quality_improvement='not_measured')
    write_json(output/'preparation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    prepare(parser.parse_args().output)
