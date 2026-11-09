"""Execute from a Git checkout of hoang in a NEW Colab notebook.

Defaults: smoke only. To train after reviewing a successful smoke, set
PILOT_STAGE='train' and keep the same PILOT_RUN_ID in the cell/session.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

from google.colab import drive, files
import torch

PILOT_STAGE = globals().get('PILOT_STAGE', 'smoke')
PILOT_RUN_ID = globals().get('PILOT_RUN_ID') or datetime.now(timezone.utc).strftime('pilot_%Y%m%d_%H%M%S')
PILOT_BATCH_SIZE = globals().get('PILOT_BATCH_SIZE', 16)
PILOT_GRAD_ACCUM = globals().get('PILOT_GRAD_ACCUM', 2)
assert PILOT_STAGE in ('smoke','train') and re.fullmatch(r'[A-Za-z0-9_-]+',PILOT_RUN_ID)
assert torch.cuda.is_available(), 'Select a CUDA GPU runtime first; CPU is not a real training smoke.'
assert PILOT_BATCH_SIZE * PILOT_GRAD_ACCUM == 32, 'Keep effective batch=32 for both arms.'

REPO=Path(globals().get('PILOT_REPO','/content/fatformer-xla')).resolve()
sys.path.insert(0,str(REPO))
from tools.phase7_git_source import git_source_identity
source=git_source_identity(REPO)
PILOT_COMMIT=globals().get('PILOT_COMMIT') or source['commit']
assert source['commit']==PILOT_COMMIT, 'Git commit changed; use the pinned commit or a new run ID.'
print('Git branch: hoang; exact commit:',PILOT_COMMIT)
drive.mount('/content/drive')
DRIVE=Path('/content/drive/MyDrive/Fatformer')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


# Keep Colab's matching torch/torchvision installation. Install only missing support packages.
packages={'yaml':'pyyaml','pytest':'pytest','ftfy':'ftfy','regex':'regex','tqdm':'tqdm',
          'sklearn':'scikit-learn','scipy':'scipy','matplotlib':'matplotlib','tabulate':'tabulate'}
missing=[package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)

expected=json.loads((REPO/'DATASET/logs/phase7/improvement_round1/preparation/expected_inputs.json').read_text())
STAGE=Path('/content/phase7_pilot_data')
archive_path=DRIVE/expected['training_archive']
assert archive_path.is_file(), f'Missing original audited data archive: {archive_path}'
assert digest(archive_path)==expected['training_archive_sha256'], 'Training archive differs from audited source.'
STAGE.mkdir(parents=True,exist_ok=True)
marker=STAGE/'.extracted_sha256'
if not marker.exists() or marker.read_text().strip()!=expected['training_archive_sha256']:
    assert not any(STAGE.iterdir()), 'Data staging contains another/incomplete extraction; use a fresh runtime.'
    assert hasattr(tarfile,'data_filter'), 'Use a Python runtime supporting tarfile data extraction filters.'
    with tarfile.open(archive_path) as archive:archive.extractall(STAGE,filter='data')
    marker.write_text(expected['training_archive_sha256'])

ASSETS=Path('/content/phase7_pilot_assets');ASSETS.mkdir(parents=True,exist_ok=True)
for name,sha in expected['expected_assets_sha256'].items():
    source=DRIVE/expected['assets']/name
    assert source.is_file() and digest(source)==sha, f'Missing/mismatched asset: {source}'
    target=ASSETS/name
    if not target.exists() or digest(target)!=sha:shutil.copy2(source,target)
    assert digest(target)==sha

RUN=Path('/content/phase7_pilot_runs')/PILOT_RUN_ID
BACKUP=DRIVE/'phase7/improvement_round1'/PILOT_RUN_ID
print('RUN_ID:',PILOT_RUN_ID,'STAGE:',PILOT_STAGE,'GPU:',torch.cuda.get_device_name())
print('Git commit:',PILOT_COMMIT,'Drive results/weights:',BACKUP)
print('A smoke pass is an engineering check, not evidence of better quality or G7.2 PASS.')
subprocess.run([sys.executable,'-u','-m','tools.phase7_training_pilot',
    '--stage',PILOT_STAGE,'--data_root',str(STAGE),'--assets',str(ASSETS),
    '--run_root',str(RUN),'--backup_root',str(BACKUP),
    '--batch_size',str(PILOT_BATCH_SIZE),'--grad_accum_steps',str(PILOT_GRAD_ACCUM)],cwd=REPO,check=True)
result=BACKUP/'pilot_results_without_weights.zip'
print('Saved:',result)
if result.exists():files.download(str(result))
