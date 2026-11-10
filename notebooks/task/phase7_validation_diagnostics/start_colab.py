"""Execute in a fresh Colab notebook after cloning Git branch hoang."""
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from google.colab import drive,files
import torch

REPO=Path(globals().get('DIAG_REPO','/content/fatformer-xla')).resolve()
assert (REPO/'.git').is_dir(),'Clone the hoang Git branch first.'
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
assert torch.cuda.is_available(),'Select any compatible CUDA GPU with enough memory before running.'
packages={'yaml':'pyyaml','sklearn':'scikit-learn','scipy':'scipy','ftfy':'ftfy','regex':'regex',
          'tqdm':'tqdm','matplotlib':'matplotlib','tabulate':'tabulate'}
missing=[package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)

from tools.phase7_git_source import git_source_identity
from tools.phase7_validation_diagnostics import sha,read
from tools.run_pilot_checkpoint_eval import safe_extract
source=git_source_identity(REPO)
DIAG_COMMIT=globals().get('DIAG_COMMIT') or source['commit']
assert source['commit']==DIAG_COMMIT,'Code changed; use its recorded commit and a new run ID.'
DIAG_RUN_ID=globals().get('DIAG_RUN_ID') or 'val_diag_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
assert DIAG_RUN_ID.startswith('val_diag_') and '/' not in DIAG_RUN_ID and '\\' not in DIAG_RUN_ID and '..' not in DIAG_RUN_ID,'Invalid diagnostic run ID'
DIAG_MAX_BATCH=globals().get('DIAG_MAX_BATCH',16)
drive.mount('/content/drive')
DRIVE=Path('/content/drive/MyDrive/Fatformer')
plan=read(REPO/'configs/phase7_validation_diagnostics/plan.json')
archive=DRIVE/plan['archive']['drive_path']
print('Hashing training archive:',archive,flush=True)
assert archive.is_file() and sha(archive)==plan['archive']['sha256'],'Wrong/missing audited training archive'
STAGE=Path('/content/phase7_validation_diagnostic_data')
marker=STAGE/'.extracted_sha256'
if marker.exists():
    assert marker.read_text().strip()==plan['archive']['sha256'],'Another staged archive; use a fresh runtime'
else:
    assert not STAGE.exists() or not any(STAGE.iterdir()),'Partial staging exists; preserve it and use a fresh runtime'
    print('Extracting audited train/validation archive to Colab SSD...',flush=True)
    safe_extract(archive,STAGE);marker.write_text(plan['archive']['sha256'])
RUN=Path('/content/phase7_validation_diagnostics')/DIAG_RUN_ID
BACKUP=DRIVE/'phase7/improvement_round2/diagnostics'/DIAG_RUN_ID
print('Git commit:',DIAG_COMMIT,'Run ID:',DIAG_RUN_ID,'GPU:',torch.cuda.get_device_name(),flush=True)
print('Validation forward only; no training, no CFE/test benchmark, no deployment replacement.',flush=True)
try:
    subprocess.run([sys.executable,'-u','-m','tools.phase7_validation_diagnostics','collect',
        '--drive_root',str(DRIVE),'--data_root',str(STAGE),'--run_root',str(RUN),
        '--backup_root',str(BACKUP),'--max_batch_size',str(DIAG_MAX_BATCH)],cwd=REPO,check=True)
finally:
    print('Drive output:',BACKUP,flush=True)
result=BACKUP/'validation_diagnostics_without_weights.zip'
assert result.is_file(),'Collector did not finish/export results'
files.download(str(result))
print('Done. Save this executed notebook; return the result folder for per-image diagnosis before defining another training experiment.')
