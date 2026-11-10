"""Run the frozen calibration-transfer experiment from Git branch hoang."""
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import subprocess
import sys

from google.colab import drive,files
import torch

REPO=Path(globals().get('TRANSFER_REPO','/content/fatformer-xla')).resolve()
assert (REPO/'.git').is_dir(),'Clone Git branch hoang in a fresh runtime first.'
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
assert torch.cuda.is_available(),'Choose any compatible CUDA GPU with enough memory.'
packages={'yaml':'pyyaml','sklearn':'scikit-learn','scipy':'scipy','ftfy':'ftfy','regex':'regex',
          'tqdm':'tqdm','matplotlib':'matplotlib','tabulate':'tabulate'}
missing=[package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)

from tools.phase7_git_source import git_source_identity
from tools.phase7_calibration_transfer import PLAN,read,sha
from tools.run_pilot_checkpoint_eval import safe_extract
source=git_source_identity(REPO)
TRANSFER_COMMIT=globals().get('TRANSFER_COMMIT') or source['commit']
assert source['commit']==TRANSFER_COMMIT,'Code changed during run; restore its exact commit or use a new run ID.'
TRANSFER_RUN_ID=globals().get('TRANSFER_RUN_ID') or 'calib_transfer_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
assert TRANSFER_RUN_ID.startswith('calib_transfer_') and '/' not in TRANSFER_RUN_ID and '\\' not in TRANSFER_RUN_ID and '..' not in TRANSFER_RUN_ID,'Invalid run ID'
TRANSFER_MAX_BATCH=globals().get('TRANSFER_MAX_BATCH',16)
drive.mount('/content/drive')
DRIVE=Path('/content/drive/MyDrive/Fatformer');plan=read(PLAN)
archive=DRIVE/plan['archive']['drive_path']
print('Hashing original train/validation archive:',archive,flush=True)
assert archive.is_file() and sha(archive)==plan['archive']['sha256'],'Missing/wrong audited training archive.'
STAGE=Path('/content/phase7_calibration_transfer_data');marker=STAGE/'.extracted_sha256'
if marker.exists():
    assert marker.read_text().strip()==plan['archive']['sha256'],'Staging input changed; preserve it and use a fresh runtime.'
else:
    assert not STAGE.exists() or not any(STAGE.iterdir()),'Partial staging exists; preserve it and use a fresh runtime.'
    print('Extracting to Colab SSD...',flush=True);safe_extract(archive,STAGE);marker.write_text(plan['archive']['sha256'])
RUN=Path('/content/phase7_calibration_transfer')/TRANSFER_RUN_ID
BACKUP=DRIVE/'phase7/improvement_round2/calibration_transfer'/TRANSFER_RUN_ID
print('Commit:',TRANSFER_COMMIT,'Run ID:',TRANSFER_RUN_ID,'GPU:',torch.cuda.get_device_name(),flush=True)
print('No training or CFE test. Fit uses val_calib only; fixed validation logits are reused only under matched environment and batch.',flush=True)
try:
    subprocess.run([sys.executable,'-u','-m','tools.phase7_calibration_transfer',
        '--drive_root',str(DRIVE),'--data_root',str(STAGE),'--run_root',str(RUN),
        '--backup_root',str(BACKUP),'--max_batch_size',str(TRANSFER_MAX_BATCH)],cwd=REPO,check=True)
finally:print('Drive results:',BACKUP,flush=True)
result=BACKUP/'calibration_transfer_without_weights.zip'
assert result.is_file(),'No complete result archive was exported.'
files.download(str(result))
print('Done. Save the executed notebook and return the extracted results for audit. Calibration is research-only; deployment is unchanged.')
