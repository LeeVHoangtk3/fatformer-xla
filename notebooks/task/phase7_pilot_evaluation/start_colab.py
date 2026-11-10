"""Run from a clean hoang Git checkout in a NEW Colab notebook; no training."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime,timezone

import torch
from google.colab import drive,files

EVAL_STAGE=globals().get('EVAL_STAGE','full')
EVAL_RUN_ID=globals().get('EVAL_RUN_ID') or datetime.now(timezone.utc).strftime('p0_eval_%Y%m%d_%H%M%S')
EVAL_MAX_BATCH=globals().get('EVAL_MAX_BATCH',16)
assert EVAL_STAGE in ['calibrate','full'] and re.fullmatch(r'[A-Za-z0-9_-]+',EVAL_RUN_ID)
assert torch.cuda.is_available(), 'Select a compatible CUDA GPU first.'
REPO=Path(globals().get('EVAL_REPO','/content/fatformer-xla')).resolve()
sys.path.insert(0,str(REPO))
from tools.phase7_git_source import git_source_identity
source=git_source_identity(REPO)
EVAL_COMMIT=globals().get('EVAL_COMMIT') or source['commit']
assert source['commit']==EVAL_COMMIT, 'Code changed; use the recorded commit or a new eval run ID.'
print('Git commit:',EVAL_COMMIT,'Eval ID:',EVAL_RUN_ID,'GPU:',torch.cuda.get_device_name(),flush=True)

packages={'yaml':'pyyaml','sklearn':'scikit-learn','scipy':'scipy','matplotlib':'matplotlib',
          'tqdm':'tqdm','regex':'regex','ftfy':'ftfy','tabulate':'tabulate'}
missing=[package for module,package in packages.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.run([sys.executable,'-m','pip','install','-q',*missing],check=True)
drive.mount('/content/drive')
DRIVE=Path('/content/drive/MyDrive/Fatformer')
RUN=Path('/content/phase7_p0_evaluation')/EVAL_RUN_ID
BACKUP=DRIVE/'phase7/improvement_round1/evaluation'/EVAL_RUN_ID
RUN.mkdir(parents=True,exist_ok=True);BACKUP.mkdir(parents=True,exist_ok=True)

from tools.run_pilot_checkpoint_eval import stage_datasets
plan=json.loads((REPO/'configs/phase7_pilot_eval/plan.json').read_text())
for spec in plan['models'].values():
    path=DRIVE/spec['drive_path']
    assert path.is_file(), f'Missing exact source checkpoint: {path}'
print('Both models are recalibrated separately; incumbent_refit means old weights, not an exact replay of package 24869052.',flush=True)
try:
    stage_datasets(DRIVE,RUN,plan)
    subprocess.run([sys.executable,'-u','-m','tools.run_pilot_checkpoint_eval',
        '--drive_root',str(DRIVE),'--run_root',str(RUN),'--backup_root',str(BACKUP),
        '--cfe_root',str(RUN/'data/cfe'),'--indist_root',str(RUN/'data/indist'),
        '--stage',EVAL_STAGE,'--max_batch_size',str(EVAL_MAX_BATCH)],cwd=REPO,check=True)
finally:
    print('Results/weights on Drive:',BACKUP,flush=True)
result=BACKUP/'evaluation_results_without_weights.zip'
if result.exists():files.download(str(result))
print('Finished. Review G7.2 and matched comparison; no automatic deployment or best-reference G7.3 claim.')
