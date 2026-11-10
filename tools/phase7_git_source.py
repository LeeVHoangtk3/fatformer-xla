"""Record the exact, clean Git source used by the hoang training workflow."""
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_PATHS = ['src','tools','tests','configs','notebooks/task/phase7_improvement_round1/*.py',
    'notebooks/task/phase7_pilot_evaluation/*.py',
    'notebooks/task/phase7_validation_diagnostics/*.py',
    'notebooks/task/phase7_calibration_transfer/*.py',
    'DATASET/logs/phase7/calibration/baseline.json',
    'DATASET/logs/phase7/improvement_round1/preparation/expected_inputs.json',
    'DATASET/logs/phase7/improvement_round1/preparation/protocol.json']


def git_source_identity(root):
    root=Path(root).resolve()
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True,encoding='utf-8',errors='strict').strip()
    try:
        top=Path(git('rev-parse','--show-toplevel')).resolve()
        commit=git('rev-parse','HEAD')
        branch=git('rev-parse','--abbrev-ref','HEAD')
    except (OSError,subprocess.CalledProcessError) as exc:
        raise ValueError('Pilot requires a Git checkout of branch hoang, not the old source ZIP') from exc
    if top!=root:
        raise ValueError('Pilot root must be the Git repository root')
    if branch=='HEAD':
        subprocess.run(['git','merge-base','--is-ancestor',commit,'refs/remotes/origin/hoang'],
                       cwd=root,check=True,capture_output=True)
    elif branch!='hoang':
        raise ValueError(f'Require branch hoang or its pinned commit; got {branch}')
    dirty=git('status','--porcelain','--untracked-files=all','--',*SOURCE_PATHS)
    if dirty:
        raise ValueError('Source/config/input files differ from the recorded commit; commit them or use a fresh checkout:\n'+dirty)
    paths=git('ls-files','-z','--',*SOURCE_PATHS).split('\0')
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths if name}
    if not files:raise ValueError('Git checkout has no tracked pilot sources')
    fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(schema='phase7-git-source/v1',delivery='git',branch='hoang',commit=commit,
                tracked_files_fingerprint=fingerprint,files=files)
