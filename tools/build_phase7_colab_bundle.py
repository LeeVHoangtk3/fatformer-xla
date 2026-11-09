"""Export the current working code and verified 400/1600 inputs, without GitHub or weights."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import numpy as np

from tools.calibration_data import sha256_file,validate_partitions
from tools.phase7_corrected import CONTRACT,TRAINING_SHA


def build(root,output):
    root=Path(root).resolve()
    sources={}
    for folder in ['src','tools','tests','configs']:
        for path in (root/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.yaml','.yml','.json','.gz'}:
                sources[path.relative_to(root).as_posix()]=path
    inputs={
      'inputs/calibration_400.npz':'DATASET/logs/phase7/task_7.15_recovery/logits_anchored_v2_calibration_400.npz',
      'inputs/calibration_split.npz':'DATASET/logs/phase7/wise_ft_sweep/logits_baseline_clean_calib_split.npz',
      'inputs/evaluation_split.npz':'DATASET/logs/phase7/wise_ft_sweep/logits_baseline_clean_eval_split.npz',
      'inputs/anchored_v2.json':'DATASET/logs/phase7/calibration/anchored_v2.json',
      'inputs/partition_audit.json':'DATASET/logs/phase7/task_7.15_recovery/partition_audit.json',
    }
    for name,path in inputs.items():
        sources[name]=root/path
    sources['notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb']=root/'notebooks/task/phase7_corrected/phase7_corrected_rerun_colab.ipynb'
    if any(not path.is_file() for path in sources.values()):
        raise FileNotFoundError('Missing bundle source/input/notebook')
    calibration=json.loads(sources['inputs/anchored_v2.json'].read_text(encoding='utf-8'))
    with np.load(sources['inputs/calibration_400.npz'],allow_pickle=False) as z:
        partition=validate_partitions(z['sample_ids'],z['targets'],
                                      sources['inputs/calibration_split.npz'],sources['inputs/evaluation_split.npz'])
        if str(z['checkpoint_sha256'])!=TRAINING_SHA:
            raise ValueError('Bundle cache belongs to the wrong training checkpoint')
    if (partition['n_calibration']!=400 or partition['n_evaluation']!=1600
            or calibration.get('partition_validation')!=partition
            or calibration.get('calibration_source_sha256')!=sha256_file(sources['inputs/calibration_400.npz'])):
        raise ValueError('Refuse export of an old or mismatched calibration bundle')
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip())
    except (OSError,subprocess.SubprocessError):
        commit,dirty=None,None
    contents={name:path.read_bytes() for name,path in sorted(sources.items())}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    manifest=dict(schema='phase7_colab_bundle/v1',contract=CONTRACT,
                  source=dict(base_git_commit=commit,working_tree_dirty=dirty,
                              method='snapshot of actual working files; not GitHub HEAD'),files=hashes,
                  note='SHA-256 checks consistency; it is not a digital signature. No model weights or datasets included.')
    manifest['preserved_notebooks_sha256']={str(path.relative_to(root)):sha256_file(path)
        for path in root.glob('notebooks/task/task_7.1[45]/*.ipynb')}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in contents.items():archive.writestr(name,data)
        archive.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    digest=sha256_file(output)
    output.with_suffix('.zip.sha256').write_text(digest+'  '+output.name+'\n',encoding='utf-8')
    return dict(path=str(output),sha256=digest,bytes=output.stat().st_size,n_files=len(contents),contract=CONTRACT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--output',default='DATASET/logs/phase7/task_7.15_recovery/colab/phase7_corrected_colab_bundle.zip')
    print(json.dumps(build(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':main()
