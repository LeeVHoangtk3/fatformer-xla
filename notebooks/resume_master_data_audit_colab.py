"""Resume an audit-stage failure in the CURRENT Colab runtime using source from Git."""
import json
from pathlib import Path
import subprocess
import sys


def resume_before_inference(namespace):
    previous = Path(namespace['OUTPUT']).resolve()
    old_repo = Path(namespace['REPO']).resolve()
    state = namespace['state']
    if any(row['name'] == 'model_load' for row in state['steps']):
        raise ValueError('This recovery is only for content-audit failures before model loading')
    if not (previous/'image_inventory.sqlite').is_file():
        raise FileNotFoundError('Old local inventory missing; keep the original Colab runtime alive')
    # Flush only committed audit evidence; partial archive rows will be restarted in a NEW run.
    if namespace.get('db') is not None:
        namespace['db'].rollback()
        namespace['db'].close()
        namespace['db'] = None
    commit = namespace.get('REPAIR_SOURCE_COMMIT') or subprocess.check_output(
        ['git', 'rev-parse', 'refs/remotes/origin/hoang'], cwd=old_repo, text=True).strip()
    if len(commit) != 40 or any(c not in '0123456789abcdefABCDEF' for c in commit):
        raise ValueError('Recovery source must be a full Git commit SHA')
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'refs/remotes/origin/hoang'], cwd=old_repo, check=True)
    source = subprocess.check_output(['git', 'show', commit+':notebooks/master_full_data_audit_colab.ipynb'],
                                     cwd=old_repo, text=True, encoding='utf-8')
    notebook = json.loads(source)
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    for name in list(sys.modules):
        if name in ('tools', 'src') or name.startswith(('tools.', 'src.')):
            del sys.modules[name]
    print('Recovery source commit:', commit)
    print('Previous content-audit evidence:', previous)
    for index, cell in enumerate(cells):
        if index == 1:
            if "GIT_COMMIT = ''" not in cell:
                raise ValueError('Cannot pin recovery source commit in this notebook version')
            cell = cell.replace("GIT_COMMIT = ''", 'GIT_COMMIT = '+repr(commit), 1)
        exec(compile(cell, f'recovery_git_cell_{index+1}', 'exec'), namespace)
        if index == 0:
            namespace['RESUME_AUDIT_FROM'] = str(previous)
            namespace['state']['recovery_driver_commit'] = commit


if __name__ == '__main__':
    resume_before_inference(globals())
