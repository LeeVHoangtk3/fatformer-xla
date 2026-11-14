from pathlib import Path
import subprocess

import pytest

from tools.phase7_git_source import git_source_identity


def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True).strip()


@pytest.fixture
def repo(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    git(root,'init','-b','hoang')
    (root/'src').mkdir();(root/'src/example.py').write_text('VALUE=1\n')
    git(root,'add','src/example.py')
    git(root,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture')
    return root


def test_clean_hoang_records_exact_commit_and_file_hash(repo):
    proof=git_source_identity(repo)
    assert proof['commit']==git(repo,'rev-parse','HEAD')
    assert proof['branch']=='hoang' and proof['delivery']=='git'
    assert len(proof['files']['src/example.py'])==64


@pytest.mark.parametrize('mode',['modified','untracked','staged'])
def test_code_not_in_commit_is_rejected(repo,mode):
    name='new.py' if mode=='untracked' else 'example.py'
    (repo/'src'/name).write_text('VALUE=2\n')
    if mode=='staged':git(repo,'add','src/example.py')
    with pytest.raises(ValueError,match='differ from the recorded commit'):
        git_source_identity(repo)


def test_generated_logs_do_not_make_training_source_dirty(repo):
    (repo/'logs').mkdir();(repo/'logs/runtime.json').write_text('{}')
    assert git_source_identity(repo)['commit']==git(repo,'rev-parse','HEAD')


def test_executed_notebook_outputs_are_evidence_not_executable_source(repo):
    folder=repo/'notebooks/task/phase7_improvement_round1';folder.mkdir(parents=True)
    (folder/'executed.ipynb').write_text('{"outputs": ["new output"]}')
    proof=git_source_identity(repo)
    assert proof['commit']==git(repo,'rev-parse','HEAD')
    (folder/'start_colab.py').write_text('print("uncommitted code")')
    with pytest.raises(ValueError,match='differ from the recorded commit'):git_source_identity(repo)


def test_wrong_branch_is_rejected(repo):
    git(repo,'switch','-c','other')
    with pytest.raises(ValueError,match='Require branch hoang'):git_source_identity(repo)


def test_pinned_detached_commit_is_accepted_when_reachable_from_origin_hoang(repo):
    commit=git(repo,'rev-parse','HEAD')
    git(repo,'update-ref','refs/remotes/origin/hoang',commit)
    git(repo,'checkout','--detach',commit)
    proof=git_source_identity(repo)
    assert proof['commit']==commit and proof['branch']=='hoang'


def test_custom_source_paths_include_frozen_inputs_and_notebook(repo):
    (repo/'configs').mkdir()
    (repo/'configs/split.json').write_text('{"samples": []}')
    (repo/'notebooks').mkdir()
    (repo/'notebooks/audit.ipynb').write_text('{"cells": []}')
    git(repo,'add','configs','notebooks')
    git(repo,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','audit inputs')
    proof=git_source_identity(repo, source_paths=['src','configs','notebooks/audit.ipynb'])
    assert 'configs/split.json' in proof['files'] and 'notebooks/audit.ipynb' in proof['files']
    (repo/'configs/split.json').write_text('{"samples": [1]}')
    with pytest.raises(ValueError,match='differ from the recorded commit'):
        git_source_identity(repo, source_paths=['src','configs','notebooks/audit.ipynb'])


def test_empty_source_scope_cannot_bypass_clean_check(repo):
    with pytest.raises(ValueError,match='must not be empty'):
        git_source_identity(repo, source_paths=[])
