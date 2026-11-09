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


def test_wrong_branch_is_rejected(repo):
    git(repo,'switch','-c','other')
    with pytest.raises(ValueError,match='Require branch hoang'):git_source_identity(repo)


def test_pinned_detached_commit_is_accepted_when_reachable_from_origin_hoang(repo):
    commit=git(repo,'rev-parse','HEAD')
    git(repo,'update-ref','refs/remotes/origin/hoang',commit)
    git(repo,'checkout','--detach',commit)
    proof=git_source_identity(repo)
    assert proof['commit']==commit and proof['branch']=='hoang'
