"""Audit GPU decisions as saved, and reject altered evidence/guard claims."""
import json

import numpy as np
import pytest

from tools import phase7_calibration_transfer as transfer
from tools.audit_phase7_calibration_transfer import audit_and_import,check_scores,real_guard


def fixture():
    plan=transfer.read(transfer.PLAN);point={'T':1.,'threshold':.5,'source_sha256':'point_fixture'}
    raw=dict(logits=np.zeros((2,2),np.float32),targets=np.array([0,1]),sample_ids=np.array(['a','b']),
             families=np.array(['real','fake']),generators=np.array(['real','fixture']),paths=np.array(['a.png','b.png']))
    # One float32 ULP below CPU softmax(.0,.0)=.5: saved GPU labels are 0.
    data={k:v.copy() for k,v in raw.items() if k!='logits'}
    data.update(prob_fake=np.full(2,np.nextafter(np.float32(.5),np.float32(0)),dtype=np.float32),labels=np.zeros(2,dtype=int),
        checkpoint_sha256=plan['models']['candidate_p0']['sha256'],plan_sha256=transfer.sha(transfer.PLAN),point_sha256='point_fixture',
        split='val_select',condition='clean',precision='fp16',probability_device='cuda',batch_size=16,T=1.,threshold=.5)
    return data,raw,point,plan


def test_cpu_boundary_does_not_rewrite_consistent_saved_gpu_labels():
    data,raw,point,plan=fixture()
    result=check_scores(data,raw,point,'candidate_p0','clean',plan,16)
    assert result['cpu_label_disagreements']==2
    np.testing.assert_array_equal(data['labels'],[0,0])


@pytest.mark.parametrize('problem',['nan','labels','ids','checkpoint'])
def test_altered_score_evidence_is_rejected(problem):
    data,raw,point,plan=fixture()
    if problem=='nan':data['prob_fake'][0]=np.nan
    elif problem=='labels':data['labels'][0]=1
    elif problem=='ids':data['sample_ids']=data['sample_ids'][::-1]
    else:data['checkpoint_sha256']='wrong'
    with pytest.raises(ValueError):check_scores(data,raw,point,'candidate_p0','clean',plan,16)


def test_fake_recall_gain_does_not_override_real_guard():
    summary={'clean':{'cfe_control':{'real_recall_pct':99.},'v2_clean_fit':{'real_recall_pct':94.}},
             'q30_post_crop':{'cfe_control':{'real_recall_pct':98.},'v2_clean_fit':{'real_recall_pct':86.95}}}
    assert real_guard(summary,.90)=={'cfe_control':True,'v2_clean_fit':False}


@pytest.mark.parametrize('incomplete',[False,True])
def test_bad_submission_fails_before_import(tmp_path,incomplete):
    source=tmp_path/'source';source.mkdir();destination=tmp_path/'logs'
    (source/'identity.json').write_text(json.dumps({'source':{}}))
    (source/'status.json').write_text(json.dumps(dict(status='failed' if incomplete else 'completed',training_started=False,
        deployment_replaced=False,files={'identity.json':'0'*64})))
    with pytest.raises(ValueError,match='incomplete|hash'):audit_and_import(source,destination)
    assert not destination.exists()
