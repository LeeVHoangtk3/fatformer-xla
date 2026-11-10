"""Reject mismatched IDs, altered reports and incomplete submitted diagnostics."""
import csv
import json

import numpy as np
import pytest

from tools.audit_phase7_validation_result import aligned,audit_and_import,csv_equal


def test_condition_pairing_uses_ids_instead_of_row_position():
    clean=dict(sample_ids=np.array(['a','b']),targets=np.array([0,1]))
    q30=dict(sample_ids=np.array(['b','a']),targets=np.array([1,0]),prob_fake=np.array([.9,.1]))
    result=aligned(clean,q30)
    np.testing.assert_array_equal(result['targets'],clean['targets'])
    np.testing.assert_array_equal(result['prob_fake'],[.1,.9])


def test_wrong_condition_population_is_not_paired():
    clean=dict(sample_ids=np.array(['a','b']),targets=np.array([0,1]))
    q30=dict(sample_ids=np.array(['a','c']),targets=np.array([0,1]))
    with pytest.raises(ValueError,match='IDs'):aligned(clean,q30)


@pytest.mark.parametrize('problem',['wrong_metric','extra_row'])
def test_corrupted_group_report_is_rejected(tmp_path,problem):
    left=tmp_path/'left.csv';right=tmp_path/'right.csv'
    left.write_text('group,recall\nCIPS,8.33333333333333\n')
    right.write_text('group,recall\nCIPS,83.33333333333333\n' if problem=='wrong_metric' else 'group,recall\nCIPS,8.33333333333333\nDiT,23.07\n')
    with pytest.raises(ValueError,match='CSV'):csv_equal(left,right)


def test_csv_float_formatting_does_not_create_a_false_failure(tmp_path):
    left=tmp_path/'left.csv';right=tmp_path/'right.csv'
    left.write_text('group,recall\nCIPS,8.33333333333333\n')
    right.write_text('group,recall\nCIPS,8.333333333333332\n')
    assert csv_equal(left,right)==1


@pytest.mark.parametrize('failed_status',[False,True])
def test_bad_submission_rejected_before_import(tmp_path,failed_status):
    source=tmp_path/'source';source.mkdir();base=tmp_path/'logs'
    (source/'identity.json').write_text(json.dumps({'source':{}}))
    (source/'status.json').write_text(json.dumps(dict(status='failed' if failed_status else 'completed',training_started=False,deployment_replaced=False,files={'identity.json':'0'*64})))
    notebook=tmp_path/'notebook.ipynb';notebook.write_text('{}')
    with pytest.raises(ValueError,match='status|hash'):audit_and_import(source,notebook,base)
    assert not base.exists()
