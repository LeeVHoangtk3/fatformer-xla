"""Protect local review against refit mistakes and altered result evidence."""
import copy
import numpy as np
import pytest

from tools.audit_pilot_evaluation import Checks,copy_preserving,file_hashes,metric_checks,same_tree,temperature_numpy


def test_temperature_has_known_binary_nll_solution():
    # Four equal score differences and 75% positives: sigmoid(beta) = 3/4.
    result=temperature_numpy(np.tile([0.,1.],(4,1)),np.array([0,1,1,1]))
    assert result['T_float64']==pytest.approx(1/np.log(3),abs=1e-10)
    assert abs(result['gradient_beta'])<1e-12


@pytest.mark.parametrize('bad',['nan','single_class','shape'])
def test_invalid_calibration_data_rejected(bad):
    logits=np.tile([0.,1.],(4,1));targets=np.array([0,1,1,1])
    if bad=='nan':logits[0,0]=np.nan
    elif bad=='single_class':targets[:]=1
    else:logits=logits[:3]
    with pytest.raises(ValueError,match='finite binary'):temperature_numpy(logits,targets)


def test_point_metrics_cannot_disagree_with_predictions():
    checks=Checks();metric_checks(checks,{'auc':87.96},{'auc':88.66})
    with pytest.raises(ValueError,match='metric_auc'):checks.finish()


@pytest.mark.parametrize('bad',['ci','hash','missing'])
def test_full_paired_comparison_rejects_tampered_evidence(bad):
    expected={'ci':[-6.1,.5],'hash':'frozen','n':1000};actual=copy.deepcopy(expected)
    if bad=='ci':actual['ci'][1]=-1
    elif bad=='hash':actual['hash']='other'
    else:actual.pop('n')
    with pytest.raises(ValueError,match='Comparison'):same_tree(actual,expected)


def test_import_does_not_overwrite_different_existing_evidence(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'evidence.bin').write_bytes(b'original\x00\xff')
    expected=file_hashes(source);target=tmp_path/'target'
    copy_preserving(source,target,expected)
    assert file_hashes(target)==expected
    (target/'evidence.bin').write_bytes(b'altered')
    with pytest.raises(ValueError,match='Refusing to overwrite'):copy_preserving(source,target,expected)


def test_source_changed_after_audit_is_rejected(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'data').write_bytes(b'original')
    expected=file_hashes(source);(source/'data').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Source changed'):copy_preserving(source,tmp_path/'target',expected)
