"""Reject corrupt optimizer evidence rather than trusting reassuring averages."""
import copy
import json

import pytest

from tools.audit_phase7_training_pilot import Checks,inspect_trace


def events():
    return [dict(schema='optimizer-update/v1',unrecoverable=False,applied=False,amp_skipped=True,
                 grad_norm=None,grad_norm_finite=False,amp_scale_before=2048.,amp_scale_after=1024.),
            dict(schema='optimizer-update/v1',unrecoverable=False,applied=True,amp_skipped=False,
                 grad_norm=4.,grad_norm_finite=True,amp_scale_before=1024.,amp_scale_after=1024.)]


def test_recorded_amp_recovery_is_valid_evidence(tmp_path):
    path=tmp_path/'trace.jsonl'
    path.write_text(''.join(json.dumps(event)+'\n' for event in events()))
    checks=Checks()
    assert len(inspect_trace(path,checks))==2
    checks.finish()


@pytest.mark.parametrize('problem',['applied_nonfinite','fake_skip','scale_jump','hidden_nan','unrecoverable'])
def test_corrupt_trace_is_rejected(tmp_path,problem):
    rows=events()
    if problem=='applied_nonfinite':rows[0].update(applied=True,amp_skipped=False)
    elif problem=='fake_skip':rows[1]['amp_skipped']=True
    elif problem=='scale_jump':rows[1]['amp_scale_before']=2048.
    elif problem=='hidden_nan':rows[1]['grad_norm']=float('nan')
    else:rows[1]['unrecoverable']=True
    path=tmp_path/'trace.jsonl';path.write_text(''.join(json.dumps(event)+'\n' for event in rows))
    checks=Checks();inspect_trace(path,checks)
    with pytest.raises(ValueError,match='Pilot evidence audit failed'):checks.finish()
