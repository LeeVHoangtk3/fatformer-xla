"""Regressions for the Task 7.15 calibration/evaluation leakage."""
import json

import numpy as np
import pytest
import torch

from tools.calibration_data import identities, validate_partitions
from tools.calibrate_and_select import run_measured_calibration, fit_thresholds_by_fpr
from tools.extract_calibration_logits import find_calib_samples, extract_logits
from tools.recover_calibration_split import recover
from tools.package_unified_fatformer import package_checkpoint


def partition(path, ids, split):
    np.savez(path, sample_ids=np.array(ids), targets=np.array([int(s.split('/')[1]) for s in ids]), split=split)


def test_frozen_split_maps_class_directory_and_never_scans_evaluation(tmp_path):
    folder = tmp_path / 'data' / 'subset' / '0_real'
    folder.mkdir(parents=True)
    for name in ['calib.png', 'evaluation.png']:
        (folder / name).touch()
    split = tmp_path / 'calib.npz'
    partition(split, ['subset/0/calib.png'], 'calibration')
    found = find_calib_samples(str(tmp_path / 'data'), str(split))
    assert [s[2] for s in found] == ['subset/0/calib.png']
    assert found[0][0] == str((folder / 'calib.png').resolve())
    (folder / 'calib.png').unlink()
    with pytest.raises(FileNotFoundError, match='Missing 1'):
        find_calib_samples(str(tmp_path / 'data'), str(split))


def test_split_is_mandatory_and_missing_file_fails(tmp_path):
    with pytest.raises(ValueError, match='explicit'):
        find_calib_samples(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        find_calib_samples(str(tmp_path), str(tmp_path / 'missing.npz'))


def test_duplicate_id_or_disagreeing_label_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        identities(['s/0_real/a.png', 's/0/a.png'], [0, 0])
    with pytest.raises(ValueError, match='disagrees'):
        identities(['s/1_fake/a.png'], [0])


@pytest.fixture
def measured_pool(tmp_path):
    calib = [f's/{y}/{i}.png' for y in [0, 1] for i in range(10)]
    evaluation = [f's/{y}/{i}.png' for y in [0, 1] for i in range(10, 12)]
    ids = evaluation + calib
    source, cal, ev = (tmp_path / n for n in ['pool.npz', 'cal.npz', 'eval.npz'])
    partition(cal, calib, 'calibration')
    partition(ev, evaluation, 'evaluation')
    targets = np.array([int(s.split('/')[1]) for s in ids])
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(len(ids), 2)).astype('float32')
    np.savez(source, logits=logits, targets=targets, paths=np.array(ids), sample_ids=np.array(ids),
             checkpoint_sha256='a' * 64, preprocessing='CLIP 224', split='calibration', evidence_level='measured')
    return source, cal, ev, ids, logits


def test_recover_selects_exact_frozen_rows_and_records_original_leakage(tmp_path, measured_pool):
    source, cal, ev, ids, logits = measured_pool
    output = tmp_path / 'recovered.npz'
    audit = recover(source, cal, ev, output, tmp_path / 'audit.json')
    assert audit['source_pool_evaluation_overlap'] == 4
    with np.load(output) as z:
        assert z['logits'].shape == (20, 2)
        assert np.array_equal(z['logits'], logits[4:])
        assert validate_partitions(z['sample_ids'], z['targets'], cal, ev)['overlap'] == 0
    with pytest.raises(ValueError, match='exactly match'):
        run_measured_calibration(str(source), str(tmp_path), 'bad', calibration_split=cal, evaluation_split=ev)


def test_overlapping_frozen_partitions_rejected(tmp_path, measured_pool):
    source, cal, ev, ids, _ = measured_pool
    partition(ev, [ids[4]], 'evaluation')
    with pytest.raises(ValueError, match='overlap'):
        recover(source, cal, ev, tmp_path / 'out.npz', tmp_path / 'audit.json')


def test_dummy_logits_cannot_become_measured_calibration(tmp_path):
    output = tmp_path / 'dummy.npz'
    extract_logits('absent.pth', str(tmp_path), str(output), test_dummy=True)
    with pytest.raises(ValueError, match='Synthetic'):
        run_measured_calibration(str(output), str(tmp_path), 'bad')


def test_threshold_with_ties_respects_fpr_after_serialization():
    scores = np.full(40, .5, dtype=np.float32)
    labels = np.array([0] * 20 + [1] * 20)
    thresholds, measured = fit_thresholds_by_fpr(scores, labels)
    for key, threshold in json.loads(json.dumps(thresholds)).items():
        assert float((scores[labels == 0] >= threshold).mean()) <= float(key)
        assert measured[key] <= float(key)


def test_bound_logits_require_frozen_partitions(tmp_path, measured_pool):
    with pytest.raises(ValueError, match='require calibration_split'):
        run_measured_calibration(str(measured_pool[0]), str(tmp_path), 'bad')


def test_recovery_rejects_wrong_labels_and_missing_rows(tmp_path, measured_pool):
    source, cal, ev, ids, logits = measured_pool
    with np.load(source) as z:
        data = {k: z[k].copy() for k in z.files}
    data['targets'][4] = 1
    np.savez(source, **data)
    with pytest.raises(ValueError, match='disagrees'):
        recover(source, cal, ev, tmp_path / 'out.npz', tmp_path / 'audit.json')


def test_corrupt_frozen_image_does_not_get_silently_skipped(tmp_path, monkeypatch):
    import tools.extract_calibration_logits as extract
    cal, ev = tmp_path / 'cal.npz', tmp_path / 'eval.npz'
    partition(cal, ['s/0/cal.png'], 'calibration')
    partition(ev, ['s/1/eval.png'], 'evaluation')
    folder = tmp_path / 's' / '0_real';folder.mkdir(parents=True)
    (folder / 'cal.png').write_bytes(b'not an image')
    monkeypatch.setattr(extract, 'extract_state_dict', lambda _: {})
    monkeypatch.setattr(extract, 'infer_adapter_blocks', lambda _: {7,15,23})
    monkeypatch.setattr(extract, 'load_canonical', lambda *_: (torch.nn.Identity(), {'sha256':'a'*64}))
    with pytest.raises(ValueError, match='Cannot read frozen'):
        extract.extract_logits('unused.pth', str(tmp_path), str(tmp_path / 'out.npz'),
                               split_file=str(cal), evaluation_split=str(ev))
    assert not (tmp_path / 'out.npz').exists()


def test_repackage_retains_training_hash_and_all_tensors(tmp_path):
    training_hash = 'a' * 64
    sd = {f'clip_model.visual.transformer.resblocks.{i}.forgery_aware_adapter.w': torch.ones(1) for i in [7, 15, 23]}
    source = tmp_path / 'old.pth'
    torch.save({'model': sd, 'metadata': {'schema': 'unified_fatformer/v1',
                                        'source_checkpoint_sha256': training_hash}}, source)
    calib = {'schema': 'calibration/v1', 'evidence_level': 'measured', 'T': 1.2, 'T_star_default': .8,
             'checkpoint': {'sha256': training_hash}, 'n_calib': 20, 'calibration_source_sha256': 'b' * 64,
             'partition_validation': {'status': 'verified_disjoint_ids', 'overlap': 0, 'n_calibration': 20,
                                      'calibration_split_sha256': 'c' * 64, 'evaluation_split_sha256': 'd' * 64,
                                      'sample_ids_sha256': 'e' * 64}}
    c = tmp_path / 'calib.json'; c.write_text(json.dumps(calib))
    meta = package_checkpoint(str(source), str(tmp_path / 'new.pth'), str(c), repackage=True)
    assert meta['source_checkpoint_sha256'] == training_hash
    assert meta['repackaged_from']['sha256'] != training_hash
    with pytest.raises(ValueError, match='explicit --repackage'):
        package_checkpoint(str(source), str(tmp_path / 'bad.pth'), str(c))


def test_locked_runner_does_not_accept_backbone_latency_or_missing_metrics():
    from tools.run_locked_benchmark_phase7 import evaluate_master_gates
    subsets = {name: {'fake_acc': 90} for name in
               ['1_commercial','2_latent_diffusion','3_pixel_diffusion','4_modern_gans','5_other_autoregressive']}
    def gates(q30):
        return evaluate_master_gates({'macro_acc': 95,'real_acc': 95},
                                     {'auc': 90,'fake_at_real90': 80}, q30, subsets, 1.0, {})
    result = gates({'auc': 85,'fake_at_real90': 70,'ece': .5})
    assert result['G7.2']['criteria']['ECE_Post_Calibration']['value'] == .005
    assert result['G7.2']['pass'] is True
    assert result['G7.3']['criteria']['GPU_Latency_ms_per_image']['pass'] is False
    result = gates({'auc': 85,'fake_at_real90': 70})
    assert result['G7.2']['criteria']['ECE_Post_Calibration']['pass'] is False
    subsets['1_commercial']['fake_acc'] = None
    assert gates({'auc': 85,'fake_at_real90': 70,'ece': 1})['G7.2']['criteria']['Worst_Subset_Fake_ACC']['pass'] is False
