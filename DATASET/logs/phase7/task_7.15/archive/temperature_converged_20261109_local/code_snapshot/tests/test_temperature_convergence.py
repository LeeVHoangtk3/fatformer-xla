"""Numerical regressions for the premature Task 7.15 temperature fit."""
import math

import numpy as np
import pytest
import torch

from tools.calibrate_and_select import ModelWithTemperature, run_measured_calibration


@pytest.mark.parametrize('initial', [.05, 1., 10.])
def test_interior_optimum_matches_analytic_binary_likelihood(initial):
    # Identical logit margin 2, positive frequency 3/4: sigmoid(2/T)=3/4.
    logits = torch.tensor([[0., 2.]] * 40)
    labels = torch.tensor([0] * 10 + [1] * 30)
    model = ModelWithTemperature(initial)
    temperature, raw = model.fit(logits, labels)
    assert temperature == pytest.approx(2. / math.log(3.), rel=1e-7)
    assert raw == temperature == model.temperature.item()
    assert model.fit_diagnostics['status'] == 'interior_converged'
    assert model.fit_diagnostics['scaled_kkt_residual'] < 1e-6
    assert model.fit_diagnostics['nll_at_fitted_T'] < model.fit_diagnostics['nll_at_T1']


@pytest.mark.parametrize('labels,expected,status', [
    ([0, 1], .05, 'lower_temperature_bound'),
    ([1, 0], 10., 'upper_temperature_bound'),
])
def test_endpoint_optimum_is_a_valid_constrained_solution(labels, expected, status):
    model = ModelWithTemperature()
    fitted, _ = model.fit(torch.tensor([[4., 0.], [0., 4.]]), torch.tensor(labels))
    assert fitted == pytest.approx(expected)
    assert model.fit_diagnostics['converged']
    assert model.fit_diagnostics['status'] == status


def test_flat_objective_retains_neutral_temperature():
    model = ModelWithTemperature(7.)
    assert model.fit(torch.zeros(4, 2), torch.tensor([0, 0, 1, 1])) == (1., 1.)
    assert model.fit_diagnostics['status'] == 'flat_objective'
    assert model.fit_diagnostics['nll_at_fitted_T'] == pytest.approx(math.log(2.))


def test_insufficient_iterations_fail_without_updating_parameter():
    model = ModelWithTemperature()
    with pytest.raises(RuntimeError, match='did not converge'):
        model.fit(torch.tensor([[0., 2.]] * 4), torch.tensor([0, 1, 1, 1]), max_iter=1)
    assert model.temperature.item() == 1.
    assert model.fit_diagnostics is None


@pytest.mark.parametrize('logits,labels', [
    ([[0., float('nan')]], [1]),
    ([[float('inf'), 0.]], [0]),
    ([[1., 2., 3.]], [1]),
    ([[0., 2.]], [2]),
    ([[0., 2.]], [1.0]),
])
def test_invalid_fit_input_is_rejected(logits, labels):
    with pytest.raises(ValueError, match='finite'):
        ModelWithTemperature().fit(torch.tensor(logits), torch.tensor(labels))


def test_measured_export_contains_convergence_and_refitted_thresholds(tmp_path):
    source = tmp_path / 'measured.npz'
    np.savez(source, logits=np.array([[0., 2.]] * 40, dtype='float32'),
             targets=np.array([0] * 10 + [1] * 30), split='calibration', evidence_level='measured')
    result = run_measured_calibration(str(source), str(tmp_path / 'output'), 'analytical')
    assert result['temperature_optimization']['converged'] is True
    assert result['T'] == pytest.approx(2. / math.log(3.), rel=1e-7)
    assert result['fpr_measured']['0.05'] == 0.  # Tied real scores cannot be split.
    assert result['fake_acc'] == 0.
    assert result['nll_after'] < result['nll_before']
