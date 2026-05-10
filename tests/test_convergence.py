"""Pytest tests for ncg.math.convergence."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from ncg.math.convergence import (
    compute_theoretical_fixed_point,
    diagnose_convergence,
    perturbation_test,
    run_full_analysis,
)
from ncg.model import NCGModel


def test_diagnose_convergence_returns_converging_when_stabilised_above_005():
    # History that stabilises above 0.05: flat tail, final > decay_floor
    history = [0.8, 0.6, 0.5, 0.48, 0.46, 0.45, 0.44, 0.44, 0.44, 0.44]
    result = diagnose_convergence(history, "alpha", stability_threshold=0.01, decay_floor=0.05)
    assert result["classification"] == "converging"
    assert result["final_value"] > 0.05
    assert result["stabilised"] is True


def test_diagnose_convergence_returns_decaying_when_approaches_zero():
    # History that approaches zero and does not stabilise above decay_floor
    history = [0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.008, 0.005]
    result = diagnose_convergence(history, "lambda", stability_threshold=0.01, decay_floor=0.05)
    assert result["classification"] == "decaying"
    assert result["final_value"] <= 0.05


def test_diagnose_convergence_returns_inconclusive_for_noisy_non_stabilising():
    # Noisy, non-stabilising: high variance in tail or rate not decreasing
    history = [0.5, 0.6, 0.4, 0.7, 0.3, 0.6, 0.5, 0.4, 0.6, 0.5]
    result = diagnose_convergence(history, "beta", stability_threshold=0.01, decay_floor=0.05)
    assert result["classification"] == "inconclusive"


def test_compute_theoretical_fixed_point_returns_values_within_last_20_percent():
    alpha_h = [0.1 * i for i in range(100)]
    beta_h = [0.01 * (i % 10) for i in range(100)]
    lambda_h = [0.5 + 0.01 * i for i in range(100)]
    result = compute_theoretical_fixed_point(alpha_h, beta_h, lambda_h, tail_fraction=0.2)
    tail_alpha = alpha_h[-20:]
    tail_beta = beta_h[-20:]
    tail_lambda = lambda_h[-20:]
    assert result["alpha_star"] == sum(tail_alpha) / len(tail_alpha)
    assert result["beta_star"] == sum(tail_beta) / len(tail_beta)
    assert result["lambda_star"] == sum(tail_lambda) / len(tail_lambda)
    assert "verdict" in result


def test_perturbation_test_returns_dict_with_all_required_keys():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=64)
    x = torch.randn(4, 16)
    y = torch.randint(0, 2, (4,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=2)
    device = torch.device("cpu")
    result = perturbation_test(model, val_loader, device, "alpha", delta=0.1, steps=5)
    required = ["original_value", "perturbed_value", "recovered_value", "delta_applied", "recovery_ratio", "verdict"]
    for key in required:
        assert key in result, f"Missing key: {key}"


def test_perturbation_test_recovery_ratio_between_0_and_1():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=64)
    x = torch.randn(6, 16)
    y = torch.randint(0, 2, (6,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=2)
    device = torch.device("cpu")
    for param_name in ("alpha", "beta", "lambda"):
        result = perturbation_test(model, val_loader, device, param_name, delta=0.1, steps=3)
        assert 0 <= result["recovery_ratio"] <= 1


def test_run_full_analysis_returns_dict_with_diagnostics_fixed_points_perturbation_tests():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=64)
    ncg_logs = {
        "alpha_per_epoch": [0.5] * 20,
        "beta_per_epoch": [0.03] * 20,
        "lambda_per_epoch": [0.6] * 20,
    }
    x = torch.randn(4, 16)
    y = torch.randint(0, 2, (4,))
    val_loader = DataLoader(TensorDataset(x, y), batch_size=2)
    device = torch.device("cpu")
    result = run_full_analysis(model, ncg_logs, val_loader, device, perturbation_steps=3)
    assert "diagnostics" in result
    assert "fixed_points" in result
    assert "perturbation_tests" in result
    assert list(result["diagnostics"].keys()) == ["alpha", "beta", "lambda"]
    assert "alpha_star" in result["fixed_points"]
    assert list(result["perturbation_tests"].keys()) == ["alpha", "beta", "lambda"]
