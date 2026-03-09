"""
Convergence diagnostics for NCG meta-parameters (α, β, λ).

Under mild smoothness assumptions, the meta-parameter gradient flow converges
to a fixed point φ* where F(φ*) = 0. At this point, α* reflects learned
confidence calibration, β* reflects optimal complexity penalty, and λ* reflects
optimal regularisation strength. Monotonic decrease alone does not confirm
convergence — stabilisation above zero is required.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader


def diagnose_convergence(
    history: list[float],
    param_name: str,
    stability_threshold: float = 0.01,
    decay_floor: float = 0.05,
) -> dict:
    """
    Distinguish genuine meta-parameter convergence from gradient decay.

    Computes smoothed rate of change over early vs late training and checks
    if the final value stabilises well above zero.

    Args:
        history: Per-epoch values of the meta-parameter.
        param_name: Label for the parameter (e.g. "alpha", "beta", "lambda").
        stability_threshold: Max absolute change in final 20% to count as stabilised.
        decay_floor: Below this, treat as "decaying" if not stabilised.

    Returns:
        Dict with keys: param, final_value, rate_decreasing (bool), stabilised (bool),
        classification ("converging" | "decaying" | "inconclusive"), message (str).
        - "converging": rate decreasing AND stabilised AND final_value > decay_floor
        - "decaying": final_value <= decay_floor AND not stabilised
        - "inconclusive": everything else
    """
    if not history:
        return {
            "param": param_name,
            "final_value": 0.0,
            "rate_decreasing": False,
            "stabilised": False,
            "classification": "inconclusive",
            "message": "Empty history.",
        }
    n = len(history)
    final_value = float(history[-1])

    # Smoothed rate: mean absolute step in early vs late third
    def mean_abs_rate(seq: list[float], start: int, end: int) -> float:
        if end <= start + 1:
            return 0.0
        steps = [abs(seq[i + 1] - seq[i]) for i in range(start, min(end, len(seq) - 1))]
        return sum(steps) / len(steps) if steps else 0.0

    early_end = max(1, n // 3)
    late_start = max(0, n - n // 3 - 1)
    rate_early = mean_abs_rate(history, 0, early_end)
    rate_late = mean_abs_rate(history, late_start, n)
    rate_decreasing = rate_late <= rate_early or rate_late < stability_threshold

    # Stabilised: final 20% has max change <= stability_threshold
    tail_size = max(1, int(0.2 * n))
    tail = history[-tail_size:]
    spread_tail = max(tail) - min(tail) if len(tail) > 1 else 0.0
    stabilised = spread_tail <= stability_threshold and final_value > decay_floor

    if rate_decreasing and stabilised and final_value > decay_floor:
        classification = "converging"
        message = f"{param_name} converges to ~{final_value:.4f} (rate decreasing, stabilised above {decay_floor})."
    elif final_value <= decay_floor and not stabilised:
        classification = "decaying"
        message = f"{param_name} decays to {final_value:.4f} (≤ {decay_floor}); may be gradient decay rather than fixed point."
    else:
        classification = "inconclusive"
        message = f"{param_name} final={final_value:.4f}, rate_decreasing={rate_decreasing}, stabilised={stabilised}."

    return {
        "param": param_name,
        "final_value": final_value,
        "rate_decreasing": rate_decreasing,
        "stabilised": stabilised,
        "classification": classification,
        "message": message,
    }


def run_diagnostics(ncg_logs: dict, verbose: bool = True) -> dict:
    """
    Run convergence diagnostics on NCG training logs.

    Takes the results dict from train_ncg() (or one seed's entry from ncg_logs list).
    Reads alpha_per_epoch, beta_per_epoch, lambda_per_epoch and runs
    diagnose_convergence on each. Optionally prints a formatted summary table.

    Returns:
        Dict with keys "alpha", "beta", "lambda", each the result of diagnose_convergence().
    """
    alpha_h = ncg_logs.get("alpha_per_epoch", [])
    beta_h = ncg_logs.get("beta_per_epoch", [])
    lambda_h = ncg_logs.get("lambda_per_epoch", [])

    results = {
        "alpha": diagnose_convergence(alpha_h, "alpha"),
        "beta": diagnose_convergence(beta_h, "beta"),
        "lambda": diagnose_convergence(lambda_h, "lambda"),
    }

    if verbose:
        print("\n" + "=" * 70)
        print("NCG meta-parameter convergence diagnostics")
        print("=" * 70)
        print(f"{'Param':<8} {'Final':<10} {'Rate↓':<8} {'Stable':<8} {'Classification':<14}")
        print("-" * 70)
        for key in ("alpha", "beta", "lambda"):
            r = results[key]
            print(f"{r['param']:<8} {r['final_value']:<10.4f} {str(r['rate_decreasing']):<8} {str(r['stabilised']):<8} {r['classification']:<14}")
        print("=" * 70)
        for key in ("alpha", "beta", "lambda"):
            print(f"  {results[key]['message']}")
        print()
    return results


def compute_theoretical_fixed_point(
    alpha_history: list[float],
    beta_history: list[float],
    lambda_history: list[float],
    tail_fraction: float = 0.2,
) -> dict:
    """
    Estimate where each meta-parameter is converging to.

    Uses the last `tail_fraction` (default 20%) of each history as the
    "converged region" and returns the mean of that region as the fixed-point
    estimate.

    Returns:
        Dict with alpha_star, beta_star, lambda_star (floats) and "verdict" (str):
        "Fixed point detected" if all three tails have low variance,
        else "No fixed point detected".
    """
    def tail_mean(h: list[float], frac: float) -> float:
        if not h:
            return 0.0
        k = max(1, int(len(h) * frac))
        tail = h[-k:]
        return sum(tail) / len(tail)

    def tail_std(h: list[float], frac: float) -> float:
        if not h or len(h) < 2:
            return 0.0
        k = max(1, int(len(h) * frac))
        tail = h[-k:]
        mean = sum(tail) / len(tail)
        var = sum((x - mean) ** 2 for x in tail) / len(tail)
        return var ** 0.5

    alpha_star = tail_mean(alpha_history, tail_fraction)
    beta_star = tail_mean(beta_history, tail_fraction)
    lambda_star = tail_mean(lambda_history, tail_fraction)

    # Low variance in tail => fixed point
    thresh = 0.02
    stable = (
        tail_std(alpha_history, tail_fraction) <= thresh
        and tail_std(beta_history, tail_fraction) <= thresh
        and tail_std(lambda_history, tail_fraction) <= thresh
    )
    verdict = "Fixed point detected" if stable else "No fixed point detected"

    return {
        "alpha_star": alpha_star,
        "beta_star": beta_star,
        "lambda_star": lambda_star,
        "verdict": verdict,
    }


# -----------------------------------------------------------------------------
# Perturbation test (experimental proof of fixed point)
# -----------------------------------------------------------------------------

_PARAM_RAW_ATTR = {"alpha": "alpha_raw", "beta": "beta_raw", "lambda": "lambda_raw"}
_PARAM_GETTER = {
    "alpha": lambda m: m.alpha.item(),
    "beta": lambda m: m.beta.item(),
    "lambda": lambda m: m.lambda_.item(),
}


def perturbation_test(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    param_name: str,
    delta: float = 0.1,
    steps: int = 20,
    lr_meta: float = 0.01,
    num_classes: int = 2,
) -> dict:
    """
    Nudge one meta-parameter by delta, run gradient meta-updates on val data,
    and check whether it returns toward the original value (fixed-point stability).

    Returns dict with: original_value, perturbed_value, recovered_value, delta_applied,
    recovery_ratio (1.0 = full recovery), verdict (fixed_point_confirmed | inconclusive | no_fixed_point).
    """
    if param_name not in _PARAM_RAW_ATTR:
        raise ValueError("param_name must be one of 'alpha', 'beta', 'lambda'")
    raw_attr = _PARAM_RAW_ATTR[param_name]
    get_val = _PARAM_GETTER[param_name]

    meta_params = model.get_meta_params()
    if not meta_params:
        return {
            "original_value": 0.0,
            "perturbed_value": 0.0,
            "recovered_value": 0.0,
            "delta_applied": delta,
            "recovery_ratio": 0.0,
            "verdict": "no_fixed_point",
        }
    opt_meta = torch.optim.Adam(meta_params, lr=lr_meta)

    model.eval()
    original_value = get_val(model)
    with torch.no_grad():
        getattr(model, raw_attr).add_(delta)
    perturbed_value = get_val(model)

    model.train()
    batches = []
    for x, y in val_loader:
        batches.append((x.to(device), y.to(device)))
        if len(batches) >= steps:
            break
    if len(batches) < steps:
        # Cycle through available batches
        from itertools import cycle
        batch_iter = cycle(batches)
        batches = [next(batch_iter) for _ in range(steps)]

    for val_x, val_y in batches:
        opt_meta.zero_grad()
        logits_meta, _ = model(val_x)
        meta_loss = model.compute_meta_loss(logits_meta, val_y, num_classes=num_classes)
        (-meta_loss).backward()
        opt_meta.step()

    recovered_value = get_val(model)
    denom = abs(perturbed_value - original_value)
    if denom < 1e-10:
        recovery_ratio = 1.0
    else:
        recovery_ratio = 1.0 - (abs(recovered_value - original_value) / denom)
    recovery_ratio = max(0.0, min(1.0, recovery_ratio))

    if recovery_ratio > 0.5:
        verdict = "fixed_point_confirmed"
    elif recovery_ratio > 0.2:
        verdict = "inconclusive"
    else:
        verdict = "no_fixed_point"

    return {
        "original_value": float(original_value),
        "perturbed_value": float(perturbed_value),
        "recovered_value": float(recovered_value),
        "delta_applied": float(delta),
        "recovery_ratio": float(recovery_ratio),
        "verdict": verdict,
    }


def run_full_analysis(
    model: torch.nn.Module,
    ncg_logs: dict,
    val_loader: DataLoader,
    device: torch.device,
    perturbation_delta: float = 0.1,
    perturbation_steps: int = 20,
) -> dict:
    """
    Run diagnostics, fixed-point estimate, and perturbation tests; print report.
    Returns dict with keys: diagnostics, fixed_points, perturbation_tests.
    """
    diagnostics = run_diagnostics(ncg_logs, verbose=False)
    alpha_h = ncg_logs.get("alpha_per_epoch", [])
    beta_h = ncg_logs.get("beta_per_epoch", [])
    lambda_h = ncg_logs.get("lambda_per_epoch", [])
    fixed_points = compute_theoretical_fixed_point(alpha_h, beta_h, lambda_h)

    perturbation_tests = {}
    for pname in ("alpha", "beta", "lambda"):
        perturbation_tests[pname] = perturbation_test(
            model, val_loader, device, pname,
            delta=perturbation_delta, steps=perturbation_steps,
        )

    # Formatted report
    print("\n" + "┌" + "─" * 42 + "┐")
    print("│     NCG Meta-Parameter Analysis         │")
    print("├" + "──────────┬──────────┬────────┬──────────┤")
    print("│ Param    │ Final    │ Class  │ Verdict  │")
    print("├" + "──────────┼──────────┼────────┼──────────┤")
    for key in ("alpha", "beta", "lambda"):
        d = diagnostics[key]
        fp_val = fixed_points[f"{key}_star"]
        pt = perturbation_tests[key]
        verdict_short = "✓ FP" if pt["verdict"] == "fixed_point_confirmed" else ("?" if pt["verdict"] == "inconclusive" else "✗")
        class_short = "conv." if d["classification"] == "converging" else ("decay" if d["classification"] == "decaying" else "inconcl.")
        print(f"│ {key:<8} │ {fp_val:<8.4f} │ {class_short:<6} │ {verdict_short:<8} │")
    print("└" + "──────────┴──────────┴────────┴──────────┘")
    n_fp = sum(1 for p in perturbation_tests.values() if p["verdict"] == "fixed_point_confirmed")
    print(f" Fixed point detected for {n_fp}/3 parameters.")
    print()

    return {
        "diagnostics": diagnostics,
        "fixed_points": fixed_points,
        "perturbation_tests": perturbation_tests,
    }
