"""
Standalone script to plot NCG meta-parameters (α, β, λ) across epochs.

Loads ncg_logs from results/ncg_logs.pkl (saved by run_all_seeds in ncg.evaluate).
For seed 42, plots α, β, and λ on 3 stacked subplots with task-boundary vertical lines.
Saves to results/figures/meta_param_trajectory.pdf.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NCG meta-parameters (α, β, λ) for seed 42")
    parser.add_argument("--results_dir", type=str, default="./results", help="Results directory (contains ncg_logs.pkl)")
    parser.add_argument("--seed", type=int, default=42, help="Seed to plot (default: 42)")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: results_dir/figures/meta_param_trajectory.pdf)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    ncg_logs_path = results_dir / "ncg_logs.pkl"

    if not ncg_logs_path.exists():
        raise FileNotFoundError(
            f"ncg_logs not found at {ncg_logs_path}. Run scripts/main.py (or run_all_seeds) first to generate it."
        )

    with open(ncg_logs_path, "rb") as f:
        data = pickle.load(f)

    ncg_logs = data["ncg_logs"]
    seed_list = data["seed_list"]

    try:
        idx = seed_list.index(args.seed)
    except ValueError:
        raise ValueError(f"Seed {args.seed} not in saved logs. Available seeds: {seed_list}")

    log = ncg_logs[idx]
    alpha_per_epoch = log["alpha_per_epoch"]
    beta_per_epoch = log["beta_per_epoch"]
    lambda_per_epoch = log["lambda_per_epoch"]
    epochs_per_task = log["epochs_per_task"]

    n_epochs = len(alpha_per_epoch)
    epochs = list(range(n_epochs))
    task_boundaries = [k * epochs_per_task for k in range(1, (n_epochs + epochs_per_task - 1) // epochs_per_task)]
    task_boundaries = [x for x in task_boundaries if 0 < x < n_epochs]

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7, 6))
    for ax, values, label, ylabel in [
        (axes[0], alpha_per_epoch, r"$\alpha$", r"$\alpha$"),
        (axes[1], beta_per_epoch, r"$\beta$", r"$\beta$"),
        (axes[2], lambda_per_epoch, r"$\lambda$", r"$\lambda$"),
    ]:
        ax.plot(epochs, values, color="C0", label=label)
        for x in task_boundaries:
            ax.axvline(x=x, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"NCG meta-parameters (seed {args.seed})", fontsize=11)
    plt.tight_layout()

    out_path = Path(args.output) if args.output else results_dir / "figures" / "meta_param_trajectory.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
