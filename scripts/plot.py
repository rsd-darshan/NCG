"""
NCG Research Prototype — Publication-quality plotting.

- Accuracy over tasks (with optional error bars for multiple seeds)
- Forgetting curve heatmap per model
- NCG growth curve, meta-parameter trajectories, novelty over time
- plot_all: generate all figures and save to results_dir/figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.rcParams["axes.grid"] = True
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.dpi"] = 150


def plot_accuracy_over_tasks(
    task_accs: Dict[str, List[List[float]]],
    save_path: str | Path,
    task_accs_std: Optional[Dict[str, List[List[float]]]] = None,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    T = None
    for model_name, accs in task_accs.items():
        if not accs:
            continue
        last_row = accs[-1]
        if T is None:
            T = len(last_row)
        x = list(range(T))
        y = last_row[:T]
        std = None
        if task_accs_std and model_name in task_accs_std and task_accs_std[model_name]:
            std_last = task_accs_std[model_name][-1]
            std = std_last[:T] if len(std_last) >= T else None
        if std is not None:
            ax.errorbar(x, y, yerr=std, label=model_name, capsize=3)
        else:
            ax.plot(x, y, "o-", label=model_name)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Accuracy (after all tasks)")
    ax.set_title("Accuracy per task (final evaluation)")
    ax.legend()
    ax.set_xticks(list(range(T)) if T else [])
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_forgetting_curve(
    task_accs: Dict[str, List[List[float]]],
    model_name: str,
    save_path: str | Path,
) -> None:
    _setup_style()
    if model_name not in task_accs:
        return
    accs = task_accs[model_name]
    if not accs:
        return
    T = len(accs)
    n_cols = T
    grid = np.zeros((T, n_cols))
    for t in range(T):
        for i in range(T):
            if i < len(accs[t]):
                grid[i, t] = accs[t][i]
            else:
                grid[i, t] = np.nan
    fig, ax = plt.subplots(figsize=(5, 4))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    im = ax.imshow(grid, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xlabel("After training task")
    ax.set_ylabel("Task evaluated on")
    ax.set_title(f"Forgetting curve — {model_name}")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(T))
    ax.set_xticklabels(range(n_cols))
    ax.set_yticklabels(range(T))
    plt.colorbar(im, ax=ax, label="Accuracy")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ncg_growth(
    hidden_size_per_epoch: List[int | float],
    save_path: str | Path,
    epochs_per_task: int = 10,
    num_tasks: int = 5,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(hidden_size_per_epoch)))
    ax.plot(x, hidden_size_per_epoch, color="tab:blue")
    for t in range(1, num_tasks):
        boundary = t * epochs_per_task
        if boundary <= len(hidden_size_per_epoch):
            ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hidden units")
    ax.set_title("NCG hidden layer size over training")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_meta_parameters(
    alpha_per_epoch: List[float],
    beta_per_epoch: List[float],
    lambda_per_epoch: List[float],
    save_path: str | Path,
    epochs_per_task: int = 10,
    num_tasks: int = 5,
) -> None:
    _setup_style()
    fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(alpha_per_epoch)
    x = list(range(n))
    axes[0].plot(x, alpha_per_epoch, color="tab:orange")
    axes[0].set_ylabel("α")
    axes[0].set_title("Exploration (α)")
    axes[0].set_ylim(0, 1.05)
    axes[1].plot(x, beta_per_epoch, color="tab:green")
    axes[1].set_ylabel("β")
    axes[1].set_title("Complexity penalty (β)")
    axes[1].set_ylim(0, None)
    axes[2].plot(x, lambda_per_epoch, color="tab:red")
    axes[2].set_ylabel("λ")
    axes[2].set_xlabel("Epoch")
    axes[2].set_title("Regularization (λ)")
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        for t in range(1, num_tasks):
            boundary = t * epochs_per_task
            if boundary <= n:
                ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_novelty(
    novelty_per_epoch: List[float],
    save_path: str | Path,
    epochs_per_task: int = 10,
    num_tasks: int = 5,
) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = list(range(len(novelty_per_epoch)))
    ax.plot(x, novelty_per_epoch, color="tab:purple")
    for t in range(1, num_tasks):
        boundary = t * epochs_per_task
        if boundary <= len(novelty_per_epoch):
            ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Novelty")
    ax.set_title("Novelty over training")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_all(
    ncg_logs: Dict[str, Any],
    task_accs: Dict[str, List[List[float]]],
    results_dir: str | Path,
    task_accs_std: Optional[Dict[str, List[List[float]]]] = None,
    epochs_per_task: int = 10,
    num_tasks: int = 5,
) -> None:
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy_over_tasks(
        task_accs,
        fig_dir / "accuracy_over_tasks.png",
        task_accs_std=task_accs_std,
    )
    for model_name in task_accs:
        plot_forgetting_curve(
            task_accs,
            model_name,
            fig_dir / f"forgetting_curve_{model_name.replace('-', '_')}.png",
        )
    if ncg_logs:
        plot_ncg_growth(
            ncg_logs["hidden_size_per_epoch"],
            fig_dir / "ncg_growth.png",
            epochs_per_task=epochs_per_task,
            num_tasks=num_tasks,
        )
        plot_meta_parameters(
            ncg_logs["alpha_per_epoch"],
            ncg_logs["beta_per_epoch"],
            ncg_logs["lambda_per_epoch"],
            fig_dir / "meta_parameters.png",
            epochs_per_task=epochs_per_task,
            num_tasks=num_tasks,
        )
        plot_novelty(
            ncg_logs["novelty_per_epoch"],
            fig_dir / "novelty.png",
            epochs_per_task=epochs_per_task,
            num_tasks=num_tasks,
        )
