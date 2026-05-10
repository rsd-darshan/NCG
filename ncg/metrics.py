"""
NCG — Continual learning metrics.

- Forgetting, backward transfer (BWT), forward transfer (FWT)
- Results table compilation and CSV export
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def compute_forgetting(task_accs: Dict[str, List[List[float]]]) -> Dict[str, float]:
    """
    Compute mean forgetting per model.

    task_accs[model] is a 2D list: task_accs[t][i] = accuracy on task i after
    training through task t (t, i in 0..T-1). Forgetting for task i = max
    accuracy on task i before final training minus accuracy on task i after
    all tasks. Mean forgetting = (1/T) * sum_i forgetting_i.

    Returns:
        Dict mapping model_name -> mean_forgetting (scalar).
    """
    out: Dict[str, float] = {}
    for model_name, accs in task_accs.items():
        T = len(accs)
        if T == 0:
            out[model_name] = 0.0
            continue
        forget_sum = 0.0
        for i in range(T):
            max_before = 0.0
            for t in range(i, T - 1):
                if t < len(accs) and i < len(accs[t]):
                    max_before = max(max_before, accs[t][i])
            acc_final = accs[T - 1][i] if i < len(accs[T - 1]) else 0.0
            forget_sum += max(0.0, max_before - acc_final)
        out[model_name] = forget_sum / T
    return out


def compute_backward_transfer(task_accs: Dict[str, List[List[float]]]) -> Dict[str, float]:
    """
    BWT = (1/(T-1)) * sum_{i<T} (acc_after_all_tasks[i] - acc_right_after_task_i[i]).
    Negative BWT indicates forgetting.

    Returns:
        Dict mapping model_name -> BWT (scalar).
    """
    out: Dict[str, float] = {}
    for model_name, accs in task_accs.items():
        T = len(accs)
        if T <= 1:
            out[model_name] = 0.0
            continue
        bwt_sum = 0.0
        for i in range(T - 1):
            acc_final = accs[T - 1][i] if i < len(accs[T - 1]) else 0.0
            acc_right_after = accs[i][i] if i < len(accs[i]) else 0.0
            bwt_sum += acc_final - acc_right_after
        out[model_name] = bwt_sum / (T - 1)
    return out


def compute_forward_transfer(
    task_accs: Dict[str, List[List[float]]],
    random_baseline: float = 0.5,
) -> Dict[str, float]:
    """
    FWT = (1/(T-1)) * sum_{i>0} (acc_on_task_i_before_training_it - random_baseline).
    Uses random_baseline = 0.5 for binary classification.

    Returns:
        Dict mapping model_name -> FWT (scalar).
    """
    out: Dict[str, float] = {}
    for model_name, accs in task_accs.items():
        T = len(accs)
        if T <= 1:
            out[model_name] = 0.0
            continue
        fwt_sum = 0.0
        for i in range(1, T):
            acc_before = accs[i - 1][i] if i < len(accs[i - 1]) else random_baseline
            fwt_sum += acc_before - random_baseline
        out[model_name] = fwt_sum / (T - 1)
    return out


def compile_results_table(
    task_accs: Dict[str, List[List[float]]],
    forgetting: Dict[str, float],
    bwt: Dict[str, float],
    fwt: Dict[str, float],
    save_path: str | Path,
) -> pd.DataFrame:
    """
    Build a results table with columns: Model, Avg_Final_Acc, Forgetting, BWT, FWT.
    Avg_Final_Acc = mean of accuracy on each task after all tasks (last row).
    Saves to save_path (e.g. results_dir/results_table.csv).

    Returns:
        DataFrame with one row per model.
    """
    rows = []
    for model_name in task_accs:
        accs = task_accs[model_name]
        T = len(accs)
        if T > 0 and len(accs[T - 1]) > 0:
            avg_final = sum(accs[T - 1]) / len(accs[T - 1])
        else:
            avg_final = 0.0
        rows.append({
            "Model": model_name,
            "Avg_Final_Acc": avg_final,
            "Forgetting": forgetting.get(model_name, 0.0),
            "BWT": bwt.get(model_name, 0.0),
            "FWT": fwt.get(model_name, 0.0),
        })
    df = pd.DataFrame(rows)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return df
