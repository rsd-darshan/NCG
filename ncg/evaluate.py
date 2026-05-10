"""
NCG Research Prototype — Evaluation and results aggregation.

- run_all_seeds: full experiment loop over seeds with mean ± std aggregation
- Uses ncg.metrics for forgetting, BWT, FWT, compile_results_table
"""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ncg.metrics import (
    compile_results_table,
    compute_backward_transfer,
    compute_forgetting,
    compute_forward_transfer,
)


def _import_train_and_models():
    from ncg.model import NCGModel, DENModel, EWC, StaticMLP
    from ncg.train import (
        get_device,
        set_seed,
        train_den,
        train_ewc,
        train_ncg,
        train_static_mlp,
    )
    return NCGModel, DENModel, EWC, StaticMLP, set_seed, train_ncg, train_den, train_static_mlp, train_ewc


def run_all_seeds(
    seed_list: List[int],
    tasks_fn: Callable[[], List[Any]],
    device: Any,
    results_dir: str | Path,
    data_dir: str = "./data",
    checkpoint_dir: str | Path | None = None,
    epochs_per_task: int = 10,
    batch_size: int = 64,
    benchmark: str = "split_mnist",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full experiment for each seed: train all seven models (NCG, NCG-NoGrowth,
    NCG-FixedMeta, DEN, StaticMLP-256, StaticMLP-512, EWC), collect task_accs and NCG logs,
    then aggregate mean ± std across seeds. Saves aggregated_results.csv to results_dir.

    When benchmark is split_cifar10, uses CNN model classes (NCGModelCNN, SimpleCNN, etc.).
    tasks_fn is a callable that returns the list of (train_loader, val_loader, test_loader)
    per task. Seed should be set before calling tasks_fn if you want different train/val splits per seed.

    Returns:
        Dict with keys: task_accs (per-seed and aggregated), ncg_logs (per-seed),
        forgetting, bwt, fwt (aggregated mean ± std), and the results table DataFrame.
    """
    from ncg.model import DENModelCNN, NCGModelCNN, SimpleCNN
    from ncg.train import SPLIT_CIFAR10_TASKS, SPLIT_MNIST_TASKS

    NCGModel, DENModel, EWC, StaticMLP, set_seed, train_ncg, train_den, train_static_mlp, train_ewc = _import_train_and_models()
    if benchmark == "split_cifar10":
        NCG_cls = NCGModelCNN
        Static_cls = SimpleCNN
        DEN_cls = DENModelCNN
        task_pairs = SPLIT_CIFAR10_TASKS
    else:
        NCG_cls = NCGModel
        Static_cls = StaticMLP
        DEN_cls = DENModel
        task_pairs = SPLIT_MNIST_TASKS
    ckpt_prefix = "cifar10_" if benchmark == "split_cifar10" else ""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is None:
        checkpoint_dir = results_dir / "checkpoints"

    all_task_accs: Dict[str, List[List[List[float]]]] = {
        "NCG": [],
        "NCG-NoGrowth": [],
        "NCG-FixedMeta": [],
        "DEN": [],
        "StaticMLP-256": [],
        "StaticMLP-512": [],
        "EWC": [],
    }
    all_ncg_logs: List[Dict[str, Any]] = []

    for seed in seed_list:
        set_seed(seed)
        tasks = tasks_fn()
        T = len(tasks)

        # NCG
        ncg = NCG_cls(hidden_size=256, num_classes=2, max_hidden=512)
        ckpt_ncg = str(Path(checkpoint_dir) / f"{ckpt_prefix}ncg_seed{seed}")
        Path(ckpt_ncg).mkdir(parents=True, exist_ok=True)
        ncg_res = train_ncg(ncg, tasks, device, epochs_per_task=epochs_per_task, checkpoint_dir=ckpt_ncg, model_name="NCG", seed=seed, task_pairs=task_pairs)
        all_task_accs["NCG"].append(ncg_res["task_accs"])
        all_ncg_logs.append({
            "hidden_size_per_epoch": ncg_res["hidden_size_per_epoch"],
            "alpha_per_epoch": ncg_res["alpha_per_epoch"],
            "beta_per_epoch": ncg_res["beta_per_epoch"],
            "lambda_per_epoch": ncg_res["lambda_per_epoch"],
            "novelty_per_epoch": ncg_res["novelty_per_epoch"],
            "epochs_per_task": epochs_per_task,
        })

        # NCG-NoGrowth (growth trigger always skipped)
        ncg_nogrowth = NCG_cls(hidden_size=256, num_classes=2, max_hidden=512)
        ckpt_nogrowth = str(Path(checkpoint_dir) / f"{ckpt_prefix}ncg_nogrowth_seed{seed}")
        Path(ckpt_nogrowth).mkdir(parents=True, exist_ok=True)
        nogrowth_res = train_ncg(ncg_nogrowth, tasks, device, epochs_per_task=epochs_per_task, checkpoint_dir=ckpt_nogrowth, model_name="NCG-NoGrowth", seed=seed, disable_growth=True, task_pairs=task_pairs)
        all_task_accs["NCG-NoGrowth"].append(nogrowth_res["task_accs"])

        # NCG-FixedMeta (α=0.5, β=0.01, λ=0.5, no meta updates)
        ncg_fixedmeta = NCG_cls(hidden_size=256, num_classes=2, max_hidden=512, fixed_meta=(0.5, 0.01, 0.5))
        ckpt_fixedmeta = str(Path(checkpoint_dir) / f"{ckpt_prefix}ncg_fixedmeta_seed{seed}")
        Path(ckpt_fixedmeta).mkdir(parents=True, exist_ok=True)
        fixedmeta_res = train_ncg(ncg_fixedmeta, tasks, device, epochs_per_task=epochs_per_task, checkpoint_dir=ckpt_fixedmeta, model_name="NCG-FixedMeta", seed=seed, task_pairs=task_pairs)
        all_task_accs["NCG-FixedMeta"].append(fixedmeta_res["task_accs"])

        # DEN (Dynamically Expandable Networks)
        den = DEN_cls(hidden_size=256, num_classes=2, max_hidden=512)
        ckpt_den = str(Path(checkpoint_dir) / f"{ckpt_prefix}den_seed{seed}")
        Path(ckpt_den).mkdir(parents=True, exist_ok=True)
        den_res = train_den(den, tasks, device, epochs_per_task=epochs_per_task, checkpoint_dir=ckpt_den, model_name="DEN", seed=seed, task_pairs=task_pairs)
        all_task_accs["DEN"].append(den_res["task_accs"])

        # Static 256 (StaticMLP or SimpleCNN)
        if benchmark == "split_cifar10":
            static256 = Static_cls(hidden_size=256, num_classes=2)
            static512 = Static_cls(hidden_size=512, num_classes=2)
        else:
            static256 = Static_cls(input_size=784, hidden_size=256, num_classes=2)
            static512 = Static_cls(input_size=784, hidden_size=512, num_classes=2)
        ckpt_256 = str(Path(checkpoint_dir) / f"{ckpt_prefix}static_mlp_256_seed{seed}")
        Path(ckpt_256).mkdir(parents=True, exist_ok=True)
        r256 = train_static_mlp(static256, tasks, device, epochs_per_task=epochs_per_task, checkpoint_dir=ckpt_256, model_name="StaticMLP-256", seed=seed, task_pairs=task_pairs)
        all_task_accs["StaticMLP-256"].append(r256["task_accs"])

        ckpt_512 = str(Path(checkpoint_dir) / f"{ckpt_prefix}static_mlp_512_seed{seed}")
        Path(ckpt_512).mkdir(parents=True, exist_ok=True)
        r512 = train_static_mlp(static512, tasks, device, epochs_per_task=epochs_per_task, checkpoint_dir=ckpt_512, model_name="StaticMLP-512", seed=seed, task_pairs=task_pairs)
        all_task_accs["StaticMLP-512"].append(r512["task_accs"])

        # EWC
        if benchmark == "split_cifar10":
            ewc = EWC(hidden_size=256, num_classes=2, backbone="cnn")
        else:
            ewc = EWC(input_size=784, hidden_size=256, num_classes=2, backbone="mlp")
        ckpt_ewc = str(Path(checkpoint_dir) / f"{ckpt_prefix}ewc_seed{seed}")
        Path(ckpt_ewc).mkdir(parents=True, exist_ok=True)
        r_ewc = train_ewc(ewc, tasks, device, epochs_per_task=epochs_per_task, checkpoint_dir=ckpt_ewc, model_name="EWC", seed=seed, task_pairs=task_pairs)
        all_task_accs["EWC"].append(r_ewc["task_accs"])

        print(f"[Seed {seed} complete] Results saved.")

    # Aggregate: for each model, compute mean and std of final-row accuracies and metrics
    n_seeds = len(seed_list)
    task_accs_mean: Dict[str, List[List[float]]] = {}
    task_accs_std: Dict[str, List[List[float]]] = {}
    for model_name, list_of_accs in all_task_accs.items():
        T = len(list_of_accs[0]) if list_of_accs else 0
        mean_grid = []
        std_grid = []
        for t in range(T):
            row_mean = []
            row_std = []
            n_cols = len(list_of_accs[0][t]) if list_of_accs and list_of_accs[0] else 0
            for i in range(n_cols):
                vals = [list_of_accs[s][t][i] for s in range(n_seeds) if s < len(list_of_accs) and t < len(list_of_accs[s]) and i < len(list_of_accs[s][t])]
                row_mean.append(sum(vals) / len(vals) if vals else 0.0)
                if len(vals) > 1:
                    vmean = sum(vals) / len(vals)
                    row_std.append((sum((x - vmean) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5)
                else:
                    row_std.append(0.0)
            mean_grid.append(row_mean)
            std_grid.append(row_std)
        task_accs_mean[model_name] = mean_grid
        task_accs_std[model_name] = std_grid

    task_accs_for_table = task_accs_mean

    forgetting_mean = compute_forgetting(task_accs_mean)
    bwt_mean = compute_backward_transfer(task_accs_mean)
    fwt_mean = compute_forward_transfer(task_accs_mean)

    forget_per_seed: Dict[str, List[float]] = {m: [] for m in all_task_accs}
    bwt_per_seed: Dict[str, List[float]] = {m: [] for m in all_task_accs}
    fwt_per_seed: Dict[str, List[float]] = {m: [] for m in all_task_accs}
    for s in range(n_seeds):
        acc_s = {m: all_task_accs[m][s] for m in all_task_accs}
        f = compute_forgetting(acc_s)
        b = compute_backward_transfer(acc_s)
        fw = compute_forward_transfer(acc_s)
        for m in all_task_accs:
            forget_per_seed[m].append(f[m])
            bwt_per_seed[m].append(b[m])
            fwt_per_seed[m].append(fw[m])

    forgetting_std = {m: (sum((x - forgetting_mean[m]) ** 2 for x in forget_per_seed[m]) / max(1, n_seeds - 1)) ** 0.5 for m in forgetting_mean}
    bwt_std = {m: (sum((x - bwt_mean[m]) ** 2 for x in bwt_per_seed[m]) / max(1, n_seeds - 1)) ** 0.5 for m in bwt_mean}
    fwt_std = {m: (sum((x - fwt_mean[m]) ** 2 for x in fwt_per_seed[m]) / max(1, n_seeds - 1)) ** 0.5 for m in fwt_mean}

    agg_rows = []
    for model_name in task_accs_for_table:
        accs = task_accs_for_table[model_name]
        T = len(accs)
        avg_final = sum(accs[T - 1]) / len(accs[T - 1]) if T > 0 and accs[T - 1] else 0.0
        final_per_seed = []
        for s in range(n_seeds):
            acc_s = all_task_accs[model_name][s]
            if len(acc_s) > 0 and len(acc_s[-1]) > 0:
                final_per_seed.append(sum(acc_s[-1]) / len(acc_s[-1]))
        avg_final_std = (sum((x - avg_final) ** 2 for x in final_per_seed) / max(1, len(final_per_seed) - 1)) ** 0.5 if len(final_per_seed) > 1 else 0.0
        agg_rows.append({
            "Model": model_name,
            "Avg_Final_Acc_mean": avg_final,
            "Avg_Final_Acc_std": avg_final_std,
            "Forgetting_mean": forgetting_mean[model_name],
            "Forgetting_std": forgetting_std[model_name],
            "BWT_mean": bwt_mean[model_name],
            "BWT_std": bwt_std[model_name],
            "FWT_mean": fwt_mean[model_name],
            "FWT_std": fwt_std[model_name],
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_path = results_dir / "aggregated_results.csv"
    agg_df.to_csv(agg_path, index=False)

    table_path = results_dir / "results_table.csv"
    table_df = compile_results_table(task_accs_for_table, forgetting_mean, bwt_mean, fwt_mean, table_path)

    ncg_logs_path = results_dir / "ncg_logs.pkl"
    with open(ncg_logs_path, "wb") as f:
        pickle.dump({"ncg_logs": all_ncg_logs, "seed_list": seed_list}, f)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copytree(results_dir, out / "results", dirs_exist_ok=True)
        shutil.copytree(Path(checkpoint_dir), out / "checkpoints", dirs_exist_ok=True)
        print(f"[run_all_seeds] Copied results and checkpoints to {output_dir}")

    return {
        "task_accs": task_accs_for_table,
        "task_accs_per_seed": all_task_accs,
        "task_accs_std": task_accs_std,
        "ncg_logs": all_ncg_logs,
        "forgetting": forgetting_mean,
        "forgetting_std": forgetting_std,
        "bwt": bwt_mean,
        "bwt_std": bwt_std,
        "fwt": fwt_mean,
        "fwt_std": fwt_std,
        "results_table": table_df,
        "aggregated_df": agg_df,
    }
