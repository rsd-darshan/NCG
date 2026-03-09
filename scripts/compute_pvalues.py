"""
Standalone script: load NCG and StaticMLP-256 checkpoints, build per-seed
accuracy matrices, compute forgetting and FWT, run Welch t-tests.
Supports --benchmark split_mnist (default) or split_cifar10.
With --convergence: run full convergence analysis and save results/convergence_analysis.json.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ncg.math.convergence import run_full_analysis
from ncg.metrics import compute_forgetting, compute_forward_transfer
from ncg.model import NCGModel, NCGModelCNN, SimpleCNN, StaticMLP
from ncg.train import evaluate, get_device, get_split_cifar10_tasks, get_split_mnist_tasks, set_seed


def load_ncg_checkpoint(path: str, device: torch.device) -> NCGModel:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["state_dict"]
    hidden_size = ckpt.get("hidden_size", 256)
    model = NCGModel(hidden_size=hidden_size, num_classes=2, max_hidden=512)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def load_ncg_cnn_checkpoint(path: str, device: torch.device) -> NCGModelCNN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["state_dict"]
    hidden_size = ckpt.get("hidden_size", 256)
    model = NCGModelCNN(hidden_size=hidden_size, num_classes=2, max_hidden=512)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def load_static_mlp_checkpoint(path: str, device: torch.device) -> StaticMLP:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = StaticMLP(hidden_size=256, num_classes=2)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device)


def load_simple_cnn_checkpoint(path: str, device: torch.device) -> SimpleCNN:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = SimpleCNN(hidden_size=256, num_classes=2)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device)


def build_accs_from_checkpoints(
    checkpoint_dir: str,
    task_indices: range,
    load_fn,
    device: torch.device,
    tasks: list,
    file_prefix: str,
    is_ncg: bool = None,
) -> list:
    accs = []
    if is_ncg is None:
        is_ncg = load_fn in (load_ncg_checkpoint, load_ncg_cnn_checkpoint)
    for t in task_indices:
        path = os.path.join(checkpoint_dir, f"{file_prefix}_task_{t}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model = load_fn(path, device)
        model.eval()
        row = []
        for i in range(len(tasks)):
            _, _, test_loader = tasks[i]
            acc, _ = evaluate(model, test_loader, device, is_ncg=is_ncg)
            row.append(acc)
        accs.append(row)
    return accs


def main():
    parser = argparse.ArgumentParser(description="Compute p-values for NCG vs StaticMLP-256 (forgetting, FWT)")
    parser.add_argument("--benchmark", type=str, default="split_mnist", choices=["split_mnist", "split_cifar10"])
    parser.add_argument("--convergence", action="store_true", help="Run full convergence analysis and save results/convergence_analysis.json")
    parser.add_argument("--results_dir", type=str, default="./results", help="Results directory (for --convergence: ncg_logs.pkl and checkpoints)")
    args = parser.parse_args()
    benchmark = args.benchmark
    results_dir = Path(args.results_dir)

    seeds = [42, 43, 44, 45, 46]
    num_tasks = 5
    device = get_device()

    if benchmark == "split_cifar10":
        get_tasks = get_split_cifar10_tasks
        load_ncg_fn = load_ncg_cnn_checkpoint
        load_mlp_fn = load_simple_cnn_checkpoint
        ckpt_prefix = "cifar10_"
        ncg_dir_tpl = "./checkpoints/ncg_cifar10_seed{seed}"
        mlp_dir_tpl = "./checkpoints/static_mlp_256_cifar10_seed{seed}"
    else:
        get_tasks = get_split_mnist_tasks
        load_ncg_fn = load_ncg_checkpoint
        load_mlp_fn = load_static_mlp_checkpoint
        ckpt_prefix = ""
        ncg_dir_tpl = "./checkpoints/ncg_seed{seed}"
        mlp_dir_tpl = "./checkpoints/static_mlp_256_seed{seed}"

    if args.convergence:
        ncg_logs_path = results_dir / "ncg_logs.pkl"
        if not ncg_logs_path.exists():
            raise FileNotFoundError(
                f"ncg_logs not found at {ncg_logs_path}. Run scripts/main.py first to generate results."
            )
        with open(ncg_logs_path, "rb") as f:
            data = pickle.load(f)
        all_ncg_logs = data["ncg_logs"]
        seed_list = data.get("seed_list", seeds)
        first_seed = seed_list[0]
        ncg_logs_first = all_ncg_logs[0]
        tasks = get_tasks(data_dir="./data", batch_size=64)
        val_loader = tasks[0][1]
        ckpt_dir = results_dir / "checkpoints" / f"{ckpt_prefix}ncg_seed{first_seed}"
        last_task = len(tasks) - 1
        ckpt_path = ckpt_dir / f"ncg_task_{last_task}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run scripts/main.py first.")
        model = load_ncg_fn(str(ckpt_path), device)
        analysis = run_full_analysis(model, ncg_logs_first, val_loader, device)
        out_path = results_dir / "convergence_analysis.json"
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"Convergence analysis saved to {out_path}")
        return

    forget_ncg, forget_mlp, fwt_ncg, fwt_mlp = [], [], [], []

    for seed in seeds:
        set_seed(seed)
        tasks = get_tasks(data_dir="./data", batch_size=64)
        if len(tasks) != num_tasks:
            raise RuntimeError(f"Expected {num_tasks} tasks, got {len(tasks)}")
        ncg_dir = ncg_dir_tpl.format(seed=seed)
        mlp_dir = mlp_dir_tpl.format(seed=seed)
        accs_ncg = build_accs_from_checkpoints(ncg_dir, range(num_tasks), load_ncg_fn, device, tasks, "ncg")
        accs_mlp = build_accs_from_checkpoints(mlp_dir, range(num_tasks), load_mlp_fn, device, tasks, "static_mlp")
        forget_ncg.append(compute_forgetting({"NCG": accs_ncg})["NCG"])
        forget_mlp.append(compute_forgetting({"StaticMLP-256": accs_mlp})["StaticMLP-256"])
        fwt_ncg.append(compute_forward_transfer({"NCG": accs_ncg})["NCG"])
        fwt_mlp.append(compute_forward_transfer({"StaticMLP-256": accs_mlp})["StaticMLP-256"])

    print("Per-seed Forgetting:")
    print("  Seed    NCG     StaticMLP-256")
    for i, s in enumerate(seeds):
        print(f"  {s:<6}  {forget_ncg[i]:.4f}  {forget_mlp[i]:.4f}")
    print()
    print("Per-seed FWT:")
    print("  Seed    NCG     StaticMLP-256")
    for i, s in enumerate(seeds):
        print(f"  {s:<6}  {fwt_ncg[i]:.4f}  {fwt_mlp[i]:.4f}")
    print()
    t_forget, p_forget = stats.ttest_ind(forget_ncg, forget_mlp, equal_var=False, alternative="less")
    t_fwt, p_fwt = stats.ttest_ind(fwt_ncg, fwt_mlp, equal_var=False, alternative="greater")
    print("Welch one-tailed t-tests:")
    print("  NCG forgetting < StaticMLP-256 forgetting:  t = {:.4f},  p = {:.4f}".format(t_forget, p_forget))
    print("  NCG FWT > StaticMLP-256 FWT:                t = {:.4f},  p = {:.4f}".format(t_fwt, p_fwt))


if __name__ == "__main__":
    main()
