"""
Standalone script: load NCG and StaticMLP-256 checkpoints, build per-seed
accuracy matrices, compute forgetting and FWT, run Welch t-tests.
Supports --benchmark split_mnist (default) or split_cifar10.
"""

import argparse
import os
from pathlib import Path

import torch
from scipy import stats

from evaluate import compute_forgetting, compute_forward_transfer
from model import NCGModel, NCGModelCNN, SimpleCNN, StaticMLP
from train import evaluate, get_device, get_split_cifar10_tasks, get_split_mnist_tasks, set_seed


def load_ncg_checkpoint(path: str, device: torch.device) -> NCGModel:
    """Load NCG checkpoint; dict has 'state_dict' and optionally 'hidden_size'."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["state_dict"]
    hidden_size = ckpt.get("hidden_size", 256)
    model = NCGModel(hidden_size=hidden_size, num_classes=2, max_hidden=512)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def load_ncg_cnn_checkpoint(path: str, device: torch.device) -> NCGModelCNN:
    """Load NCG CNN checkpoint (Split-CIFAR-10); dict has 'state_dict' and optionally 'hidden_size'."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt["state_dict"]
    hidden_size = ckpt.get("hidden_size", 256)
    model = NCGModelCNN(hidden_size=hidden_size, num_classes=2, max_hidden=512)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def load_static_mlp_checkpoint(path: str, device: torch.device) -> StaticMLP:
    """Load StaticMLP-256 checkpoint; dict has 'state_dict'."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = StaticMLP(hidden_size=256, num_classes=2)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device)


def load_simple_cnn_checkpoint(path: str, device: torch.device) -> SimpleCNN:
    """Load SimpleCNN-256 checkpoint (Split-CIFAR-10); dict has 'state_dict'."""
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
    """Build 2D list accs[t][i] = accuracy on task i after training through task t."""
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
    parser.add_argument("--benchmark", type=str, default="split_mnist", choices=["split_mnist", "split_cifar10"], help="Benchmark (default: split_mnist)")
    args = parser.parse_args()
    benchmark = args.benchmark

    seeds = [42, 43, 44, 45, 46]
    num_tasks = 5
    device = get_device()

    if benchmark == "split_cifar10":
        # Split-CIFAR-10: use CNN loaders and CIFAR-10 checkpoint paths only
        get_tasks = get_split_cifar10_tasks
        load_ncg_fn = load_ncg_cnn_checkpoint
        load_mlp_fn = load_simple_cnn_checkpoint
        ncg_dir_tpl = "./checkpoints/ncg_cifar10_seed{seed}"
        mlp_dir_tpl = "./checkpoints/static_mlp_256_cifar10_seed{seed}"
        assert load_ncg_fn is load_ncg_cnn_checkpoint, "split_cifar10 must use load_ncg_cnn_checkpoint"
        assert load_mlp_fn is load_simple_cnn_checkpoint, "split_cifar10 must use load_simple_cnn_checkpoint"
    else:
        get_tasks = get_split_mnist_tasks
        load_ncg_fn = load_ncg_checkpoint
        load_mlp_fn = load_static_mlp_checkpoint
        ncg_dir_tpl = "./checkpoints/ncg_seed{seed}"
        mlp_dir_tpl = "./checkpoints/static_mlp_256_seed{seed}"

    forget_ncg = []
    forget_mlp = []
    fwt_ncg = []
    fwt_mlp = []

    for seed in seeds:
        set_seed(seed)
        tasks = get_tasks(data_dir="./data", batch_size=64)
        if len(tasks) != num_tasks:
            raise RuntimeError(f"Expected {num_tasks} tasks, got {len(tasks)}")

        ncg_dir = ncg_dir_tpl.format(seed=seed)
        mlp_dir = mlp_dir_tpl.format(seed=seed)

        accs_ncg = build_accs_from_checkpoints(
            ncg_dir,
            range(num_tasks),
            load_ncg_fn,
            device,
            tasks,
            "ncg",
        )
        accs_mlp = build_accs_from_checkpoints(
            mlp_dir,
            range(num_tasks),
            load_mlp_fn,
            device,
            tasks,
            "static_mlp",
        )

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
