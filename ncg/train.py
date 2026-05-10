"""
NCG Research Prototype — Training infrastructure.

- Split-MNIST data loading (5 binary tasks)
- Device selection (CUDA / MPS / CPU)
- Seed management for reproducibility
- Training loops: NCG, StaticMLP, EWC
- Evaluation helper and checkpoint saving
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ncg.model import NCGModel, NCGModelCNN, DENModel, DENModelCNN, EWC, StaticMLP


# -----------------------------------------------------------------------------
# Data: Split-MNIST
# -----------------------------------------------------------------------------

SPLIT_MNIST_TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


def _filter_mnist_by_digits(
    dataset: datasets.MNIST,
    digit_a: int,
    digit_b: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract samples with labels in {digit_a, digit_b} and relabel to {0, 1}."""
    mask = (dataset.targets == digit_a) | (dataset.targets == digit_b)
    indices = torch.where(mask)[0]
    x = dataset.data[indices].float() / 255.0
    y = dataset.targets[indices]
    y = (y == digit_b).long()
    return x, y


def get_split_mnist_tasks(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.15,
    num_workers: int = 0,
) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Download MNIST and create 5 binary classification tasks.
    Each task returns (train_loader, val_loader, test_loader) with relabeled targets {0, 1}.
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_mnist = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    tasks: List[Tuple[DataLoader, DataLoader, DataLoader]] = []
    for digit_a, digit_b in SPLIT_MNIST_TASKS:
        x_train_full = []
        y_train_full = []
        for i in range(len(train_mnist)):
            img, t = train_mnist[i]
            if t == digit_a or t == digit_b:
                x_train_full.append(img)
                y_train_full.append(1 if t == digit_b else 0)
        x_train_full = torch.stack(x_train_full)
        y_train_full = torch.tensor(y_train_full, dtype=torch.long)

        x_test_list = []
        y_test_list = []
        for i in range(len(test_mnist)):
            img, t = test_mnist[i]
            if t == digit_a or t == digit_b:
                x_test_list.append(img)
                y_test_list.append(1 if t == digit_b else 0)
        x_test = torch.stack(x_test_list)
        y_test = torch.tensor(y_test_list, dtype=torch.long)

        n = len(x_train_full)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        indices = torch.randperm(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_dataset = torch.utils.data.TensorDataset(
            x_train_full[train_idx],
            y_train_full[train_idx],
        )
        val_dataset = torch.utils.data.TensorDataset(
            x_train_full[val_idx],
            y_train_full[val_idx],
        )
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        tasks.append((train_loader, val_loader, test_loader))
    return tasks


# -----------------------------------------------------------------------------
# Data: Split-CIFAR-10
# -----------------------------------------------------------------------------

SPLIT_CIFAR10_TASKS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_split_cifar10_tasks(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_split: float = 0.15,
    num_workers: int = 0,
) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Download CIFAR-10 and create 5 binary classification tasks (0/1, 2/3, 4/5, 6/7, 8/9).
    Same return format as get_split_mnist_tasks: list of (train_loader, val_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_cifar = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_cifar = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    tasks: List[Tuple[DataLoader, DataLoader, DataLoader]] = []
    for class_a, class_b in SPLIT_CIFAR10_TASKS:
        x_train_full = []
        y_train_full = []
        for i in range(len(train_cifar)):
            img, t = train_cifar[i]
            if t == class_a or t == class_b:
                x_train_full.append(img)
                y_train_full.append(1 if t == class_b else 0)
        x_train_full = torch.stack(x_train_full)
        y_train_full = torch.tensor(y_train_full, dtype=torch.long)

        x_test_list = []
        y_test_list = []
        for i in range(len(test_cifar)):
            img, t = test_cifar[i]
            if t == class_a or t == class_b:
                x_test_list.append(img)
                y_test_list.append(1 if t == class_b else 0)
        x_test = torch.stack(x_test_list)
        y_test = torch.tensor(y_test_list, dtype=torch.long)

        n = len(x_train_full)
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        indices = torch.randperm(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_dataset = torch.utils.data.TensorDataset(
            x_train_full[train_idx],
            y_train_full[train_idx],
        )
        val_dataset = torch.utils.data.TensorDataset(
            x_train_full[val_idx],
            y_train_full[val_idx],
        )
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        tasks.append((train_loader, val_loader, test_loader))
    return tasks


# -----------------------------------------------------------------------------
# Device and seed
# -----------------------------------------------------------------------------


def get_device() -> torch.device:
    """Auto-detect CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Evaluation and checkpoints
# -----------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    is_ncg: bool = False,
) -> Tuple[float, float]:
    """Returns (accuracy, mean_loss). For NCG, forward returns (logits, h); use logits only."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            if is_ncg:
                logits, _ = model(x)
            else:
                logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    model.train()
    return correct / max(total, 1), total_loss / max(total, 1)


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    task_id: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model state_dict and optional metadata (e.g. hidden_size for NCG)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "task_id": task_id}
    if isinstance(model, (NCGModel, NCGModelCNN, DENModel, DENModelCNN)):
        payload["hidden_size"] = model.hidden_size
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


# -----------------------------------------------------------------------------
# Training: NCG
# -----------------------------------------------------------------------------


def train_ncg(
    model: nn.Module,
    tasks: List[Tuple[DataLoader, DataLoader, DataLoader]],
    device: torch.device,
    epochs_per_task: int = 20,
    lr: float = 1e-3,
    lr_meta: float = 0.01,
    num_classes: int = 2,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "Model",
    seed: int = 0,
    disable_growth: bool = False,
    task_pairs: Optional[List[Tuple[int, int]]] = None,
    adapter: Optional[Any] = None,
    meta: Optional[Any] = None,
    novelty_layer_getter: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train NCG sequentially on each task. Works with NCGModel/NCGModelCNN (default)
    or any PyTorch model when adapter/meta/novelty_layer_getter are provided.
    """
    model = model.to(device)
    is_ncg_model = isinstance(model, (NCGModel, NCGModelCNN))

    if is_ncg_model:
        weight_params = model.get_weight_params()
        meta_params = model.get_meta_params()
        _meta = None
    elif meta is not None:
        weight_params = [p for p in model.parameters() if p.requires_grad]
        meta_params = meta.get_params()
        _meta = meta
    else:
        weight_params = [p for p in model.parameters() if p.requires_grad]
        meta_params = []
        _meta = None

    if is_ncg_model:
        _novelty_monitor = None
    elif novelty_layer_getter is not None:
        from ncg.novelty import NoveltyMonitor
        _novelty_monitor = NoveltyMonitor(model, novelty_layer_getter)
    else:
        _novelty_monitor = None

    lr_weights = 0.0005
    opt = torch.optim.Adam(weight_params, lr=lr_weights)
    if meta_params:
        opt_meta = torch.optim.Adam(meta_params, lr=lr_meta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_per_task, eta_min=1e-5)
    num_tasks = len(tasks)

    results: Dict[str, Any] = {
        "task_accs": [],
        "val_accs_per_epoch": [],
        "novelty_per_epoch": [],
        "hidden_size_per_epoch": [],
        "alpha_per_epoch": [],
        "beta_per_epoch": [],
        "lambda_per_epoch": [],
    }

    task_pairs = task_pairs or SPLIT_MNIST_TASKS
    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        class_a, class_b = task_pairs[task_id]
        class_pair = f"{class_a}/{class_b}"
        if verbose:
            print(f"============================================================")
            print(f"[Task {task_id + 1}/{num_tasks} starting] Classes: {class_pair}")
            print(f"============================================================")

        recent_val_accs: List[float] = []
        for epoch in range(epochs_per_task):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits, h = model(x)
                if is_ncg_model:
                    loss = model.compute_training_loss(logits, y, num_classes=num_classes)
                elif _meta is not None:
                    loss = _meta.compute_training_loss(logits, y, model)
                else:
                    loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                if is_ncg_model:
                    model.update_knowledge(h)

            train_acc, _ = evaluate(model, train_loader, device, is_ncg=True)
            val_acc, _ = evaluate(model, val_loader, device, is_ncg=True)
            recent_val_accs.append(val_acc)
            results["val_accs_per_epoch"].append(val_acc)

            val_x_list = []
            val_y_list = []
            for x, y in val_loader:
                val_x_list.append(x)
                val_y_list.append(y)
            val_x = torch.cat(val_x_list, dim=0).to(device)
            val_y = torch.cat(val_y_list, dim=0).to(device)
            model.train()
            logits_meta, _ = model(val_x)
            if is_ncg_model and meta_params:
                opt_meta.zero_grad()
                meta_loss = model.compute_meta_loss(logits_meta, val_y, num_classes=num_classes)
                (-meta_loss).backward()
                opt_meta.step()
            elif _meta is not None and meta_params:
                opt_meta.zero_grad()
                meta_loss = _meta.compute_meta_loss(logits_meta, val_y, num_classes=num_classes, model=model)
                (-meta_loss).backward()
                opt_meta.step()

            if is_ncg_model:
                novelty = model.compute_novelty(logits_meta.detach(), num_classes).item()
            elif _novelty_monitor is not None:
                novelty = _novelty_monitor.compute(val_loader, device)
            else:
                novelty = 0.3

            results["novelty_per_epoch"].append(novelty)
            current_hidden = model.hidden_size if is_ncg_model else (adapter.current_size(model) if adapter is not None else 0)
            results["hidden_size_per_epoch"].append(current_hidden)

            if is_ncg_model:
                results["alpha_per_epoch"].append(model.alpha.item())
                results["beta_per_epoch"].append(model.beta.item())
                results["lambda_per_epoch"].append(model.lambda_.item())
            elif _meta is not None:
                snap = _meta.snapshot()
                results["alpha_per_epoch"].append(snap["alpha"])
                results["beta_per_epoch"].append(snap["beta"])
                results["lambda_per_epoch"].append(snap["lambda"])
            else:
                results["alpha_per_epoch"].append(0.0)
                results["beta_per_epoch"].append(0.0)
                results["lambda_per_epoch"].append(0.0)

            if verbose:
                print(f"[Seed {seed} | Model {model_name} | Task {task_id + 1}/{num_tasks} | Epoch {epoch + 1}/{epochs_per_task}] Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Novelty: {novelty:.2f}")
                if is_ncg_model:
                    print(f"[NCG] Hidden units: {model.hidden_size} | α: {model.alpha.item():.2f} | β: {model.beta.item():.2f} | λ: {model.lambda_.item():.2f}")
                elif _meta is not None:
                    print(f"[NCG] Hidden units: {current_hidden} | α: {snap['alpha']:.2f} | β: {snap['beta']:.2f} | λ: {snap['lambda']:.2f}")

            scheduler.step()

            if not disable_growth:
                if is_ncg_model:
                    lam = model.lambda_.item()
                elif _meta is not None:
                    lam = _meta.lambda_.item()
                else:
                    lam = 0.5

                if is_ncg_model:
                    should_grow = model.check_growth_trigger(recent_val_accs, novelty, verbose=verbose)
                else:
                    from ncg.adapters import GrowthAdapter
                    should_grow = (
                        adapter is not None
                        and novelty < 0.5
                        and lam > 0.3
                        and len(recent_val_accs) >= 3
                        and (max(recent_val_accs[-3:]) - min(recent_val_accs[-3:])) < 0.005
                    )

                if should_grow:
                    old_size = model.hidden_size if is_ncg_model else adapter.current_size(model)
                    if is_ncg_model:
                        model.grow(64)
                    elif adapter is not None:
                        adapter.expand(model)
                    new_size = model.hidden_size if is_ncg_model else adapter.current_size(model)
                    if verbose:
                        print(f"[NCG] Growth: {old_size} → {new_size} (task {task_id+1}, epoch {epoch+1})")
                    weight_params = model.get_weight_params() if is_ncg_model else [p for p in model.parameters() if p.requires_grad]
                    meta_params_new = model.get_meta_params() if is_ncg_model else (_meta.get_params() if _meta else [])
                    opt = torch.optim.Adam(weight_params, lr=lr_weights)
                    if meta_params_new:
                        opt_meta = torch.optim.Adam(meta_params_new, lr=lr_meta)
                    remaining = epochs_per_task - (epoch + 1)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, remaining), eta_min=1e-5)

        task_accs_so_far: List[float] = []
        for ti in range(len(tasks)):
            _, _, test_loader_t = tasks[ti]
            acc, _ = evaluate(model, test_loader_t, device, is_ncg=True)
            task_accs_so_far.append(acc)
        results["task_accs"].append(task_accs_so_far)

        if checkpoint_dir:
            save_checkpoint(
                model,
                os.path.join(checkpoint_dir, f"ncg_task_{task_id}.pt"),
                task_id=task_id,
            )

    return results


# -----------------------------------------------------------------------------
# Training: Static MLP
# -----------------------------------------------------------------------------


def train_static_mlp(
    model: StaticMLP,
    tasks: List[Tuple[DataLoader, DataLoader, DataLoader]],
    device: torch.device,
    epochs_per_task: int = 20,
    lr: float = 1e-3,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "Model",
    seed: int = 0,
    task_pairs: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Standard sequential training; no continual-learning mechanism."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    results: Dict[str, Any] = {"task_accs": []}
    num_tasks = len(tasks)
    task_pairs = task_pairs or SPLIT_MNIST_TASKS

    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        class_a, class_b = task_pairs[task_id]
        class_pair = f"{class_a}/{class_b}"
        if verbose:
            print(f"============================================================")
            print(f"[Task {task_id + 1}/{num_tasks} starting] Classes: {class_pair}")
            print(f"============================================================")

        for epoch in range(epochs_per_task):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

            train_acc, _ = evaluate(model, train_loader, device, is_ncg=False)
            val_acc, _ = evaluate(model, val_loader, device, is_ncg=False)
            if verbose:
                print(f"[Seed {seed} | Model {model_name} | Task {task_id + 1}/{num_tasks} | Epoch {epoch + 1}/{epochs_per_task}] Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Novelty: N/A")

        task_accs_so_far = []
        for ti in range(len(tasks)):
            _, _, test_loader_t = tasks[ti]
            acc, _ = evaluate(model, test_loader_t, device, is_ncg=False)
            task_accs_so_far.append(acc)
        results["task_accs"].append(task_accs_so_far)

        if checkpoint_dir:
            save_checkpoint(
                model,
                os.path.join(checkpoint_dir, f"static_mlp_task_{task_id}.pt"),
                task_id=task_id,
            )

    return results


# -----------------------------------------------------------------------------
# Training: DEN (Dynamically Expandable Networks)
# -----------------------------------------------------------------------------


def train_den(
    model: DENModel,
    tasks: List[Tuple[DataLoader, DataLoader, DataLoader]],
    device: torch.device,
    epochs_per_task: int = 20,
    lr: float = 1e-3,
    val_loss_threshold: float = 0.3,
    retrain_epochs: int = 5,
    grow_units: int = 64,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "Model",
    seed: int = 0,
    task_pairs: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train DEN like StaticMLP; after each task, if val CE loss > val_loss_threshold,
    call model.grow(grow_units) and retrain the same task for retrain_epochs.
    """
    if not isinstance(model, (DENModel, DENModelCNN)):
        raise TypeError(f"train_den expects DENModel or DENModelCNN, got {type(model).__name__}")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    results: Dict[str, Any] = {"task_accs": []}
    num_tasks = len(tasks)
    task_pairs = task_pairs or SPLIT_MNIST_TASKS

    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        class_a, class_b = task_pairs[task_id]
        class_pair = f"{class_a}/{class_b}"
        if verbose:
            print(f"============================================================")
            print(f"[Task {task_id + 1}/{num_tasks} starting] Classes: {class_pair}")
            print(f"============================================================")

        for epoch in range(epochs_per_task):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

            train_acc, _ = evaluate(model, train_loader, device, is_ncg=False)
            val_acc, val_loss = evaluate(model, val_loader, device, is_ncg=False)
            if verbose:
                print(f"[Seed {seed} | Model {model_name} | Task {task_id + 1}/{num_tasks} | Epoch {epoch + 1}/{epochs_per_task}] Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Val Loss: {val_loss:.4f}")

        _, val_loss = evaluate(model, val_loader, device, is_ncg=False)
        if val_loss > val_loss_threshold:
            old_size = model.hidden_size
            model.grow(grow_units)
            new_size = model.hidden_size
            if verbose:
                print(f"[DEN GROWTH] Val loss {val_loss:.4f} > {val_loss_threshold}; expanding {old_size} → {new_size}, retraining {retrain_epochs} epochs")
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for epoch in range(retrain_epochs):
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.backward()
                    opt.step()
                train_acc, _ = evaluate(model, train_loader, device, is_ncg=False)
                val_acc, val_loss = evaluate(model, val_loader, device, is_ncg=False)
                if verbose:
                    print(f"[DEN retrain | Task {task_id + 1} | Epoch {epoch + 1}/{retrain_epochs}] Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Val Loss: {val_loss:.4f}")

        task_accs_so_far = []
        for ti in range(len(tasks)):
            _, _, test_loader_t = tasks[ti]
            acc, _ = evaluate(model, test_loader_t, device, is_ncg=False)
            task_accs_so_far.append(acc)
        results["task_accs"].append(task_accs_so_far)

        if checkpoint_dir:
            save_checkpoint(
                model,
                os.path.join(checkpoint_dir, f"den_task_{task_id}.pt"),
                task_id=task_id,
            )

    return results


# -----------------------------------------------------------------------------
# Training: EWC
# -----------------------------------------------------------------------------


def train_ewc(
    model: EWC,
    tasks: List[Tuple[DataLoader, DataLoader, DataLoader]],
    device: torch.device,
    epochs_per_task: int = 20,
    lr: float = 1e-3,
    ewc_lambda: float = 400.0,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "Model",
    seed: int = 0,
    task_pairs: Optional[List[Tuple[int, int]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train with EWC penalty after consolidating Fisher per task."""
    if not isinstance(model, EWC):
        raise TypeError(f"train_ewc expects EWC model, got {type(model).__name__}")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    results: Dict[str, Any] = {"task_accs": []}
    num_tasks = len(tasks)
    task_pairs = task_pairs or SPLIT_MNIST_TASKS

    for task_id, (train_loader, val_loader, test_loader) in enumerate(tasks):
        class_a, class_b = task_pairs[task_id]
        class_pair = f"{class_a}/{class_b}"
        if verbose:
            print(f"============================================================")
            print(f"[Task {task_id + 1}/{num_tasks} starting] Classes: {class_pair}")
            print(f"============================================================")

        for epoch in range(epochs_per_task):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y) + model.compute_ewc_loss()
                loss.backward()
                opt.step()

            train_acc, _ = evaluate(model, train_loader, device, is_ncg=False)
            val_acc, _ = evaluate(model, val_loader, device, is_ncg=False)
            if verbose:
                print(f"[Seed {seed} | Model {model_name} | Task {task_id + 1}/{num_tasks} | Epoch {epoch + 1}/{epochs_per_task}] Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f} | Novelty: N/A")

        model.consolidate(train_loader, device)

        task_accs_so_far = []
        for ti in range(len(tasks)):
            _, _, test_loader_t = tasks[ti]
            acc, _ = evaluate(model, test_loader_t, device, is_ncg=False)
            task_accs_so_far.append(acc)
        results["task_accs"].append(task_accs_so_far)

        if checkpoint_dir:
            save_checkpoint(
                model,
                os.path.join(checkpoint_dir, f"ewc_task_{task_id}.pt"),
                task_id=task_id,
            )

    return results
