"""Pytest tests for ncg.train."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ncg.train import (
    evaluate,
    get_split_mnist_tasks,
    train_ncg,
    train_static_mlp,
)
from ncg.model import NCGModel, StaticMLP


@pytest.mark.integration
def test_get_split_mnist_tasks_returns_5_tasks():
    tasks = get_split_mnist_tasks(data_dir="./data", batch_size=32)
    assert len(tasks) == 5


@pytest.mark.integration
def test_each_task_is_tuple_of_3_dataloaders():
    tasks = get_split_mnist_tasks(data_dir="./data", batch_size=32)
    for t in tasks:
        assert len(t) == 3
        train_loader, val_loader, test_loader = t
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)


def test_evaluate_returns_accuracy_between_0_and_1():
    model = StaticMLP(input_size=16, hidden_size=8, num_classes=2)
    x = torch.randn(8, 16)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    device = torch.device("cpu")
    acc, loss = evaluate(model, loader, device, is_ncg=False)
    assert 0 <= acc <= 1
    assert loss >= 0


def test_train_ncg_runs_one_epoch_per_task_tiny_data():
    device = torch.device("cpu")
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=32)
    # Tiny dummy tasks: 4 samples, 16-dim input
    x1 = torch.randn(4, 16)
    y1 = torch.randint(0, 2, (4,))
    x2 = torch.randn(4, 16)
    y2 = torch.randint(0, 2, (4,))
    tasks = [
        (
            DataLoader(TensorDataset(x1, y1), batch_size=2),
            DataLoader(TensorDataset(x1, y1), batch_size=2),
            DataLoader(TensorDataset(x1, y1), batch_size=2),
        ),
        (
            DataLoader(TensorDataset(x2, y2), batch_size=2),
            DataLoader(TensorDataset(x2, y2), batch_size=2),
            DataLoader(TensorDataset(x2, y2), batch_size=2),
        ),
    ]
    res = train_ncg(
        model, tasks, device,
        epochs_per_task=1,
        verbose=False,
        task_pairs=[(0, 1), (2, 3)],
    )
    assert "task_accs" in res
    assert len(res["task_accs"]) == 2


def test_train_static_mlp_runs_one_epoch_per_task_tiny_data():
    device = torch.device("cpu")
    model = StaticMLP(input_size=16, hidden_size=8, num_classes=2)
    x1 = torch.randn(4, 16)
    y1 = torch.randint(0, 2, (4,))
    x2 = torch.randn(4, 16)
    y2 = torch.randint(0, 2, (4,))
    tasks = [
        (
            DataLoader(TensorDataset(x1, y1), batch_size=2),
            DataLoader(TensorDataset(x1, y1), batch_size=2),
            DataLoader(TensorDataset(x1, y1), batch_size=2),
        ),
        (
            DataLoader(TensorDataset(x2, y2), batch_size=2),
            DataLoader(TensorDataset(x2, y2), batch_size=2),
            DataLoader(TensorDataset(x2, y2), batch_size=2),
        ),
    ]
    res = train_static_mlp(
        model, tasks, device,
        epochs_per_task=1,
        verbose=False,
        task_pairs=[(0, 1), (2, 3)],
    )
    assert "task_accs" in res
    assert len(res["task_accs"]) == 2
