"""Tests for universal NCG: StandaloneMetaParameters, NoveltyMonitor, train_ncg with custom models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ncg import train_ncg, get_device, LinearGrowthAdapter, StandaloneMetaParameters
from ncg.model import NCGModel
from ncg.novelty import NoveltyMonitor


def _tiny_custom_model(input_size=16, hidden=8, num_classes=2):
    """Custom model that returns (logits, h)."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden)
            self.fc2 = nn.Linear(hidden, num_classes)

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            return self.fc2(h), h

    return M()


def test_standalone_meta_alpha_in_01_at_init():
    meta = StandaloneMetaParameters(alpha_init=0.5)
    a = meta.alpha.item()
    assert 0 < a < 1


def test_standalone_meta_beta_positive_at_init():
    meta = StandaloneMetaParameters(beta_init=0.01)
    b = meta.beta.item()
    assert b > 0


def test_standalone_meta_lambda_in_01_at_init():
    meta = StandaloneMetaParameters(lambda_init=0.5)
    lam = meta.lambda_.item()
    assert 0 < lam < 1


def test_standalone_meta_compute_training_loss_returns_scalar_with_grad():
    meta = StandaloneMetaParameters()
    model = nn.Linear(8, 2)
    logits = torch.randn(4, 2)
    targets = torch.randint(0, 2, (4,))
    loss = meta.compute_training_loss(logits, targets, model)
    assert loss.dim() == 0
    assert loss.requires_grad
    loss.backward()


def test_standalone_meta_get_params_returns_three():
    meta = StandaloneMetaParameters()
    params = meta.get_params()
    assert len(params) == 3


def test_novelty_monitor_compute_returns_float_in_01():
    model = nn.Linear(8, 4)
    monitor = NoveltyMonitor(model, lambda m: m)
    x = torch.randn(32, 8)
    loader = DataLoader(TensorDataset(x, torch.zeros(32, dtype=torch.long)), batch_size=8)
    score = monitor.compute(loader, torch.device("cpu"), max_batches=2)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_novelty_monitor_history_appends_after_compute():
    model = nn.Linear(8, 4)
    monitor = NoveltyMonitor(model, lambda m: m)
    x = torch.randn(16, 8)
    loader = DataLoader(TensorDataset(x, torch.zeros(16, dtype=torch.long)), batch_size=4)
    assert len(monitor.history()) == 0
    monitor.compute(loader, torch.device("cpu"), max_batches=2)
    assert len(monitor.history()) == 1
    monitor.compute(loader, torch.device("cpu"), max_batches=2)
    assert len(monitor.history()) == 2


def test_train_ncg_ncg_model_still_works():
    device = torch.device("cpu")
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=32)
    x1 = torch.randn(32, 16)
    y1 = torch.randint(0, 2, (32,))
    x2 = torch.randn(32, 16)
    y2 = torch.randint(0, 2, (32,))
    tasks = [
        (
            DataLoader(TensorDataset(x1, y1), batch_size=8),
            DataLoader(TensorDataset(x1, y1), batch_size=8),
            DataLoader(TensorDataset(x1, y1), batch_size=8),
        ),
        (
            DataLoader(TensorDataset(x2, y2), batch_size=8),
            DataLoader(TensorDataset(x2, y2), batch_size=8),
            DataLoader(TensorDataset(x2, y2), batch_size=8),
        ),
    ]
    res = train_ncg(model, tasks, device, epochs_per_task=1, verbose=False, task_pairs=[(0, 1), (2, 3)])
    assert "task_accs" in res
    assert len(res["task_accs"]) == 2


def test_train_ncg_custom_model_adapter_meta_runs():
    device = torch.device("cpu")
    model = _tiny_custom_model(16, 8, 2)
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=4)
    meta = StandaloneMetaParameters()
    x1 = torch.randn(32, 16)
    y1 = torch.randint(0, 2, (32,))
    x2 = torch.randn(32, 16)
    y2 = torch.randint(0, 2, (32,))
    tasks = [
        (
            DataLoader(TensorDataset(x1, y1), batch_size=8),
            DataLoader(TensorDataset(x1, y1), batch_size=8),
            DataLoader(TensorDataset(x1, y1), batch_size=8),
        ),
        (
            DataLoader(TensorDataset(x2, y2), batch_size=8),
            DataLoader(TensorDataset(x2, y2), batch_size=8),
            DataLoader(TensorDataset(x2, y2), batch_size=8),
        ),
    ]
    res = train_ncg(
        model, tasks, device,
        epochs_per_task=1,
        verbose=False,
        adapter=adapter,
        meta=meta,
        novelty_layer_getter=lambda m: m.fc1,
        task_pairs=[(0, 1), (2, 3)],
    )
    assert "task_accs" in res
    assert len(res["task_accs"]) == 2


def test_train_ncg_custom_model_adapter_only_runs():
    device = torch.device("cpu")
    model = _tiny_custom_model(16, 8, 2)
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=4)
    x1 = torch.randn(32, 16)
    y1 = torch.randint(0, 2, (32,))
    tasks = [
        (
            DataLoader(TensorDataset(x1, y1), batch_size=8),
            DataLoader(TensorDataset(x1, y1), batch_size=8),
            DataLoader(TensorDataset(x1, y1), batch_size=8),
        ),
    ]
    res = train_ncg(
        model, tasks, device,
        epochs_per_task=1,
        verbose=False,
        adapter=adapter,
        task_pairs=[(0, 1)],
    )
    assert "task_accs" in res
    assert len(res["task_accs"]) == 1
