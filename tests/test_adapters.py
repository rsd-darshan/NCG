"""Pytest tests for ncg.adapters."""

import torch
import torch.nn as nn

from ncg.adapters import (
    Conv2dGrowthAdapter,
    GrowthAdapter,
    LinearGrowthAdapter,
    NCGGrowthAdapter,
    TransformerGrowthAdapter,
)
from ncg.model import NCGModel, NCGModelCNN


# ---- Helpers for tests ----
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class TinyTransformerBlock(nn.Module):
    def __init__(self, d=32, ffn=64):
        super().__init__()
        self.linear1 = nn.Linear(d, ffn)
        self.linear2 = nn.Linear(ffn, d)

    def forward(self, x):
        return x + self.linear2(torch.relu(self.linear1(x)))


# ---- LinearGrowthAdapter ----
def test_linear_growth_adapter_expand_increases_out_features():
    model = TinyMLP()
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=8)
    assert adapter.current_size(model) == 16
    adapter.expand(model)
    assert adapter.current_size(model) == 16 + 8


def test_linear_growth_adapter_expand_updates_downstream_in_features():
    model = TinyMLP()
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=8)
    assert model.fc2.in_features == 16
    adapter.expand(model)
    assert model.fc2.in_features == 16 + 8


def test_linear_growth_adapter_forward_after_expand():
    model = TinyMLP()
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=8)
    adapter.expand(model)
    x = torch.randn(3, 8)
    y = model(x)
    assert y.shape == (3, 2)


def test_linear_growth_adapter_multiple_expands():
    model = TinyMLP()
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=4)
    adapter.expand(model)
    assert adapter.current_size(model) == 20
    adapter.expand(model)
    assert adapter.current_size(model) == 24
    assert model.fc2.in_features == 24


# ---- Conv2dGrowthAdapter ----
def test_conv2d_growth_adapter_expand_increases_out_channels():
    model = TinyCNN()
    adapter = Conv2dGrowthAdapter(lambda m: m.conv2, lambda m: m.fc, growth_units=4)
    assert adapter.current_size(model) == 8
    adapter.expand(model)
    assert adapter.current_size(model) == 8 + 4


def test_conv2d_growth_adapter_expand_updates_downstream_conv_in_channels():
    model = nn.Sequential(
        nn.Conv2d(2, 4, 3),
        nn.Conv2d(4, 8, 3),
    )
    adapter = Conv2dGrowthAdapter(
        layer_getter=lambda m: m[0],
        downstream_getter=lambda m: m[1],
        growth_units=2,
    )
    assert model[1].in_channels == 4
    adapter.expand(model)
    assert model[1].in_channels == 4 + 2


def test_conv2d_growth_adapter_forward_after_expand():
    model = TinyCNN()
    adapter = Conv2dGrowthAdapter(lambda m: m.conv2, lambda m: m.fc, growth_units=4)
    adapter.expand(model)
    x = torch.randn(2, 2, 10, 10)
    y = model(x)
    assert y.shape == (2, 2)


# ---- NCGGrowthAdapter ----
def test_ncg_growth_adapter_current_size_matches_model_hidden_size():
    model = NCGModel(input_size=16, hidden_size=32, num_classes=2, max_hidden=128)
    adapter = NCGGrowthAdapter(model_type="mlp", growth_units=64)
    assert adapter.current_size(model) == model.hidden_size
    model.grow(64)
    assert adapter.current_size(model) == model.hidden_size == 96


def test_ncg_growth_adapter_expand_delegates_to_model_grow():
    model = NCGModel(input_size=16, hidden_size=32, num_classes=2, max_hidden=128)
    adapter = NCGGrowthAdapter(model_type="mlp", growth_units=64)
    assert model.hidden_size == 32
    adapter.expand(model)
    assert model.hidden_size == 32 + 64


# ---- GrowthAdapter base ----
def test_growth_adapter_growth_count_increments_on_expand():
    model = TinyMLP()
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=4)
    assert adapter._growth_count == 0
    adapter.expand(model)
    assert adapter._growth_count == 1
    adapter.expand(model)
    assert adapter._growth_count == 2


def test_growth_adapter_history_returns_correct_entries():
    model = TinyMLP()
    adapter = LinearGrowthAdapter(lambda m: m.fc1, lambda m: m.fc2, growth_units=4)
    adapter.expand(model)
    adapter.expand(model)
    hist = adapter.history()
    assert len(hist) == 2
    assert hist[0]["old_size"] == 16 and hist[0]["new_size"] == 20
    assert hist[1]["old_size"] == 20 and hist[1]["new_size"] == 24
