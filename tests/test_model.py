"""Pytest tests for ncg.model."""

import pytest
import torch

from ncg.model import EWC, NCGModel, NCGModelCNN, StaticMLP


def test_ncg_forward_returns_logits_and_h():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2)
    x = torch.randn(4, 16)
    logits, h = model(x)
    assert logits.shape == (4, 2)
    assert h.shape == (4, 8)


def test_ncg_grow_increases_hidden_size_by_64():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=256)
    assert model.hidden_size == 8
    model.grow(64)
    assert model.hidden_size == 8 + 64


def test_ncg_forward_after_grow():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=256)
    model.grow(64)
    x = torch.randn(4, 16)
    logits, h = model(x)
    assert logits.shape == (4, 2)
    assert h.shape == (4, 8 + 64)


def test_ncg_check_growth_trigger_returns_false_when_novelty_high():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=256)
    # novelty >= 0.5 -> False
    assert model.check_growth_trigger([0.9, 0.9, 0.9], novelty=0.6, verbose=False) is False


def test_ncg_check_growth_trigger_returns_false_when_lambda_low():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2, max_hidden=256)
    model.lambda_raw.data.fill_(-2.0)  # sigmoid -> small lambda
    # lambda <= 0.3 -> False (need recent_val_accs with plateau and novelty < 0.5)
    assert model.check_growth_trigger([0.5, 0.5, 0.5], novelty=0.2, verbose=False) is False


def test_ncg_meta_params_valid_at_init():
    model = NCGModel(input_size=16, hidden_size=8, num_classes=2)
    a, b, lam = model.alpha.item(), model.beta.item(), model.lambda_.item()
    assert 0 <= a <= 1
    assert 0 <= b <= 0.1  # beta is sigmoid * 0.1
    assert 0 <= lam <= 1


def test_ncg_fixed_meta_constant():
    model = NCGModel(
        input_size=16, hidden_size=8, num_classes=2,
        fixed_meta=(0.5, 0.01, 0.5),
    )
    assert model.alpha.item() == pytest.approx(0.5)
    assert model.beta.item() == pytest.approx(0.01)
    assert model.lambda_.item() == pytest.approx(0.5)


def test_static_mlp_forward_shape():
    model = StaticMLP(input_size=16, hidden_size=8, num_classes=2)
    x = torch.randn(4, 16)
    out = model(x)
    assert out.shape == (4, 2)


def test_ewc_forward_shape():
    model = EWC(input_size=16, hidden_size=8, num_classes=2, backbone="mlp")
    x = torch.randn(4, 16)
    out = model(x)
    assert out.shape == (4, 2)


def test_ncg_cnn_forward_shape():
    model = NCGModelCNN(hidden_size=8, num_classes=2, max_hidden=256)
    x = torch.randn(4, 3, 32, 32)
    logits, h = model(x)
    assert logits.shape == (4, 2)
    assert h.shape == (4, 8)
