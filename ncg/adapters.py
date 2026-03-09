"""
Architecture-agnostic growth adapters for NCG.

Any PyTorch model can be made compatible with NCG's growth
mechanism by providing a GrowthAdapter subclass. The adapter
encapsulates all layer-specific expansion logic, keeping the
growth trigger (in ncg/train.py) completely model-agnostic.

Supported out of the box:
- LinearGrowthAdapter: nn.Linear hidden expansion (MLP)
- Conv2dGrowthAdapter: nn.Conv2d channel expansion (CNN)
- TransformerGrowthAdapter: FFN intermediate dim expansion
- NCGGrowthAdapter: bridge for existing NCGModel/NCGModelCNN

To add support for a new layer type, subclass GrowthAdapter
and implement expand() and current_size().
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn


def _replace_module(model: nn.Module, old_module: nn.Module, new_module: nn.Module) -> None:
    """Replace old_module with new_module in model's hierarchy by finding its attribute path."""
    for name, mod in model.named_modules():
        if mod is old_module:
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_module)
            return
    raise ValueError("old_module not found in model hierarchy")


# -----------------------------------------------------------------------------
# GrowthAdapter (ABC)
# -----------------------------------------------------------------------------


class GrowthAdapter(ABC):
    """
    Encapsulates all layer-specific logic needed to expand a layer
    in-place without disrupting existing weights.

    To support a new layer type, subclass GrowthAdapter and implement:
        - expand(model)
        - current_size(model) -> int
    """

    def __init__(
        self,
        layer_getter: Callable[[nn.Module], Any],
        growth_units: int = 64,
        init_std: Optional[float] = None,
    ) -> None:
        self.layer_getter = layer_getter
        self.growth_units = growth_units
        self._init_std = init_std
        self._growth_count = 0
        self._size_history: List[Dict[str, Any]] = []

    def _compute_init_std(self, weight: torch.Tensor) -> float:
        """Small std relative to existing weight norm; minimum 1e-4."""
        if self._init_std is not None:
            return self._init_std
        norm = weight.data.norm().item()
        numel = weight.numel()
        if numel == 0:
            return 1e-4
        std = (norm / math.sqrt(numel)) * 0.1
        return max(std, 1e-4)

    def log_growth(
        self,
        old_size: int,
        new_size: int,
        epoch: Optional[int] = None,
        task: Optional[int] = None,
    ) -> None:
        self._size_history.append({
            "old_size": old_size,
            "new_size": new_size,
            "epoch": epoch,
            "task": task,
        })
        self._growth_count += 1

    def history(self) -> list:
        return list(self._size_history)

    @abstractmethod
    def expand(self, model: nn.Module) -> None:
        ...

    @abstractmethod
    def current_size(self, model: nn.Module) -> int:
        ...


# -----------------------------------------------------------------------------
# LinearGrowthAdapter
# -----------------------------------------------------------------------------


class LinearGrowthAdapter(GrowthAdapter):
    """
    Expands an nn.Linear hidden layer (output dim) and automatically
    updates the downstream layer's input dim.
    """

    def __init__(
        self,
        layer_getter: Callable[[nn.Module], nn.Linear],
        downstream_getter: Optional[Callable[[nn.Module], nn.Module]] = None,
        growth_units: int = 64,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__(layer_getter=layer_getter, growth_units=growth_units, init_std=init_std)
        self.downstream_getter = downstream_getter

    def expand(self, model: nn.Module) -> None:
        layer = self.layer_getter(model)
        if not isinstance(layer, nn.Linear):
            raise TypeError("layer_getter must return nn.Linear")
        dev = layer.weight.device
        dtype = layer.weight.dtype
        old_out = layer.out_features
        in_feat = layer.in_features
        new_out = old_out + self.growth_units
        std = self._compute_init_std(layer.weight)

        # New rows for expanded layer
        new_weight_rows = torch.empty(self.growth_units, in_feat, device=dev, dtype=dtype)
        nn.init.normal_(new_weight_rows, mean=0.0, std=std)
        new_weight = torch.cat([layer.weight.data, new_weight_rows], dim=0)
        new_bias = torch.zeros(new_out, device=dev, dtype=dtype)
        new_bias[:old_out] = layer.bias.data
        # new_bias[old_out:] already zero

        new_layer = nn.Linear(in_feat, new_out, bias=True).to(dev)
        new_layer.weight.data = new_weight
        new_layer.bias.data = new_bias
        _replace_module(model, layer, new_layer)

        # Downstream: append zero columns (new input features)
        if self.downstream_getter is not None:
            down = self.downstream_getter(model)
            if isinstance(down, nn.Linear):
                down_old_in = down.in_features
                down_new_in = down_old_in + self.growth_units
                new_down_weight = torch.zeros(down.out_features, down_new_in, device=dev, dtype=down.weight.dtype)
                new_down_weight[:, :down_old_in] = down.weight.data
                new_down = nn.Linear(down_new_in, down.out_features, bias=down.bias is not None).to(dev)
                new_down.weight.data = new_down_weight
                if down.bias is not None:
                    new_down.bias.data.copy_(down.bias.data)
                _replace_module(model, down, new_down)

        self.log_growth(old_out, new_out)

    def current_size(self, model: nn.Module) -> int:
        layer = self.layer_getter(model)
        return int(layer.out_features)


# -----------------------------------------------------------------------------
# Conv2dGrowthAdapter
# -----------------------------------------------------------------------------


class Conv2dGrowthAdapter(GrowthAdapter):
    """
    Expands an nn.Conv2d layer by adding output channels.
    Downstream can be Conv2d (in_channels updated) or Linear (in_features updated).
    """

    def __init__(
        self,
        layer_getter: Callable[[nn.Module], nn.Conv2d],
        downstream_getter: Optional[Callable[[nn.Module], nn.Module]] = None,
        growth_units: int = 16,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__(layer_getter=layer_getter, growth_units=growth_units, init_std=init_std)
        self.downstream_getter = downstream_getter

    def expand(self, model: nn.Module) -> None:
        layer = self.layer_getter(model)
        if not isinstance(layer, nn.Conv2d):
            raise TypeError("layer_getter must return nn.Conv2d")
        dev = layer.weight.device
        dtype = layer.weight.dtype
        old_out = layer.out_channels
        in_ch = layer.in_channels
        kH, kW = layer.kernel_size
        if isinstance(kH, tuple):
            kH, kW = kH[0], kW[0]
        new_out = old_out + self.growth_units
        std = self._compute_init_std(layer.weight)

        # New filters: (growth_units, in_channels, kH, kW)
        new_filters = torch.empty(self.growth_units, in_ch, kH, kW, device=dev, dtype=dtype)
        nn.init.normal_(new_filters, mean=0.0, std=std)
        new_weight = torch.cat([layer.weight.data, new_filters], dim=0)
        new_bias = torch.zeros(new_out, device=dev, dtype=dtype)
        new_bias[:old_out] = layer.bias.data if layer.bias is not None else 0

        new_layer = nn.Conv2d(in_ch, new_out, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding).to(dev)
        new_layer.weight.data = new_weight
        new_layer.bias.data = new_bias
        _replace_module(model, layer, new_layer)

        if self.downstream_getter is not None:
            down = self.downstream_getter(model)
            if isinstance(down, nn.Conv2d):
                down_old_in = down.in_channels
                down_new_in = down_old_in + self.growth_units
                # (out_ch, in_ch, kH, kW) -> append zero input channels
                new_down_weight = torch.zeros(down.out_channels, down_new_in, *down.kernel_size, device=dev, dtype=down.weight.dtype)
                new_down_weight[:, :down_old_in] = down.weight.data
                new_down = nn.Conv2d(down_new_in, down.out_channels, kernel_size=down.kernel_size, stride=down.stride, padding=down.padding).to(dev)
                new_down.weight.data = new_down_weight
                new_down.bias.data.copy_(down.bias.data) if down.bias is not None else None
                _replace_module(model, down, new_down)
            elif isinstance(down, nn.Linear):
                down_old_in = down.in_features
                extra = (down_old_in // old_out) * self.growth_units if old_out else self.growth_units
                down_new_in = down_old_in + extra
                new_down_weight = torch.zeros(down.out_features, down_new_in, device=dev, dtype=down.weight.dtype)
                new_down_weight[:, :down_old_in] = down.weight.data
                new_down = nn.Linear(down_new_in, down.out_features, bias=down.bias is not None).to(dev)
                new_down.weight.data = new_down_weight
                if down.bias is not None:
                    new_down.bias.data.copy_(down.bias.data)
                _replace_module(model, down, new_down)

        self.log_growth(old_out, new_out)

    def current_size(self, model: nn.Module) -> int:
        layer = self.layer_getter(model)
        return int(layer.out_channels)


# -----------------------------------------------------------------------------
# TransformerGrowthAdapter
# -----------------------------------------------------------------------------


class TransformerGrowthAdapter(GrowthAdapter):
    """
    Expands the FFN intermediate dimension of a Transformer block
    by reusing LinearGrowthAdapter logic (ffn_up -> linear1, ffn_down -> linear2).
    """

    def __init__(
        self,
        ffn_up_getter: Callable[[nn.Module], nn.Linear],
        ffn_down_getter: Callable[[nn.Module], nn.Linear],
        growth_units: int = 64,
        init_std: Optional[float] = None,
    ) -> None:
        super().__init__(layer_getter=ffn_up_getter, growth_units=growth_units, init_std=init_std)
        self.ffn_up_getter = ffn_up_getter
        self.ffn_down_getter = ffn_down_getter
        self._linear_adapter = LinearGrowthAdapter(
            layer_getter=ffn_up_getter,
            downstream_getter=ffn_down_getter,
            growth_units=growth_units,
            init_std=init_std,
        )

    def expand(self, model: nn.Module) -> None:
        old_size = self.current_size(model)
        self._linear_adapter.expand(model)
        new_size = self.current_size(model)
        self.log_growth(old_size, new_size)

    def current_size(self, model: nn.Module) -> int:
        return self.ffn_up_getter(model).out_features


# -----------------------------------------------------------------------------
# NCGGrowthAdapter
# -----------------------------------------------------------------------------


class NCGGrowthAdapter(GrowthAdapter):
    """
    Bridge adapter that wraps NCGModel.grow() / NCGModelCNN.grow()
    so they conform to the GrowthAdapter interface.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        growth_units: int = 64,
    ) -> None:
        def layer_getter(m: nn.Module) -> Any:
            return m
        super().__init__(layer_getter=layer_getter, growth_units=growth_units)
        self.model_type = model_type

    def expand(self, model: nn.Module) -> None:
        old_size = self.current_size(model)
        model.grow(self.growth_units)
        self.log_growth(old_size, self.current_size(model))

    def current_size(self, model: nn.Module) -> int:
        return int(model.hidden_size)
