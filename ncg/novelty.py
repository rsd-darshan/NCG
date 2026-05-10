"""
Standalone novelty monitor via forward hooks.
Works with any model; no need for compute_novelty() on the model.
"""

from __future__ import annotations

import math
from typing import Callable, List

import torch
from torch.utils.data import DataLoader


class NoveltyMonitor:
    """
    Captures activations from a layer via forward hook and computes
    novelty as normalised entropy of mean absolute activation distribution.
    """

    def __init__(self, model: torch.nn.Module, layer_getter: Callable[[torch.nn.Module], torch.nn.Module]) -> None:
        self._model = model
        self._layer_getter = layer_getter
        self._activations: List[torch.Tensor] = []
        self._handle = None
        self._history: List[float] = []
        self._last: float = 0.5

        layer = layer_getter(model)
        if layer is None:
            raise ValueError("layer_getter returned None")

        def _hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() > 2:
                out = out.flatten(1)
            self._activations.append(out.detach())

        self._handle = layer.register_forward_hook(_hook)

    def compute(
        self,
        val_loader: DataLoader,
        device: torch.device,
        max_batches: int = 10,
    ) -> float:
        """Run model in eval, collect activations via hook, return novelty in [0, 1]."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        layer = self._layer_getter(self._model)

        def _hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.dim() > 2:
                out = out.flatten(1)
            self._activations.append(out.detach())

        self._handle = layer.register_forward_hook(_hook)
        self._activations = []
        self._model.eval()
        batch_count = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                _ = self._model(x)
                batch_count += 1
                if batch_count >= max_batches:
                    break

        if self._handle is not None:
            self._handle.remove()
            self._handle = None

        if not self._activations:
            self._last = 0.5
            self._history.append(self._last)
            return self._last

        activations = torch.cat(self._activations, dim=0)
        mean_abs = activations.abs().mean(dim=0)
        total = mean_abs.sum()
        if total < 1e-10:
            self._last = 0.5
            self._history.append(self._last)
            return self._last
        p = mean_abs / total
        d = p.numel()
        eps = 1e-10
        H = -(p * (p + eps).log()).sum().item()
        novelty = H / math.log(max(d, 2))
        novelty = max(0.0, min(1.0, novelty))
        self._last = float(novelty)
        self._history.append(self._last)
        return self._last

    def last(self) -> float:
        return self._last

    def history(self) -> List[float]:
        return list(self._history)
