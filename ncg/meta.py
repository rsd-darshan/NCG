"""
Standalone meta-parameters for NCG.

Use this when your model is NOT an NCGModel but you still want
NCG's self-regulating meta-parameter behavior.

Example:
    meta = StandaloneMetaParameters()
    optimizer_meta = torch.optim.Adam(meta.get_params(), lr=0.01)
    loss = meta.compute_training_loss(logits, targets, your_model)
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_sigmoid(y: float) -> float:
    """Inverse of sigmoid: x such that sigmoid(x) = y. Clamps y to (1e-7, 1-1e-7)."""
    y = max(1e-7, min(1.0 - 1e-7, y))
    return math.log(y / (1.0 - y))


def _inv_softplus(y: float) -> float:
    """Inverse of softplus: x such that softplus(x) = y. For y<=0 returns small negative."""
    if y <= 0:
        return -10.0
    return math.log(math.exp(y) - 1.0)


class StandaloneMetaParameters(nn.Module):
    """
    Standalone α, β, λ with same semantics as NCGModel: exploration, complexity, regularisation.
    """

    def __init__(
        self,
        alpha_init: float = 0.5,
        beta_init: float = 0.01,
        lambda_init: float = 0.5,
        lr: float = 0.01,
    ) -> None:
        super().__init__()
        self._lr = lr
        self.alpha_raw = nn.Parameter(torch.tensor(_inv_sigmoid(alpha_init)))
        # beta = softplus(beta_raw) * 0.1 => at init we want beta_init, so softplus(beta_raw) = beta_init/0.1
        self.beta_raw = nn.Parameter(torch.tensor(_inv_softplus(beta_init / 0.1)))
        self.lambda_raw = nn.Parameter(torch.tensor(_inv_sigmoid(lambda_init)))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.beta_raw) * 0.1

    @property
    def lambda_(self) -> torch.Tensor:
        return torch.sigmoid(self.lambda_raw)

    def _arch_norm(self, model: nn.Module) -> torch.Tensor:
        """Sum of squared weights of all Linear layers in model."""
        total = torch.tensor(0.0, device=next(model.parameters()).device)
        for m in model.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                total = total + (m.weight ** 2).sum()
        return total

    def compute_training_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
        num_classes: int = 2,
    ) -> torch.Tensor:
        """
        L = CE - alpha*entropy + beta*arch_norm + lambda*weight_norm
        (alpha, beta, lambda detached so only weights get gradients.)
        """
        ce = F.cross_entropy(logits, targets)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()
        alpha = self.alpha.detach()
        beta = self.beta.detach()
        lambda_ = self.lambda_.detach()
        arch_norm = self._arch_norm(model)
        weight_norm = arch_norm
        return ce - alpha * entropy + beta * arch_norm + lambda_ * weight_norm

    def compute_meta_loss(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        num_classes: int = 2,
        model: Optional[nn.Module] = None,
        tau_nov: float = 0.3,
        tau_arch: float = 0.01,
        tau_reg: float = 0.1,
    ) -> torch.Tensor:
        """
        Lagrangian-style meta-loss. Use (-L_meta).backward() for gradient ascent.
        Novelty computed from val_logits (normalized predictive entropy).
        """
        ce_val = F.cross_entropy(val_logits, val_targets)
        probs = F.softmax(val_logits, dim=1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()
        novelty = entropy / math.log(max(num_classes, 2))
        if model is not None:
            arch_norm_raw = self._arch_norm(model)
            n_params = sum(p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in (m.weight,))
            n_params = max(n_params, 1)
            arch_norm = arch_norm_raw / n_params
            weight_norm = (arch_norm_raw / n_params + 1e-8).sqrt()
        else:
            dev = val_logits.device
            arch_norm = torch.tensor(0.01, device=dev)
            weight_norm = torch.tensor(0.1, device=dev)
        alpha = self.alpha
        beta = self.beta
        lambda_ = self.lambda_
        L_meta = (
            ce_val
            - alpha * (novelty - tau_nov)
            - beta * (tau_arch - arch_norm)
            - lambda_ * (tau_reg - weight_norm)
        )
        return L_meta

    def get_params(self) -> List[torch.Tensor]:
        return [self.alpha_raw, self.beta_raw, self.lambda_raw]

    def snapshot(self) -> dict:
        return {
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "lambda": self.lambda_.item(),
        }
