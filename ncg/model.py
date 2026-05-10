"""
NCG Research Prototype — Model definitions.

Contains:
- NCGModel: Novelty-triggered Capacity Growth with growable hidden layer,
  meta-parameters (α, β, λ), knowledge gating, and growth trigger.
- StaticMLP: Fixed-width 2-layer MLP baseline.
- EWC: Elastic Weight Consolidation baseline (wraps StaticMLP).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# NCGModel
# -----------------------------------------------------------------------------


class NCGModel(nn.Module):
    """
    Novelty-triggered Capacity Growth: self-evolving MLP with
    - Learnable weights W(t)
    - Dynamic hidden size (growable by +64 units)
    - Knowledge embedding K(t) with gated write
    - Meta-parameters ω(t) = {α, β, λ} for exploration, complexity, regularization
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        num_classes: int = 2,
        max_hidden: int = 512,
        fixed_meta: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_hidden = max_hidden
        self._fixed_meta = fixed_meta

        # Main layers (growable)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Knowledge embedding: fixed-size buffer, gated write
        self.register_buffer("K", torch.zeros(hidden_size))
        self.gate_layer = nn.Linear(hidden_size, hidden_size)

        # Meta-parameters: either fixed (buffers) or learnable (Parameters)
        if fixed_meta is not None:
            alpha_val, beta_val, lambda_val = fixed_meta
            self.register_buffer("alpha_val", torch.tensor(alpha_val, dtype=torch.float32))
            self.register_buffer("beta_val", torch.tensor(beta_val, dtype=torch.float32))
            self.register_buffer("lambda_val", torch.tensor(lambda_val, dtype=torch.float32))
        else:
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))
            self.beta_raw = nn.Parameter(torch.tensor(0.0))
            self.lambda_raw = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.gate_layer.weight)
        nn.init.zeros_(self.gate_layer.bias)

    @property
    def alpha(self) -> torch.Tensor:
        if self._fixed_meta is not None:
            return self.alpha_val
        return torch.sigmoid(self.alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        if self._fixed_meta is not None:
            return self.beta_val
        return torch.sigmoid(self.beta_raw) * 0.1

    @property
    def lambda_(self) -> torch.Tensor:
        if self._fixed_meta is not None:
            return self.lambda_val
        return torch.sigmoid(self.lambda_raw)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns (logits, hidden_activations)."""
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        h = h + self.K.unsqueeze(0)
        logits = self.fc2(h)
        return logits, h

    def update_knowledge(self, h: torch.Tensor) -> None:
        """
        Gated write into knowledge buffer K.
        gate = sigmoid(gate_layer(h_mean)); K = (1-gate)*K + gate*h_mean
        """
        with torch.no_grad():
            h_mean = h.mean(0)
            gate = torch.sigmoid(self.gate_layer(h_mean))
            self.K.data = (1 - gate) * self.K + gate * h_mean

    def compute_novelty(self, logits: torch.Tensor, C: int) -> torch.Tensor:
        """Normalized predictive entropy: Nov = H(p(y|x)) / log(C)."""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()
        return entropy / math.log(max(C, 2))

    def compute_training_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = 2,
    ) -> torch.Tensor:
        """
        L_train = CE - α*entropy + β*arch_norm + λ*||W||^2
        (α, β, λ detached so only weights get gradients from this loss.)
        """
        ce = F.cross_entropy(logits, targets)
        entropy = -(F.softmax(logits, dim=1) * (F.log_softmax(logits, dim=1) + 1e-10)).sum(dim=1).mean()
        alpha = self.alpha.detach()
        beta = self.beta.detach()
        lambda_ = self.lambda_.detach()

        # Architecture norm: L2 of hidden-layer weights (simplified width proxy)
        arch_norm = (self.fc1.weight ** 2).sum() + (self.fc2.weight ** 2).sum()
        weight_norm = arch_norm  # same for this 2-layer net

        return ce - alpha * entropy + beta * arch_norm + lambda_ * weight_norm

    def compute_meta_loss(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        num_classes: int = 2,
        tau_nov: float = 0.3,
        tau_arch: float = 0.01,
        tau_reg: float = 0.1,
    ) -> torch.Tensor:
        """
        Lagrangian-style meta-loss. We MAXIMIZE L_meta w.r.t. dual vars (alpha, beta, lambda)
        so use (-L_meta).backward() in the training loop for gradient ascent.
        L_meta = CE_val - alpha*(novelty - tau_nov) - beta*(tau_arch - arch_norm_norm) - lambda_*(tau_reg - weight_norm_norm)
        Normalized arch/weight norms so tau_arch and tau_reg are on a 0.01--0.1 scale.
        """
        ce_val = F.cross_entropy(val_logits, val_targets)
        novelty = self.compute_novelty(val_logits, num_classes)
        arch_norm_raw = (self.fc1.weight ** 2).sum() + (self.fc2.weight ** 2).sum()
        n_params = self.fc1.weight.numel() + self.fc2.weight.numel()
        arch_norm = arch_norm_raw / max(n_params, 1)
        weight_norm = (arch_norm_raw / max(n_params, 1) + 1e-8).sqrt()

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

    def check_growth_trigger(
        self,
        recent_val_accs: List[float],
        novelty: float,
        window: int = 3,
        plateau_threshold: float = 0.005,
        verbose: bool = True,
    ) -> bool:
        """
        Returns True when: (1) accuracy plateau over window, (2) novelty < 0.5,
        (3) lambda > 0.3, (4) hidden_size < max_hidden.
        """
        if self.hidden_size >= self.max_hidden:
            return False
        if novelty >= 0.5:
            return False
        lam = self.lambda_.item()
        if lam <= 0.3:
            return False
        if len(recent_val_accs) < window:
            return False
        smoothed = [
            sum(recent_val_accs[max(0, i - 2) : i + 1]) / min(3, i + 1)
            for i in range(len(recent_val_accs))
        ]
        recent = smoothed[-window:]
        spread = max(recent) - min(recent)
        if spread > plateau_threshold:
            return False
        return True

    def grow(self, num_new: int = 64) -> None:
        """
        Expand hidden layer by num_new units. Preserves existing weights;
        new fc1 weights use Kaiming init; new fc2 weights zero-initialized.
        """
        dev = next(self.parameters()).device
        old_h = self.hidden_size
        new_h = old_h + num_new
        if new_h > self.max_hidden:
            num_new = self.max_hidden - old_h
            new_h = self.max_hidden
        if num_new <= 0:
            return

        # Grow fc1: out_features  old_h -> new_h
        new_fc1 = nn.Linear(self.input_size, new_h, bias=True).to(dev)
        with torch.no_grad():
            new_fc1.weight[:old_h] = self.fc1.weight
            new_fc1.bias[:old_h] = self.fc1.bias
            nn.init.kaiming_normal_(new_fc1.weight[old_h:], mode="fan_in", nonlinearity="relu")
            new_fc1.bias[old_h:].zero_()
        self.fc1 = new_fc1

        # Grow fc2: in_features  old_h -> new_h; new columns zero so output unchanged
        new_fc2 = nn.Linear(new_h, self.num_classes, bias=True).to(dev)
        with torch.no_grad():
            new_fc2.weight[:, :old_h] = self.fc2.weight
            new_fc2.weight[:, old_h:].zero_()
            new_fc2.bias.copy_(self.fc2.bias)
        self.fc2 = new_fc2

        # Grow knowledge buffer
        new_K = torch.zeros(new_h, device=dev, dtype=self.K.dtype)
        new_K[:old_h] = self.K
        self.register_buffer("K", new_K)

        # Grow gate layer
        new_gate = nn.Linear(new_h, new_h, bias=True).to(dev)
        with torch.no_grad():
            new_gate.weight[:old_h, :old_h] = self.gate_layer.weight
            new_gate.weight[:old_h, old_h:].zero_()
            new_gate.weight[old_h:, :].zero_()
            new_gate.bias[:old_h] = self.gate_layer.bias
            new_gate.bias[old_h:].zero_()
        self.gate_layer = new_gate

        self.hidden_size = new_h

    def get_weight_params(self) -> List[torch.Tensor]:
        """Parameters to optimize with main optimizer (exclude meta-params)."""
        return [p for n, p in self.named_parameters() if "alpha_raw" not in n and "beta_raw" not in n and "lambda_raw" not in n]

    def get_meta_params(self) -> List[torch.Tensor]:
        """Meta-parameters α, β, λ for separate optimizer. Empty when fixed_meta is set."""
        if self._fixed_meta is not None:
            return []
        return [self.alpha_raw, self.beta_raw, self.lambda_raw]


# -----------------------------------------------------------------------------
# StaticMLP
# -----------------------------------------------------------------------------


class StaticMLP(nn.Module):
    """Fixed-width 2-layer MLP: Linear(784, hidden) -> ReLU -> Linear(hidden, num_classes)."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        return self.fc2(h)


# -----------------------------------------------------------------------------
# SimpleCNN (for Split-CIFAR-10)
# -----------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """
    Two conv layers (32 and 64 filters, 3x3, ReLU, maxpool), one FC hidden 256, output num_classes.
    Input 3x32x32. Used as baseline for Split-CIFAR-10.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        # 32x32 -> 30 -> 15; 15 -> 13 -> 6 => 64*6*6 = 2304
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 6 * 6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32)
        h = self.pool(F.relu(self.conv1(x)))   # (B, 32, 15, 15)
        h = self.pool(F.relu(self.conv2(h)))   # (B, 64, 6, 6)
        h = h.view(h.size(0), -1)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


# -----------------------------------------------------------------------------
# NCGModelCNN (NCG with CNN feature extractor for Split-CIFAR-10)
# -----------------------------------------------------------------------------


class NCGModelCNN(nn.Module):
    """
    Same as NCGModel but: input 3x32x32 goes through CNN feature extractor (two conv layers);
    the FC hidden layer is growable (+64), same meta-parameters (α, β, λ) and knowledge gating.
    """

    CNN_FEATURE_SIZE = 64 * 6 * 6  # 2304

    def __init__(
        self,
        hidden_size: int = 256,
        num_classes: int = 2,
        max_hidden: int = 512,
        fixed_meta: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_hidden = max_hidden
        self._fixed_meta = fixed_meta

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(self.CNN_FEATURE_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.register_buffer("K", torch.zeros(hidden_size))
        self.gate_layer = nn.Linear(hidden_size, hidden_size)

        if fixed_meta is not None:
            alpha_val, beta_val, lambda_val = fixed_meta
            self.register_buffer("alpha_val", torch.tensor(alpha_val, dtype=torch.float32))
            self.register_buffer("beta_val", torch.tensor(beta_val, dtype=torch.float32))
            self.register_buffer("lambda_val", torch.tensor(lambda_val, dtype=torch.float32))
        else:
            self.alpha_raw = nn.Parameter(torch.tensor(0.0))
            self.beta_raw = nn.Parameter(torch.tensor(0.0))
            self.lambda_raw = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.gate_layer.weight)
        nn.init.zeros_(self.gate_layer.bias)

    @property
    def alpha(self) -> torch.Tensor:
        if self._fixed_meta is not None:
            return self.alpha_val
        return torch.sigmoid(self.alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        if self._fixed_meta is not None:
            return self.beta_val
        return torch.sigmoid(self.beta_raw) * 0.1

    @property
    def lambda_(self) -> torch.Tensor:
        if self._fixed_meta is not None:
            return self.lambda_val
        return torch.sigmoid(self.lambda_raw)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        return h.view(h.size(0), -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self._features(x)
        h = F.relu(self.fc1(feat))
        h = h + self.K.unsqueeze(0)
        logits = self.fc2(h)
        return logits, h

    def update_knowledge(self, h: torch.Tensor) -> None:
        with torch.no_grad():
            h_mean = h.mean(0)
            gate = torch.sigmoid(self.gate_layer(h_mean))
            self.K.data = (1 - gate) * self.K + gate * h_mean

    def compute_novelty(self, logits: torch.Tensor, C: int) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=1).mean()
        return entropy / math.log(max(C, 2))

    def compute_training_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_classes: int = 2,
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        entropy = -(F.softmax(logits, dim=1) * (F.log_softmax(logits, dim=1) + 1e-10)).sum(dim=1).mean()
        alpha = self.alpha.detach()
        beta = self.beta.detach()
        lambda_ = self.lambda_.detach()
        arch_norm = (self.fc1.weight ** 2).sum() + (self.fc2.weight ** 2).sum()
        weight_norm = arch_norm
        return ce - alpha * entropy + beta * arch_norm + lambda_ * weight_norm

    def compute_meta_loss(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        num_classes: int = 2,
        tau_nov: float = 0.3,
        tau_arch: float = 0.01,
        tau_reg: float = 0.1,
    ) -> torch.Tensor:
        ce_val = F.cross_entropy(val_logits, val_targets)
        novelty = self.compute_novelty(val_logits, num_classes)
        arch_norm_raw = (self.fc1.weight ** 2).sum() + (self.fc2.weight ** 2).sum()
        n_params = self.fc1.weight.numel() + self.fc2.weight.numel()
        arch_norm = arch_norm_raw / max(n_params, 1)
        weight_norm = (arch_norm_raw / max(n_params, 1) + 1e-8).sqrt()
        alpha, beta, lambda_ = self.alpha, self.beta, self.lambda_
        L_meta = (
            ce_val
            - alpha * (novelty - tau_nov)
            - beta * (tau_arch - arch_norm)
            - lambda_ * (tau_reg - weight_norm)
        )
        return L_meta

    def check_growth_trigger(
        self,
        recent_val_accs: List[float],
        novelty: float,
        window: int = 3,
        plateau_threshold: float = 0.005,
        verbose: bool = True,
    ) -> bool:
        if self.hidden_size >= self.max_hidden:
            return False
        if novelty >= 0.5:
            return False
        lam = self.lambda_.item()
        if lam <= 0.3:
            return False
        if len(recent_val_accs) < window:
            return False
        smoothed = [
            sum(recent_val_accs[max(0, i - 2) : i + 1]) / min(3, i + 1)
            for i in range(len(recent_val_accs))
        ]
        recent = smoothed[-window:]
        spread = max(recent) - min(recent)
        if spread > plateau_threshold:
            return False
        return True

    def grow(self, num_new: int = 64) -> None:
        dev = next(self.parameters()).device
        old_h = self.hidden_size
        new_h = old_h + num_new
        if new_h > self.max_hidden:
            num_new = self.max_hidden - old_h
            new_h = self.max_hidden
        if num_new <= 0:
            return

        new_fc1 = nn.Linear(self.CNN_FEATURE_SIZE, new_h, bias=True).to(dev)
        with torch.no_grad():
            new_fc1.weight[:old_h] = self.fc1.weight
            new_fc1.bias[:old_h] = self.fc1.bias
            nn.init.kaiming_normal_(new_fc1.weight[old_h:], mode="fan_in", nonlinearity="relu")
            new_fc1.bias[old_h:].zero_()
        self.fc1 = new_fc1

        new_fc2 = nn.Linear(new_h, self.num_classes, bias=True).to(dev)
        with torch.no_grad():
            new_fc2.weight[:, :old_h] = self.fc2.weight
            new_fc2.weight[:, old_h:].zero_()
            new_fc2.bias.copy_(self.fc2.bias)
        self.fc2 = new_fc2

        new_K = torch.zeros(new_h, device=dev, dtype=self.K.dtype)
        new_K[:old_h] = self.K
        self.register_buffer("K", new_K)

        new_gate = nn.Linear(new_h, new_h, bias=True).to(dev)
        with torch.no_grad():
            new_gate.weight[:old_h, :old_h] = self.gate_layer.weight
            new_gate.weight[:old_h, old_h:].zero_()
            new_gate.weight[old_h:, :].zero_()
            new_gate.bias[:old_h] = self.gate_layer.bias
            new_gate.bias[old_h:].zero_()
        self.gate_layer = new_gate

        self.hidden_size = new_h

    def get_weight_params(self) -> List[torch.Tensor]:
        return [p for n, p in self.named_parameters() if "alpha_raw" not in n and "beta_raw" not in n and "lambda_raw" not in n]

    def get_meta_params(self) -> List[torch.Tensor]:
        if self._fixed_meta is not None:
            return []
        return [self.alpha_raw, self.beta_raw, self.lambda_raw]


# -----------------------------------------------------------------------------
# DEN (Dynamically Expandable Networks)
# -----------------------------------------------------------------------------


class DENModelCNN(nn.Module):
    """
    DEN with CNN feature extractor for Split-CIFAR-10: same architecture as SimpleCNN
    but FC hidden layer is growable via grow(64), like DENModel.
    """

    CNN_FEATURE_SIZE = 64 * 6 * 6  # 2304

    def __init__(
        self,
        hidden_size: int = 256,
        num_classes: int = 2,
        max_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_hidden = max_hidden
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.CNN_FEATURE_SIZE, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        return h.view(h.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self._features(x)))
        return self.fc2(h)

    def grow(self, num_new: int = 64) -> None:
        dev = next(self.parameters()).device
        old_h = self.hidden_size
        new_h = old_h + num_new
        if new_h > self.max_hidden:
            num_new = self.max_hidden - old_h
            new_h = self.max_hidden
        if num_new <= 0:
            return

        new_fc1 = nn.Linear(self.CNN_FEATURE_SIZE, new_h, bias=True).to(dev)
        with torch.no_grad():
            new_fc1.weight[:old_h] = self.fc1.weight
            new_fc1.bias[:old_h] = self.fc1.bias
            nn.init.kaiming_normal_(new_fc1.weight[old_h:], mode="fan_in", nonlinearity="relu")
            new_fc1.bias[old_h:].zero_()
        self.fc1 = new_fc1

        new_fc2 = nn.Linear(new_h, self.num_classes, bias=True).to(dev)
        with torch.no_grad():
            new_fc2.weight[:, :old_h] = self.fc2.weight
            new_fc2.weight[:, old_h:].zero_()
            new_fc2.bias.copy_(self.fc2.bias)
        self.fc2 = new_fc2

        self.hidden_size = new_h


class DENModel(nn.Module):
    """
    Dynamically Expandable Networks: same architecture as StaticMLP but with
    a grow(new_units=64) method (fc1/fc2 expansion identical to NCGModel.grow).
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        num_classes: int = 2,
        max_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.max_hidden = max_hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def grow(self, num_new: int = 64) -> None:
        """
        Expand hidden layer by num_new units. Preserves existing weights;
        new fc1 weights use Kaiming init; new fc2 weights zero-initialized.
        Identical to NCGModel.grow() for fc1/fc2 only.
        """
        dev = next(self.parameters()).device
        old_h = self.hidden_size
        new_h = old_h + num_new
        if new_h > self.max_hidden:
            num_new = self.max_hidden - old_h
            new_h = self.max_hidden
        if num_new <= 0:
            return

        new_fc1 = nn.Linear(self.input_size, new_h, bias=True).to(dev)
        with torch.no_grad():
            new_fc1.weight[:old_h] = self.fc1.weight
            new_fc1.bias[:old_h] = self.fc1.bias
            nn.init.kaiming_normal_(new_fc1.weight[old_h:], mode="fan_in", nonlinearity="relu")
            new_fc1.bias[old_h:].zero_()
        self.fc1 = new_fc1

        new_fc2 = nn.Linear(new_h, self.num_classes, bias=True).to(dev)
        with torch.no_grad():
            new_fc2.weight[:, :old_h] = self.fc2.weight
            new_fc2.weight[:, old_h:].zero_()
            new_fc2.bias.copy_(self.fc2.bias)
        self.fc2 = new_fc2

        self.hidden_size = new_h


# -----------------------------------------------------------------------------
# EWC (Elastic Weight Consolidation)
# -----------------------------------------------------------------------------


class EWC(nn.Module):
    """
    EWC baseline: backbone (StaticMLP or SimpleCNN) with diagonal Fisher and
    parameter snapshots per task. Loss = CE + ewc_lambda * sum_t (F_t * (theta - theta_t*)^2).
    backbone: 'mlp' (default) for Split-MNIST, 'cnn' for Split-CIFAR-10.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 256,
        num_classes: int = 2,
        ewc_lambda: float = 400.0,
        backbone: str = "mlp",
    ) -> None:
        super().__init__()
        if backbone == "cnn":
            self.backbone = SimpleCNN(hidden_size=hidden_size, num_classes=num_classes)
        else:
            self.backbone = StaticMLP(
                input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
            )
        self.ewc_lambda = ewc_lambda
        self._fisher_snapshots: List[Tuple[dict, dict]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def consolidate(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
        """
        Compute diagonal Fisher on current task data and store current params.
        Call after training on each task.
        """
        self.eval()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in self.named_parameters() if p.requires_grad}
        n_samples = 0

        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            self.zero_grad()
            logits = self(x)
            log_probs = F.log_softmax(logits, dim=1)
            for i in range(x.size(0)):
                self.zero_grad()
                log_probs[i : i + 1].sum().backward(retain_graph=(i < x.size(0) - 1))
                for n, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data ** 2
                n_samples += 1

        for n in fisher:
            fisher[n] /= max(n_samples, 1)

        star_params = {n: p.detach().clone() for n, p in self.named_parameters() if p.requires_grad}
        self._fisher_snapshots.append((fisher, star_params))
        self.train()

    def compute_ewc_loss(self) -> torch.Tensor:
        """Sum over previous tasks: ewc_lambda * (F_t * (theta - theta_t*)^2)."""
        if not self._fisher_snapshots:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for fisher, star in self._fisher_snapshots:
            for n, p in self.named_parameters():
                if n in fisher and n in star:
                    loss = loss + (fisher[n] * (p - star[n]) ** 2).sum()
        return self.ewc_lambda * loss
