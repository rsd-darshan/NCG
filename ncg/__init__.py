"""
NCG — Novelty-triggered Capacity Growth.

Self-regulating continual learning that grows capacity only when needed.
"""

from ncg.adapters import (
    Conv2dGrowthAdapter,
    GrowthAdapter,
    LinearGrowthAdapter,
    NCGGrowthAdapter,
    TransformerGrowthAdapter,
)
from ncg.meta import StandaloneMetaParameters
from ncg.model import (
    DENModel,
    DENModelCNN,
    EWC,
    NCGModel,
    NCGModelCNN,
    StaticMLP,
)
from ncg.novelty import NoveltyMonitor
from ncg.train import (
    get_device,
    get_split_cifar10_tasks,
    get_split_mnist_tasks,
    set_seed,
    train_den,
    train_ewc,
    train_ncg,
    train_static_mlp,
)

__all__ = [
    "GrowthAdapter",
    "LinearGrowthAdapter",
    "Conv2dGrowthAdapter",
    "TransformerGrowthAdapter",
    "NCGGrowthAdapter",
    "NCGModel",
    "NCGModelCNN",
    "StaticMLP",
    "DENModel",
    "DENModelCNN",
    "EWC",
    "train_ncg",
    "train_static_mlp",
    "train_den",
    "train_ewc",
    "get_device",
    "set_seed",
    "get_split_mnist_tasks",
    "get_split_cifar10_tasks",
    "StandaloneMetaParameters",
    "NoveltyMonitor",
]
