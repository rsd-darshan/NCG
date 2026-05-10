"""
Example: Using NCG on a completely custom model.
No need to inherit from NCGModel.
"""
import torch
import torch.nn as nn
from ncg import (
    train_ncg,
    get_split_mnist_tasks,
    get_device,
    LinearGrowthAdapter,
    StandaloneMetaParameters,
)

# 1. Your completely custom model
class MyCustomMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.relu(self.fc1(x))
        return self.fc2(h), h  # must return (logits, hidden)

# 2. Attach NCG components
model = MyCustomMLP()
adapter = LinearGrowthAdapter(
    layer_getter=lambda m: m.fc1,
    downstream_getter=lambda m: m.fc2,
    growth_units=64,
)
meta = StandaloneMetaParameters(
    alpha_init=0.5,
    beta_init=0.01,
    lambda_init=0.5,
)

# 3. Train with full NCG self-regulation
tasks = get_split_mnist_tasks(data_dir="./data", batch_size=64)
device = get_device()
results = train_ncg(
    model=model,
    tasks=tasks,
    device=device,
    adapter=adapter,
    meta=meta,
    novelty_layer_getter=lambda m: m.fc1,
    epochs_per_task=10,
)

print(f"Growth events: {sum(1 for s in results['hidden_size_per_epoch'] if s > 256)}")
