# NCG — Novelty-triggered Capacity Growth

[![CI](https://github.com/rsd-darshan/NCG/actions/workflows/ci.yml/badge.svg)](https://github.com/rsd-darshan/NCG/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20prototype-purple.svg)](https://github.com/rsd-darshan/NCG)

**Self-regulating continual learning that expands model capacity only when needed.**

NCG is a continual-learning framework that treats growth as a decision problem, not a fixed design choice.  
It combines a novelty signal, learnable meta-parameters, and a gated knowledge memory to preserve past competence while adapting to new tasks.

---

## Motivation

Modern autonomous systems and future AGI agents must learn continuously without catastrophic forgetting.  
Static-capacity models are often either under-parameterized (forgetting) or over-parameterized (inefficient). NCG explores a third path: **adaptive capacity** that grows only when learning dynamics indicate it is necessary.

Why this matters:
- Continual adaptation is essential for long-horizon autonomous systems.
- Compute/memory should scale with task novelty, not with worst-case assumptions.
- Growth policies can be learned from data rather than hand-engineered.

## Core Idea

NCG monitors training with:
- **Novelty signal** (is current experience still information-rich?)
- **Meta-parameters** `alpha`, `beta`, `lambda` (exploration, complexity penalty, regularization)
- **Knowledge embedding `K`** with gated updates to retain transferable structure

Capacity growth is triggered only when all are true:
1. novelty is low (`novelty < 0.5`)
2. regularization is sufficiently active (`lambda > 0.3`)
3. validation performance has plateaued (windowed delta < `0.005`)

## How NCG Works

```text
Task stream --> Encoder/Backbone --> Hidden representation h --> Classifier
                    |                      |
                    |                      +--> Gated memory write: K <- (1-g)*K + g*h_mean
                    |
                    +--> Novelty monitor + meta-parameters (alpha, beta, lambda)
                                      |
                                      +--> Growth decision (if low novelty + plateau + lambda high)
                                                       |
                                                       +--> Expand hidden units (+64), preserve old weights
```

| Component | Role in the system |
|---|---|
| **Meta-parameters (`alpha`, `beta`, `lambda`)** | Adapt exploration/regularization/complexity pressure during learning. |
| **Knowledge embedding `K`** | Compact memory updated through a learned gate to preserve prior task knowledge. |
| **Novelty monitor** | Estimates whether incoming batches are still adding new signal. |
| **Growth trigger** | Activates only under low novelty + high regularization + validation plateau. |
| **Growth operator** | Adds hidden units while preserving learned weights and safe initialization for new params. |

In short: NCG delays growth until optimization dynamics suggest representation bottlenecks.

## Key Features

- **Autonomous growth policy** driven by learned signals rather than manual schedules.
- **Forgetting-aware design** via memory gating + selective capacity expansion.
- **Ablation-ready** setup (`NCG`, `NCG-NoGrowth`, `NCG-FixedMeta`) for causal analysis.
- **Benchmark coverage** on Split-MNIST and Split-CIFAR-10.
- **Reproducible scripts** for multi-seed evaluation and diagnostics.

## Results

### Split-MNIST (mean over seeds)

| Model           | Avg Acc | Forgetting | BWT    | FWT   |
|-----------------|---------|------------|--------|-------|
| NCG             | 0.551   | 0.331      | -0.407 | 0.024 |
| NCG-NoGrowth    | 0.552   | 0.373      | -0.466 | 0.039 |
| NCG-FixedMeta   | 0.557   | 0.356      | -0.445 | 0.051 |
| DEN             | 0.580   | 0.417      | -0.521 | 0.032 |
| StaticMLP-256   | 0.579   | 0.419      | -0.524 | 0.035 |
| StaticMLP-448   | 0.573   | 0.421      | -0.531 | 0.027 |
| StaticMLP-512   | 0.572   | 0.425      | -0.531 | 0.034 |
| EWC             | 0.732   | 0.229      | -0.286 | 0.026 |

### Split-CIFAR-10 (mean over seeds)

| Model           | Avg Acc | Forgetting | BWT    | FWT   |
|-----------------|---------|------------|--------|-------|
| NCG             | 0.673   | 0.084      | -0.086 | 0.061 |
| NCG-NoGrowth    | 0.666   | 0.103      | -0.108 | 0.076 |
| NCG-FixedMeta   | 0.673   | 0.096      | -0.119 | 0.077 |
| DEN             | 0.688   | 0.222      | -0.278 | 0.088 |
| StaticMLP-256   | 0.683   | 0.230      | -0.288 | 0.086 |
| StaticMLP-512   | 0.687   | 0.227      | -0.284 | 0.088 |
| EWC             | 0.702   | 0.163      | -0.203 | 0.088 |

NCG achieves **~63% lower forgetting** than `StaticMLP-256` on Split-CIFAR-10 (`0.084` vs `0.230`) while growing capacity only when trigger conditions are met.

### Metric Definitions

- **Avg Acc**: final mean test accuracy across tasks (higher is better).
- **Forgetting**: average drop from each task's best historical accuracy to final accuracy (lower is better).
- **BWT (Backward Transfer)**: effect of learning new tasks on old ones; less negative is better.
- **FWT (Forward Transfer)**: transfer to future tasks before training on them; higher is better.

## Figures / Plots

> Tip: keep visuals compact for faster scanning.

<img src="results/figures/accuracy_over_tasks.png" alt="Split-MNIST accuracy over tasks" width="420" />
<img src="results/figures/ncg_growth.png" alt="Split-MNIST NCG growth over epochs" width="420" />

<img src="results_cifar10/figures/accuracy_over_tasks.png" alt="Split-CIFAR-10 accuracy over tasks" width="420" />
<img src="results_cifar10/figures/ncg_growth.png" alt="Split-CIFAR-10 NCG growth over epochs" width="420" />

Recommended additional visuals:
- novelty trajectory over epochs
- `alpha/beta/lambda` evolution
- forgetting heatmaps for each baseline

## Installation

Package import is:

```python
import ncg
```

PyPI release is planned; for now install from source:

```bash
git clone https://github.com/rsd-darshan/NCG.git
cd NCG
pip install -e .
```

For contributors:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import ncg
from ncg.metrics import compute_forgetting

ncg.set_seed(42)
device = ncg.get_device()

model = ncg.NCGModel(hidden_size=256, num_classes=2, max_hidden=512)
tasks = ncg.get_split_mnist_tasks(data_dir="./data", batch_size=64)

result = ncg.train_ncg(
    model=model,
    tasks=tasks,
    device=device,
    epochs_per_task=2,
    verbose=True,
)

forgetting = compute_forgetting({"NCG": result["task_accs"]})["NCG"]
print(f"Final forgetting: {forgetting:.4f}")
```

## Running Experiments

```bash
python scripts/main.py --benchmark split_mnist --seeds 42 43 44 45 46 47 48 49 50 51
python scripts/main.py --benchmark split_cifar10 --seeds 42 43 44 45 46 47 48 49 50 51
```

CI note: GitHub Actions runs unit tests on Python 3.9/3.11, excluding `integration` tests that require dataset downloads. A separate scheduled/manual integration workflow runs `integration` tests.

Results policy: this repository tracks curated result tables/figures used in the README, while large training artifacts (checkpoints, raw dumps) are ignored.

## Reproducibility

- Environment: Python 3.9+ and PyTorch 2.0+.
- Determinism: set a fixed seed via `ncg.set_seed(seed)` and run multiple seeds for robust reporting.
- Suggested command:

```bash
python scripts/main.py --benchmark split_cifar10 --seeds 42 43 44 45 46
```

- Output locations:
  - `results/results_table.csv`
  - `results_cifar10/results_table.csv`
  - `results*/figures/`

## Convergence Diagnostics

```python
import pickle
from ncg.math.convergence import run_diagnostics

with open("results/ncg_logs.pkl", "rb") as f:
    logs = pickle.load(f)

run_diagnostics(logs["ncg_logs"][0])
```

## Limitations & Future Work

- Current evaluation is focused on image classification continual-learning benchmarks.
- Growth policy is validated on modest-scale backbones; large-model scaling is future work.
- Future directions:
  - language and multimodal continual-learning settings
  - stronger memory mechanisms and retrieval
  - benchmark expansion (longer task horizons, domain shift)
  - budget-aware growth for production agent systems

## Paper

- [NCG.pdf](NCG.pdf)

## Citation

```bibtex
@article{poudel2026ncg,
  title   = {Novelty-triggered Capacity Growth for Continual Learning},
  author  = {Poudel, Darshan},
  year    = {2026},
  note    = {Preprint. Under review.},
  url     = {https://github.com/rsd-darshan/NCG}
}
```

## License

MIT License. See [LICENSE](LICENSE).
