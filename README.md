# NCG — Novelty-triggered Capacity Growth

**Novelty-triggered Capacity Growth (NCG)** is a continual learning research codebase. It implements a self-evolving architecture with learnable meta-parameters (α, β, λ), knowledge gating, and a growth trigger driven by novelty and validation plateau.

## Description

This repository contains the research prototype for **NCG** (Novelty-triggered Capacity Growth), including:

- **NCGModel** / **NCGModelCNN**: Growable MLP (and CNN variant for Split-CIFAR-10) with meta-parameters and knowledge embedding
- Baselines: StaticMLP, EWC, DEN (Dynamically Expandable Networks)
- Ablations: **NCG-NoGrowth**, **NCG-FixedMeta**
- Benchmarks: Split-MNIST, Split-CIFAR-10, Permuted-MNIST
- Evaluation: forgetting, BWT, FWT; p-value scripts (Welch t-tests)
- Plotting: accuracy over tasks, forgetting curves, NCG growth, meta-parameters, novelty

## Requirements

- Python 3.x
- PyTorch, torchvision, numpy, pandas, matplotlib, scipy

See `requirements.txt` for versions.

## Usage

- **Full pipeline (Split-MNIST or Split-CIFAR-10):**
  ```bash
  python main.py --benchmark split_mnist --seeds 42 43 44 45 46
  python main.py --benchmark split_cifar10 --seeds 42 43 44 45 46
  ```
- **P-values (NCG vs StaticMLP-256):**
  ```bash
  python compute_pvalues.py --benchmark split_mnist
  python compute_pvalues.py --benchmark split_cifar10
  ```
- **Plot NCG meta-parameters:**
  ```bash
  python plot_meta_params.py --results_dir ./results --seed 42
  ```
- **Permuted-MNIST:**
  ```bash
  python run_permuted_mnist.py
  ```

Results and figures are written to `./results` (and `./checkpoints` for model checkpoints). NCG logs are saved as `results/ncg_logs.pkl` for plotting.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ncg2025,
  title        = {NCG: Novelty-triggered Capacity Growth},
  author       = {...},
  year         = {2025},
  howpublished = {Research codebase},
  url          = {https://github.com/...}
}
```

(Update author and URL as appropriate.)

## License

See repository license file.
