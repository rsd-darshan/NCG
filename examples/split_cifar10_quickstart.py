"""
Minimal Split-CIFAR-10 example using NCG (CNN backbone).
Full pipeline: python scripts/main.py --benchmark split_cifar10
"""

import ncg
from ncg.metrics import compute_forgetting

def main():
    device = ncg.get_device()
    ncg.set_seed(42)
    model = ncg.NCGModelCNN(hidden_size=256, num_classes=2, max_hidden=512)
    tasks = ncg.get_split_cifar10_tasks(data_dir="./data", batch_size=64)
    res = ncg.train_ncg(model, tasks, device, epochs_per_task=2, verbose=True)
    forgetting = compute_forgetting({"NCG": res["task_accs"]})["NCG"]
    print(f"Final forgetting: {forgetting:.4f}")

if __name__ == "__main__":
    main()
