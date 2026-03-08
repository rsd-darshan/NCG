"""
Permuted-MNIST benchmark for NCG paper.
5 tasks, each a different fixed permutation of MNIST pixels.
Models: NCG, StaticMLP-256, StaticMLP-448, StaticMLP-512, EWC
Seeds: 42-46
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import random
import os
import json

# ── reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ── Permuted-MNIST data ───────────────────────────────────────────────────────
def get_permuted_mnist_tasks(n_tasks=5, seed=0, batch_size=64, data_dir="./data"):
    rng = np.random.RandomState(seed)
    permutations = [rng.permutation(784) for _ in range(n_tasks)]

    raw = datasets.MNIST(data_dir, train=True, download=True,
                         transform=transforms.ToTensor())
    raw_test = datasets.MNIST(data_dir, train=False, download=True,
                               transform=transforms.ToTensor())

    all_x = raw.data.float().view(-1, 784) / 255.0
    all_y = raw.targets
    all_x_test = raw_test.data.float().view(-1, 784) / 255.0
    all_y_test = raw_test.targets

    tasks = []
    for perm in permutations:
        px = all_x[:, perm]
        px_test = all_x_test[:, perm]

        n_val = int(0.1 * len(px))
        train_x, val_x = px[n_val:], px[:n_val]
        train_y, val_y = all_y[n_val:], all_y[:n_val]

        train_loader = DataLoader(TensorDataset(train_x, train_y),
                                  batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(val_x, val_y),
                                  batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(TensorDataset(px_test, all_y_test),
                                  batch_size=batch_size, shuffle=False)
        tasks.append({"train": train_loader, "val": val_loader, "test": test_loader})
    return tasks

# ── Models ────────────────────────────────────────────────────────────────────
class StaticMLP(nn.Module):
    def __init__(self, hidden_size=256, num_classes=10, input_size=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class NCG(nn.Module):
    def __init__(self, initial_hidden=256, num_classes=10, input_size=784):
        super().__init__()
        self.input_size  = input_size
        self.num_classes = num_classes
        self.hidden_size = initial_hidden

        self.fc1 = nn.Linear(input_size, initial_hidden)
        self.fc2 = nn.Linear(initial_hidden, num_classes)
        self.relu = nn.ReLU()

        # meta-parameters (raw, unconstrained)
        self.raw_alpha  = nn.Parameter(torch.tensor(0.0))
        self.raw_beta   = nn.Parameter(torch.tensor(-3.0))
        self.raw_lambda = nn.Parameter(torch.tensor(0.5))

    @property
    def alpha(self):  return torch.sigmoid(self.raw_alpha)
    @property
    def beta(self):   return torch.nn.functional.softplus(self.raw_beta)
    @property
    def lam(self):    return torch.sigmoid(self.raw_lambda)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)

    def grow(self, new_units=64):
        old_h = self.hidden_size
        new_h = old_h + new_units
        new_fc1 = nn.Linear(self.input_size, new_h)
        new_fc2 = nn.Linear(new_h, self.num_classes)
        with torch.no_grad():
            new_fc1.weight[:old_h] = self.fc1.weight
            new_fc1.bias[:old_h]   = self.fc1.bias
            new_fc1.weight[old_h:].normal_(0, 0.01)
            new_fc1.bias[old_h:].zero_()
            new_fc2.weight[:, :old_h] = self.fc2.weight
            new_fc2.weight[:, old_h:].normal_(0, 0.01)
            new_fc2.bias = nn.Parameter(self.fc2.bias.clone())
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.hidden_size = new_h
        print(f"  [GROWTH] {old_h} → {new_h} hidden units")

    def novelty(self, loader, device):
        self.eval()
        acts = []
        with torch.no_grad():
            for x, _ in loader:
                h = self.relu(self.fc1(x.to(device)))
                acts.append(h.cpu())
        acts = torch.cat(acts, 0)
        spread = acts.std(0).mean().item()
        return 1.0 - min(spread, 1.0)

# ── Training helpers ──────────────────────────────────────────────────────────
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1)
            correct += (pred == y.to(device)).sum().item()
            total   += y.size(0)
    return correct / total

def train_static(model, tasks, device, epochs=20, lr=5e-4, name="Static", seed=0):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    task_accs = []   # task_accs[epoch_overall][task_idx]
    val_accs_per_task = []

    for t_idx, task in enumerate(tasks):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        epoch_val_accs = []
        for ep in range(epochs):
            model.train()
            for x, y in task["train"]:
                optimizer.zero_grad()
                loss = criterion(model(x.to(device)), y.to(device))
                loss.backward()
                optimizer.step()
            scheduler.step()
            val_a = accuracy(model, task["val"], device)
            epoch_val_accs.append(val_a)
            print(f"[Seed {seed} | {name} | Task {t_idx+1}/{len(tasks)} | Epoch {ep+1}/{epochs}] "
                  f"Val Acc: {val_a:.2f}")

        # evaluate all tasks seen so far
        row = [accuracy(model, tasks[i]["test"], device) for i in range(t_idx + 1)]
        val_accs_per_task.append(row)
        print(f"  After task {t_idx+1}: {[f'{a:.3f}' for a in row]}")

    return {"val_accs": val_accs_per_task, "hidden_size": getattr(model, "hidden_size", None)}


def train_ncg(model, tasks, device, epochs=20, lr=5e-4, seed=0):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_accs_per_task = []

    for t_idx, task in enumerate(tasks):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        recent_vals = []
        for ep in range(epochs):
            model.train()
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss + model.beta * sum(p.pow(2).sum() for p in model.parameters()) \
                            + model.lam * loss.detach()
                loss.backward()
                optimizer.step()
            scheduler.step()
            val_a = accuracy(model, task["val"], device)
            recent_vals.append(val_a)
            print(f"[Seed {seed} | NCG | Task {t_idx+1}/{len(tasks)} | Epoch {ep+1}/{epochs}] "
                  f"Val Acc: {val_a:.2f} | Hidden: {model.hidden_size}")

            # growth trigger (only after 3 epochs)
            if len(recent_vals) >= 3:
                spread = max(recent_vals[-3:]) - min(recent_vals[-3:])
                nov = model.novelty(task["val"], device)
                if nov < 0.5 and model.lam.item() > 0.3 and spread < 0.005:
                    model.grow(64)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=epochs - ep - 1, eta_min=1e-5)

        row = [accuracy(model, tasks[i]["test"], device) for i in range(t_idx + 1)]
        val_accs_per_task.append(row)
        print(f"  After task {t_idx+1}: {[f'{a:.3f}' for a in row]}")

    return {"val_accs": val_accs_per_task, "hidden_size": model.hidden_size}


def train_ewc(model, tasks, device, epochs=20, lr=5e-4, importance=1000, seed=0):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    val_accs_per_task = []
    fisher_list, optima_list = [], []

    def compute_fisher(loader):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        for x, y in loader:
            model.zero_grad()
            out = model(x.to(device))
            loss = criterion(out, y.to(device))
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
        for n in fisher:
            fisher[n] /= len(loader)
        return fisher

    for t_idx, task in enumerate(tasks):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        for ep in range(epochs):
            model.train()
            for x, y in task["train"]:
                optimizer.zero_grad()
                loss = criterion(model(x.to(device)), y.to(device))
                for f, opt in zip(fisher_list, optima_list):
                    for n, p in model.named_parameters():
                        loss += (importance / 2) * (f[n] * (p - opt[n]).pow(2)).sum()
                loss.backward()
                optimizer.step()
            scheduler.step()
            val_a = accuracy(model, task["val"], device)
            print(f"[Seed {seed} | EWC | Task {t_idx+1}/{len(tasks)} | Epoch {ep+1}/{epochs}] "
                  f"Val Acc: {val_a:.2f}")

        fisher_list.append(compute_fisher(task["train"]))
        optima_list.append({n: p.clone().detach() for n, p in model.named_parameters()})

        row = [accuracy(model, tasks[i]["test"], device) for i in range(t_idx + 1)]
        val_accs_per_task.append(row)
        print(f"  After task {t_idx+1}: {[f'{a:.3f}' for a in row]}")

    return {"val_accs": val_accs_per_task}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(val_accs):
    """
    val_accs[i][j] = accuracy on task j after training on task i (i >= j)
    Returns forgetting, BWT, FWT, avg_final_acc
    """
    T = len(val_accs)
    # final accs after all tasks
    final_accs = [val_accs[T-1][j] if j < len(val_accs[T-1]) else 0 for j in range(T)]
    avg_acc = np.mean(final_accs)

    # forgetting: for each task j, max acc seen - final acc
    forgetting_vals = []
    for j in range(T - 1):
        max_acc = max(val_accs[i][j] for i in range(j, T) if j < len(val_accs[i]))
        forgetting_vals.append(max_acc - final_accs[j])
    forgetting = np.mean(forgetting_vals) if forgetting_vals else 0.0

    # BWT: average of (final_acc_j - acc_right_after_task_j) for j < T
    bwt_vals = []
    for j in range(T - 1):
        if j < len(val_accs[j]):
            bwt_vals.append(final_accs[j] - val_accs[j][j])
    bwt = np.mean(bwt_vals) if bwt_vals else 0.0

    # FWT: average acc on task j BEFORE training on it (using acc from row j-1)
    fwt_vals = []
    for j in range(1, T):
        if j-1 < len(val_accs) and j < len(val_accs[j-1]):
            fwt_vals.append(val_accs[j-1][j])
    # baseline: random = 0.1 for 10-class
    fwt = (np.mean(fwt_vals) - 0.1) if fwt_vals else 0.0

    return {"avg_acc": avg_acc, "forgetting": forgetting, "bwt": bwt, "fwt": fwt}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device  = get_device()
    seeds   = [42, 43, 44, 45, 46]
    N_TASKS = 5
    EPOCHS  = 20

    results = {m: {"avg_acc": [], "forgetting": [], "bwt": [], "fwt": []}
               for m in ["NCG", "StaticMLP-256", "StaticMLP-448", "StaticMLP-512", "EWC"]}

    for seed in seeds:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")
        set_seed(seed)
        tasks = get_permuted_mnist_tasks(n_tasks=N_TASKS, seed=seed, data_dir="./data")

        # NCG
        set_seed(seed)
        ncg = NCG(initial_hidden=256, num_classes=10)
        r = train_ncg(ncg, tasks, device, epochs=EPOCHS, seed=seed)
        m = compute_metrics(r["val_accs"])
        for k in m: results["NCG"][k].append(m[k])
        print(f"NCG final hidden: {r['hidden_size']}")

        # StaticMLP-256
        set_seed(seed)
        mlp256 = StaticMLP(256, 10)
        r = train_static(mlp256, tasks, device, epochs=EPOCHS, name="StaticMLP-256", seed=seed)
        m = compute_metrics(r["val_accs"])
        for k in m: results["StaticMLP-256"][k].append(m[k])

        # StaticMLP-448
        set_seed(seed)
        mlp448 = StaticMLP(448, 10)
        r = train_static(mlp448, tasks, device, epochs=EPOCHS, name="StaticMLP-448", seed=seed)
        m = compute_metrics(r["val_accs"])
        for k in m: results["StaticMLP-448"][k].append(m[k])

        # StaticMLP-512
        set_seed(seed)
        mlp512 = StaticMLP(512, 10)
        r = train_static(mlp512, tasks, device, epochs=EPOCHS, name="StaticMLP-512", seed=seed)
        m = compute_metrics(r["val_accs"])
        for k in m: results["StaticMLP-512"][k].append(m[k])

        # EWC
        set_seed(seed)
        ewc_model = StaticMLP(256, 10)
        r = train_ewc(ewc_model, tasks, device, epochs=EPOCHS, seed=seed)
        m = compute_metrics(r["val_accs"])
        for k in m: results["EWC"][k].append(m[k])

        print(f"\nSeed {seed} done.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("=== PERMUTED-MNIST RESULTS (mean ± std, 5 seeds) ===")
    print("="*65)
    print(f"{'Model':<16} {'Avg Acc':>12} {'Forgetting':>12} {'BWT':>12} {'FWT':>12}")
    print("-"*65)
    for model_name in ["NCG", "StaticMLP-256", "StaticMLP-448", "StaticMLP-512", "EWC"]:
        r = results[model_name]
        print(f"{model_name:<16} "
              f"{np.mean(r['avg_acc']):>6.3f}±{np.std(r['avg_acc'], ddof=1):.3f}  "
              f"{np.mean(r['forgetting']):>6.3f}±{np.std(r['forgetting'], ddof=1):.3f}  "
              f"{np.mean(r['bwt']):>6.3f}±{np.std(r['bwt'], ddof=1):.3f}  "
              f"{np.mean(r['fwt']):>6.3f}±{np.std(r['fwt'], ddof=1):.3f}")

    # save to JSON for later
    os.makedirs("results", exist_ok=True)
    with open("results/permuted_mnist_results.json", "w") as f:
        json.dump({k: {m: list(v) for m, v in vd.items()} for k, vd in results.items()}, f, indent=2)
    print("\nSaved to results/permuted_mnist_results.json")
