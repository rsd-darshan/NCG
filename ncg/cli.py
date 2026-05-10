"""Command-line entrypoint for running NCG experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ncg.evaluate import run_all_seeds
from ncg.plot import plot_all
from ncg.train import get_device, get_split_cifar10_tasks, get_split_mnist_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NCG research runner (Split-MNIST / Split-CIFAR-10)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="split_mnist",
        choices=["split_mnist", "split_cifar10"],
        help="Benchmark: split_mnist (default) or split_cifar10",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds (default: 42 43 44 45 46)",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for data")
    parser.add_argument(
        "--results_dir", type=str, default="./results", help="Directory for results and figures"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="./checkpoints", help="Directory for checkpoints"
    )
    parser.add_argument("--epochs_per_task", type=int, default=10, help="Epochs per task")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    device = get_device()
    if args.benchmark == "split_cifar10":
        def tasks_fn():
            return get_split_cifar10_tasks(data_dir=args.data_dir, batch_size=args.batch_size)
    else:
        def tasks_fn():
            return get_split_mnist_tasks(data_dir=args.data_dir, batch_size=args.batch_size)

    output_dir = "/kaggle/working/output" if os.path.exists("/kaggle") else None
    print("Running experiments for seeds:", args.seeds, "| benchmark:", args.benchmark)
    agg = run_all_seeds(
        seed_list=args.seeds,
        tasks_fn=tasks_fn,
        device=device,
        results_dir=results_dir,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        benchmark=args.benchmark,
        output_dir=output_dir,
    )

    ncg_logs = agg["ncg_logs"]
    task_accs = agg["task_accs"]
    task_accs_std = agg.get("task_accs_std")
    if ncg_logs:
        plot_all(
            ncg_logs[0],
            task_accs,
            results_dir,
            task_accs_std=task_accs_std,
            epochs_per_task=args.epochs_per_task,
            num_tasks=5,
        )

    df = agg["results_table"]
    print("\n" + "=" * 60)
    print("Results summary (mean across seeds)")
    print("=" * 60)
    print(f"{'Model':<16} | {'Avg Acc':<8} | {'Forgetting':<10} | {'BWT':<8} | {'FWT':<8}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(
            f"{row['Model']:<16} | {row['Avg_Final_Acc']:<8.4f} | "
            f"{row['Forgetting']:<10.4f} | {row['BWT']:<8.4f} | {row['FWT']:<8.4f}"
        )
    print("=" * 60)
    print(f"\nResults saved to {results_dir}")
    print(f"  - {results_dir / 'results_table.csv'}")
    print(f"  - {results_dir / 'aggregated_results.csv'}")
    print(f"  - {results_dir / 'figures'}/")


if __name__ == "__main__":
    main()
