from __future__ import annotations

import argparse
from pathlib import Path

from cnn_project.model import get_experiment_configs
from cnn_project.train import save_results_table, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train configurable CNN models on the Stage 3 pickle datasets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the original MNIST, ORL, and CIFAR pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory where models, plots, and tables will be saved.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MNIST", "ORL", "CIFAR"],
        choices=["MNIST", "ORL", "CIFAR"],
        help="Which datasets to train on.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline", "deeper", "larger_kernel", "wider_hidden"],
        choices=["baseline", "deeper", "larger_kernel", "wider_hidden"],
        help="Which CNN configurations to run.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs for each run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=170, help="Random seed.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging. Leave unset for full training.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap for quick debugging. Leave unset for full evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_configs = {config.name: config for config in get_experiment_configs()}
    selected_configs = [all_configs[name] for name in args.experiments]

    results: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        dataset_path = args.data_dir / dataset_name
        for config in selected_configs:
            result = train_model(
                dataset_path=dataset_path,
                config=config,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
            )
            results.append(result)

    save_results_table(results, args.output_dir)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

