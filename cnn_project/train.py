from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_project.data import PickleImageDataset
from cnn_project.model import CNNConfig, SimpleCNN


@dataclass
class TrainingHistory:
    train_loss: list[float]
    train_accuracy: list[float]
    test_loss: list[float]
    test_accuracy: list[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data_loaders(
    dataset_path: Path,
    batch_size: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, PickleImageDataset]:
    train_dataset = PickleImageDataset(dataset_path, split="train", max_samples=max_train_samples)
    test_dataset = PickleImageDataset(dataset_path, split="test", max_samples=max_test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_examples += images.size(0)

    average_loss = total_loss / total_examples
    accuracy = correct_predictions / total_examples
    return average_loss, accuracy


def train_model(
    dataset_path: Path,
    config: CNNConfig,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
) -> dict[str, object]:
    set_seed(seed)

    train_loader, test_loader, train_dataset = build_data_loaders(
        dataset_path=dataset_path,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )

    metadata = train_dataset.metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(metadata.input_shape, metadata.num_classes, config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = TrainingHistory([], [], [], [])
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)

        history.train_loss.append(train_loss)
        history.train_accuracy.append(train_accuracy)
        history.test_loss.append(test_loss)
        history.test_accuracy.append(test_accuracy)

        print(
            f"[{metadata.name:>5}] {config.name:<13} "
            f"epoch {epoch:>2}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
            f"test loss {test_loss:.4f} acc {test_accuracy:.4f}"
        )

    elapsed_seconds = time.time() - start_time

    model_dir = output_dir / "models"
    history_dir = output_dir / "history"
    figure_dir = output_dir / "figures"
    model_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{metadata.name}_{config.name}"
    model_path = model_dir / f"{run_name}.pt"
    history_path = history_dir / f"{run_name}.json"
    figure_path = figure_dir / f"{run_name}_learning_curves.png"

    torch.save(model.state_dict(), model_path)

    history_payload = {
        "dataset": metadata.name,
        "config": asdict(config),
        "metadata": {
            "input_shape": metadata.input_shape,
            "original_shape": metadata.original_shape,
            "num_classes": metadata.num_classes,
            "class_names": metadata.class_names,
            "train_size": metadata.train_size,
            "test_size": metadata.test_size,
        },
        "history": asdict(history),
    }
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    save_learning_curves(history, metadata.name, config.name, figure_path)

    return {
        "dataset": metadata.name,
        "experiment": config.name,
        "input_shape": str(metadata.input_shape),
        "original_shape": str(metadata.original_shape),
        "num_classes": metadata.num_classes,
        "train_size": len(train_loader.dataset),
        "test_size": len(test_loader.dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_train_loss": history.train_loss[-1],
        "final_train_accuracy": history.train_accuracy[-1],
        "final_test_loss": history.test_loss[-1],
        "final_test_accuracy": history.test_accuracy[-1],
        "elapsed_seconds": elapsed_seconds,
        "model_path": str(model_path),
        "curve_path": str(figure_path),
    }


def save_learning_curves(
    history: TrainingHistory,
    dataset_name: str,
    experiment_name: str,
    output_path: Path,
) -> None:
    epochs = range(1, len(history.train_loss) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, history.test_loss, marker="s", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.train_accuracy, marker="o", label="Train Accuracy")
    plt.plot(epochs, history.test_accuracy, marker="s", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.suptitle(f"{dataset_name} - {experiment_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_results_table(results: list[dict[str, object]], output_dir: Path) -> None:
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(results)
    dataframe = dataframe.sort_values(["dataset", "experiment"]).reset_index(drop=True)

    csv_path = results_dir / "evaluation_results.csv"
    markdown_path = results_dir / "evaluation_results.md"

    dataframe.to_csv(csv_path, index=False)
    markdown_path.write_text(dataframe_to_markdown(dataframe), encoding="utf-8")


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    """Create a Markdown table without needing the optional tabulate package."""
    headers = [str(column) for column in dataframe.columns]
    separator = ["---"] * len(headers)
    rows = [headers, separator]

    for row in dataframe.itertuples(index=False):
        rows.append([str(value) for value in row])

    return "\n".join("| " + " | ".join(row) + " |" for row in rows)
