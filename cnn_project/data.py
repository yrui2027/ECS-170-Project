from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetMetadata:
    name: str
    input_shape: tuple[int, int, int]
    original_shape: tuple[int, ...]
    num_classes: int
    class_names: list[int]
    train_size: int
    test_size: int


def load_pickle_dataset(dataset_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load the original pickle file without modifying it."""
    with open(dataset_path, "rb") as file:
        return pickle.load(file)


def _is_grayscale_rgb(image: np.ndarray) -> bool:
    """Detect ORL-style RGB images whose channels all contain the same values."""
    return (
        image.ndim == 3
        and image.shape[2] == 3
        and np.array_equal(image[:, :, 0], image[:, :, 1])
        and np.array_equal(image[:, :, 1], image[:, :, 2])
    )


def infer_dataset_metadata(
    dataset_name: str,
    loaded_dataset: dict[str, list[dict[str, Any]]],
) -> DatasetMetadata:
    """Infer image shape and label information directly from the pickle data."""
    train_split = loaded_dataset["train"]
    test_split = loaded_dataset["test"]
    sample_image = np.asarray(train_split[0]["image"])
    class_names = sorted({int(item["label"]) for item in train_split + test_split})

    if sample_image.ndim == 2:
        input_shape = (1, sample_image.shape[0], sample_image.shape[1])
    elif sample_image.ndim == 3 and _is_grayscale_rgb(sample_image):
        input_shape = (1, sample_image.shape[0], sample_image.shape[1])
    elif sample_image.ndim == 3:
        input_shape = (sample_image.shape[2], sample_image.shape[0], sample_image.shape[1])
    else:
        raise ValueError(f"Unsupported image shape for {dataset_name}: {sample_image.shape}")

    return DatasetMetadata(
        name=dataset_name,
        input_shape=input_shape,
        original_shape=tuple(sample_image.shape),
        num_classes=len(class_names),
        class_names=class_names,
        train_size=len(train_split),
        test_size=len(test_split),
    )


class PickleImageDataset(Dataset):
    """PyTorch dataset for MNIST, ORL, and CIFAR pickle files."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        max_samples: int | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.loaded_dataset = load_pickle_dataset(self.dataset_path)
        self.metadata = infer_dataset_metadata(self.dataset_name, self.loaded_dataset)
        self.label_to_index = {
            original_label: index for index, original_label in enumerate(self.metadata.class_names)
        }

        split_data = self.loaded_dataset[split]
        if max_samples is not None:
            split_data = split_data[:max_samples]
        self.samples = split_data

    def __len__(self) -> int:
        return len(self.samples)

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy images into normalized channel-first tensors."""
        image = image.astype(np.float32)
        expected_channels, _, _ = self.metadata.input_shape

        if image.ndim == 2 and expected_channels == 1:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 2 and expected_channels == 3:
            image = np.stack([image, image, image], axis=0)
        elif image.ndim == 3 and expected_channels == 1 and _is_grayscale_rgb(image):
            image = np.expand_dims(image[:, :, 0], axis=0)
        elif image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if image.max() > 1.0:
            image = image / 255.0

        return torch.from_numpy(image)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = np.asarray(sample["image"])
        original_label = int(sample["label"])
        mapped_label = self.label_to_index[original_label]
        return self._prepare_image(image), torch.tensor(mapped_label, dtype=torch.long)
