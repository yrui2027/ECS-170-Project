from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CNNConfig:
    name: str
    conv_channels: tuple[int, ...]
    kernel_size: int
    hidden_dim: int
    dropout: float = 0.25


class SimpleCNN(nn.Module):
    """A configurable CNN that works for all three datasets."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        config: CNNConfig,
    ) -> None:
        super().__init__()
        in_channels, height, width = input_shape

        feature_layers: list[nn.Module] = []
        current_channels = in_channels

        for out_channels in config.conv_channels:
            feature_layers.extend(
                [
                    nn.Conv2d(
                        in_channels=current_channels,
                        out_channels=out_channels,
                        kernel_size=config.kernel_size,
                        padding=config.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            current_channels = out_channels

        self.features = nn.Sequential(*feature_layers)
        flattened_features = self._infer_flattened_size(input_shape)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes),
        )

    def _infer_flattened_size(self, input_shape: tuple[int, int, int]) -> int:
        with torch.no_grad():
            dummy_batch = torch.zeros(1, *input_shape)
            features = self.features(dummy_batch)
        return features.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_experiment_configs() -> list[CNNConfig]:
    """Required experiment configurations for the report."""
    return [
        CNNConfig(
            name="baseline",
            conv_channels=(16, 32),
            kernel_size=3,
            hidden_dim=128,
        ),
        CNNConfig(
            name="deeper",
            conv_channels=(16, 32, 64),
            kernel_size=3,
            hidden_dim=128,
        ),
        CNNConfig(
            name="larger_kernel",
            conv_channels=(16, 32),
            kernel_size=5,
            hidden_dim=128,
        ),
        CNNConfig(
            name="wider_hidden",
            conv_channels=(16, 32),
            kernel_size=3,
            hidden_dim=256,
        ),
    ]

