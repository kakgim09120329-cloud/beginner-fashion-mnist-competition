from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class NetworkConfig:
    in_channels: int = 1
    hidden_size2: int = 64
    output_size: int = 10
    learning_rate: float = 0.05
    batch_size: int = 64
    seed: int = 42


class SimpleCNN(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=config.in_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, config.hidden_size2)
        self.fc2 = nn.Linear(config.hidden_size2, config.output_size)
        self.relu = nn.ReLU()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def to_state(self) -> dict[str, object]:
        return {
            "model_type": "SimpleCNN",
            "config": {
                "in_channels": self.config.in_channels,
                "hidden_size2": self.config.hidden_size2,
                "output_size": self.config.output_size,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "seed": self.config.seed,
            },
            "params": self.state_dict(),
        }

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "SimpleCNN":
        config_obj = state.get("config")
        if not isinstance(config_obj, dict):
            raise ValueError("Invalid state: 'config' must be a dict")

        config = NetworkConfig(
            in_channels=int(config_obj.get("in_channels", 1)),
            hidden_size2=int(config_obj.get("hidden_size2", 64)),
            output_size=int(config_obj.get("output_size", 10)),
            learning_rate=float(config_obj.get("learning_rate", 0.05)),
            batch_size=int(config_obj.get("batch_size", 64)),
            seed=int(config_obj.get("seed", 42)),
        )

        model = cls(config)
        model.load_state_dict(state["params"])
        model.eval()
        return model
