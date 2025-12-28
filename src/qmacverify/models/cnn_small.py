from __future__ import annotations

import torch
from torch import nn


class CNNSmall(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = nn.functional.avg_pool2d(x, kernel_size=4)
        x = self.flatten(x)
        return self.fc(x)


def build_model() -> CNNSmall:
    return CNNSmall()
