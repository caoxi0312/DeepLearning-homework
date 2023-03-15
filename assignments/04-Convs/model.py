import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    This is a CNN model
    """
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=24, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(24)
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(in_features=6912, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        y = self.conv1(x)
        y = F.relu(y)
        y = self.bn1(y)
        # print(y.size())
        y = self.conv2(y)
        y = F.relu(y)
        y = self.flat(y)
        # print(y.size())
        y = self.linear1(y)
        y = self.linear2(y)

        return y
