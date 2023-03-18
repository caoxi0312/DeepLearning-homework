import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    This is a CNN model
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            stride=2,
            kernel_size=5,
            padding=2,
        )
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=24, stride=2, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=20, stride=2, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(24)
        self.bn3 = nn.BatchNorm2d(20)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(in_features=2048, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        y = self.conv1(x)
        y = F.relu(y)
        # y = self.bn1(y)
        # print(y.size())
        # y = self.conv2(y)
        y = self.pool(y)
        # y = F.relu(y)
        # print(y.size())
        # y = self.conv3(y)
        # y = F.relu(y)
        # y = self.bn3(y)
        y = self.flat(y)
        # print(y.size())
        y = self.linear1(y)
        y = F.relu(y)
        y = self.linear2(y)

        return y
