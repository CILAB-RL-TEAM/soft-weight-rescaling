from typing import Tuple
import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: int):
        super(MLP, self).__init__()
        in_features = int(np.prod(input_dim))
        self.fc1 = nn.Linear(in_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_dim)

    def forward(self, x):
        batch_size, *_ = x.shape
        x = x.view(batch_size, -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: int):
        super(CNN, self).__init__()
        if len(input_dim) == 2:
            channels, feature_size = (1, input_dim[1])
        elif len(input_dim) == 3:
            channels, feature_size = input_dim[:2]
        else:
            raise ValueError(f"Invalid input_dim: {input_dim}")

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, stride=1, padding=1)
        feature_size -= (5 - 1 - 2)  # conv -> kernel_size - stride - 2 * padding
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        feature_size //= 2  # pooling -> kernel_size
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=1)
        feature_size -= (5 - 1 - 2)  # conv -> kernel_size - stride - 2 * padding
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        feature_size //= 2  # pooling -> kernel_size
        self.fc1 = nn.Linear(16 * feature_size * feature_size, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN_BN(nn.Module):
    def __init__(self, input_dim: Tuple[int, int, int], output_dim: int):
        super(CNN_BN, self).__init__()
        if len(input_dim) == 2:
            channels, feature_size = (1, input_dim[1])
        elif len(input_dim) == 3:
            channels, feature_size = input_dim[:2]
        else:
            raise ValueError(f"Invalid input_dim: {input_dim}")

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, stride=1, padding=1, bias=False)
        feature_size -= (5 - 1 - 2)  # conv -> kernel_size - stride - 2 * padding
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        feature_size //= 2  # pooling -> kernel_size
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=1, bias=False)
        feature_size -= (5 - 1 - 2)  # conv -> kernel_size - stride - 2 * padding
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        feature_size //= 2  # pooling -> kernel_size
        self.fc1 = nn.Linear(16 * feature_size * feature_size, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VGG16:
    pass