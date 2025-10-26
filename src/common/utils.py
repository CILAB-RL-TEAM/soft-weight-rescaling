import os
import random
import numpy as np
import torch
from torch import nn

from src.common.models import MLP, CNN, CNN_BN, VGG16


def freeze_seed(seed: int = 42, reproduce: bool = True) -> None:
    # set seed for built-in random module
    random.seed(seed)

    # set seed for numpy
    np.random.seed(seed)

    # set seed for torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # set seed for cudnn
    if reproduce:
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    # Set PYTHONHASHSEED environment variable for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_model(model_name: str, dataset: str):
    if dataset in ['MNIST']:
        input_dim = (1, 28, 28)
        output_dim = 10
    elif dataset in ['CIFAR10']:
        input_dim = (3, 32, 32)
        output_dim = 10
    elif dataset in ['CIFAR100']:
        input_dim = (3, 32, 32)
        output_dim = 100
    elif dataset in ['TinyImageNet']:
        input_dim = (3, 64, 64)
        output_dim = 200
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if model_name == 'MLP':
        model = MLP(input_dim=input_dim, output_dim=output_dim)
    elif model_name == 'CNN':
        model = CNN(input_dim=input_dim, output_dim=output_dim)
    elif model_name == 'CNN_BN':
        model = CNN_BN(input_dim=input_dim, output_dim=output_dim)
    elif model_name == 'VGG16':
        model = VGG16(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def test_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    training_mode = model.training
    model.train(mode=False)

    total, correct = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            images, labels = images.to(device), labels.to(device)

            # forward
            outputs = model(images)

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train(mode=training_mode)
    return correct / total