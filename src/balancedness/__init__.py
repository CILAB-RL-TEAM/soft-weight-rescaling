import os
import requests
from io import BytesIO
from copy import deepcopy
import zipfile
import torch
from torch import nn
from torchvision import datasets, transforms

from src.common.const import DATA_DIR


def get_dataloader(dataset: str, batch_size: int):
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load datasets
        trainset = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        testset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Load datasets
        trainset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        # Load datasets
        trainset = datasets.CIFAR100(root=DATA_DIR, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root=DATA_DIR, train=False, download=True, transform=transform)
    elif dataset == 'TinyImageNet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])

        # Dataset directories
        root_dir = DATA_DIR
        train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val')
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            # Download Tiny ImageNet dataset
            response = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            if response.status_code == 200:
                with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(root_dir)
            else:
                raise Exception(f"Failed to download dataset, status code: {response.status_code}")

            # Create train and validation directories
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Separate validation images into separate sub-folders
            val_annotations_file = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'val_annotations.txt')
            val_images_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val', 'images')

            # Read validation annotations
            with open(val_annotations_file, 'r') as f:
                data = f.readlines()

            val_img_dict = {}
            for line in data:
                words = line.split('\t')
                val_img_dict[words[0]] = words[1]

            # Create directories and move images
            val_counters = {label: 0 for label in set(val_img_dict.values())}
            for image, label in val_img_dict.items():
                folder_path = os.path.join(val_dir, label)
                os.makedirs(folder_path, exist_ok=True)
                os.makedirs(os.path.join(folder_path, 'images'), exist_ok=True)

                val_counters[label] += 1
                os.rename(os.path.join(val_images_dir, image),
                          os.path.join(folder_path, label + f'_{val_counters[label]}.JPEG'))

                os.rmdir(os.path.join(folder_path, 'images'))
            os.rmdir(os.path.join(val_dir, 'images'))

        # Load datasets
        trainset = datasets.ImageFolder(root=train_dir, transform=transform)
        testset = datasets.ImageFolder(root=val_dir, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def lpq_norm(model: nn.Module, p=2, q=2):
    layer_norms = []

    with torch.no_grad():
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                inner_sum = torch.sum(torch.abs(layer.weight.data) ** p, dim=1)
                inner_power = inner_sum ** (q / p)
                layer_norms.append(inner_power)

    total_sum = torch.sum(torch.cat(layer_norms))
    model_norm = total_sum ** (1 / q)

    return model_norm


def min_norm_fix(
    model: nn.Module,
    p: int = 2,
    q: int = 2,
    tol: float = 1e-5,
    max_iter: int = 100,
    device: torch.device = torch.device('cpu'),
):
    def _calculate_rho(model, p=2, q=2):
        rho_values = []

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                W = layer.weight.data
                rho_i = torch.sum(torch.abs(W) ** p, dim=1) ** (q / p)
                rho_values.append(rho_i)

        return rho_values

    def _calculate_pi(model, p=2):
        pi_values = []

        for layer in model.children():
            if isinstance(layer, nn.Linear):
                W = layer.weight.data
                abs_W_p = torch.abs(W) ** p
                sum_abs_W_p = torch.sum(abs_W_p, dim=1, keepdim=True)
                pi_ij = abs_W_p / sum_abs_W_p
                pi_values.append(pi_ij)

        return pi_values

    def _calculate_scaling_factors(rho_values, pi_values, p=2, q=2):
        scaling_factors = []
        scaling_factors.append(torch.ones(pi_values[0].shape[1], device=device))

        for i in range(1, len(rho_values)):
            rho_prev = rho_values[i - 1]
            rho_curr = rho_values[i]
            pi = pi_values[i]

            pi_T = pi.transpose(0, 1)
            numerator = torch.matmul(pi_T, rho_curr)
            denominator = rho_prev
            a = (numerator / denominator) ** (1 / (4 * max(p, q)))
            scaling_factors.append(a)

        scaling_factors.append(torch.ones(pi_values[-1].shape[0], device=device))

        return scaling_factors

    def _rescale_weights(model, scaling_factors):
        layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]

        for i, layer in enumerate(layers):
            # set device
            input_scaling = scaling_factors[i]
            output_scaling = scaling_factors[i + 1]

            layer.weight.data *= output_scaling.unsqueeze(1) / input_scaling.unsqueeze(0)
            layer.bias.data *= output_scaling

    for _ in range(max_iter):
        rho = _calculate_rho(model, p, q)
        pi = _calculate_pi(model, p)
        scaling_factors = _calculate_scaling_factors(rho, pi, p, q)

        _rescale_weights(model, scaling_factors)

        all_scaling_factors = torch.cat(scaling_factors)
        delta = torch.max(torch.abs(all_scaling_factors - 1.0))
        if delta < tol:
            break


def get_model_balancedness(model: nn.Module, p=2, q=2, tol=1e-5, max_iter=100, device: torch.device = torch.device('cpu')):
    # Compute optimal norm
    copy_model = deepcopy(model)
    min_norm_fix(copy_model, p, q, tol, max_iter, device)
    optimal_norm = lpq_norm(copy_model, p, q)

    # Compute current norm
    current_norm = lpq_norm(model, p, q)

    return optimal_norm / current_norm