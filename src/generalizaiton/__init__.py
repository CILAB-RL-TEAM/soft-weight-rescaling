import os
import requests
from io import BytesIO
import zipfile
import torch
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