import argparse
from collections import defaultdict
from copy import deepcopy
import wandb
from tqdm import tqdm
import numpy as np
import torch
from torch import nn

from src.common.utils import freeze_seed, build_model, test_model
from src.continual_learning import get_dataloader


def l2_init_regularization(source_model: nn.Module, target_model: nn.Module):
    loss = 0
    for src_param, tar_param in zip(source_model.parameters(), target_model.parameters()):
        loss += torch.sum((src_param - tar_param.data) ** 2)
    return loss


def main(args):
    freeze_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Initialize task
    trainloaders, testloader = get_dataloader(args.dataset, args.batch_size, args.n_chunks, args.access)

    # Initialize model
    model = build_model(args.model, args.dataset).to(device)
    init_model = deepcopy(model)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f'l2_init-{args.coef}',
        group=f'continual_learning',
        config=vars(args),
        mode="online" if args.use_wandb else "disabled",
        reinit=True,
    )

    # Pre-training
    global_step = 0
    for chunk in range(args.n_chunks):
        trainloader = trainloaders[chunk]

        if args.reinit:
            init_model = deepcopy(model)

        pbar = tqdm(range(args.n_epochs), leave=True)
        for epoch in pbar:
            pbar.set_description(f'[Chunk {chunk+1}/{args.n_chunks}] Epoch - {epoch}')

            logging_stats = defaultdict(list)

            # Train model one epoch
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # forward
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # calculate loss
                ce_loss = criterion(outputs, labels)
                loss = ce_loss
                logging_stats['ce_loss'].append(ce_loss.item())

                # regularization
                l2_init_loss = l2_init_regularization(model, init_model)
                loss += args.coef * l2_init_loss
                logging_stats['l2_init_loss'].append(l2_init_loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                logging_stats['loss'].append(loss.item())
                global_step += 1

            test_acc = test_model(model, testloader, device)
            pbar.set_postfix({'test_acc': f'{test_acc:.4f}'})
            wandb.log({
                'test/acc': test_acc,
                'global_step': global_step, 'global_epoch': epoch,
                **{f'train/{k}': np.mean(v) for k, v in logging_stats.items() if len(v) > 0},
            })

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10", "CIFAR100", "TinyImageNet"])
    parser.add_argument("--model", type=str, default="CNN", choices=["MLP", "CNN", "CNN_BN", "VGG16"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_chunks", type=int, default=10)
    parser.add_argument("--access", type=str, default="full", choices=["full", "limited"])
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--coef", type=float, default=1e-2)
    parser.add_argument("--reinit", action='store_true')
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='Plasticity')
    parser.add_argument('--wandb_project', type=str, default='soft_weight_rescaling')
    args = parser.parse_args()
    main(args)