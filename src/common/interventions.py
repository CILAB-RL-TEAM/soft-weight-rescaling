import torch
from torch import nn


# ----- Full Reset -----


@torch.no_grad()
def full_reset(model: torch.nn.Module, init_model: torch.nn.Module):
    for param, init_param in zip(model.parameters(), init_model.parameters()):
        param.data = init_param.data.clone()


# ----- Head Reset -----


@torch.no_grad()
def head_reset(model: torch.nn.Module, init_model: torch.nn.Module):
    # TODO: implement here
    pass


# ----- L2 Regularization -----


def l2_regularization(model: nn.Module):
    loss = 0
    for param in model.parameters():
        loss += torch.sum(param ** 2)
    return loss


# ----- L2 Init Regularization -----


def l2_init_regularization(source_model: nn.Module, target_model: nn.Module):
    loss = 0
    for src_param, tar_param in zip(source_model.parameters(), target_model.parameters()):
        loss += torch.sum((src_param - tar_param.data) ** 2)
    return loss


# ----- S&P -----


@torch.no_grad()
def shrink_and_perturb(model: nn.Module, init_model: nn.Module, shrink_coef: float):
    for param, init_param in zip(model.parameters(), init_model.parameters()):
        param.data = (1 - shrink_coef) * param.data.clone() + shrink_coef * init_param.data.clone()


# ----- SWR -----


@torch.no_grad()
def get_weight_norms(model: nn.Module):
    weight_norms = {}
    for name, param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            weight_norms[name] = torch.norm(param.data).item()
    return weight_norms


def soft_weight_rescaling(model: nn.Module, init_weight_norm: dict, coef: float):
    cum_c = 1.0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                curr_norm = torch.norm(param.data)
                c = coef * init_weight_norm[name] / curr_norm + (1 - coef)
                param.data.mul_(c)
                cum_c *= c
            elif 'bias' in name:
                param.data.mul_(cum_c)