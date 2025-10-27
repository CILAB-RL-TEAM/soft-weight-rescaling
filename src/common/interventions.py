import numpy as np
import torch
from torch import nn

from src.common.models import MLP, CNN, CNN_BN, VGG16


# ----- Full Reset -----


@torch.no_grad()
def full_reset(model: torch.nn.Module, init_model: torch.nn.Module):
    for param, init_param in zip(model.parameters(), init_model.parameters()):
        param.data = init_param.data.clone()


# ----- Head Reset -----


@torch.no_grad()
def head_reset(model: torch.nn.Module, init_model: torch.nn.Module):
    if isinstance(model, MLP):
        model.fc3.weight.data = init_model.fc3.weight.data.clone()
        model.fc3.bias.data = init_model.fc3.bias.data.clone()
    elif isinstance(model, (CNN, CNN_BN)):
        model.fc1.weight.data = init_model.fc1.weight.data.clone()
        model.fc1.bias.data = init_model.fc1.bias.data.clone()
        model.fc2.weight.data = init_model.fc2.weight.data.clone()
        model.fc2.bias.data = init_model.fc2.bias.data.clone()
    elif isinstance(model, VGG16):
        # TODO: implement here
        pass
    else:
        raise ValueError(f"Unsupported model: {type(model)}")


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


@torch.no_grad()
def soft_weight_rescaling(
    model: nn.Module,
    init_weight_norm: dict,
    feature_extractor_coef: float,
    classifier_coef: float,
):
    cs = [1.0]
    if isinstance(model, MLP):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'fc1' in name or 'fc2' in name:
                    coef = feature_extractor_coef
                elif 'fc3' in name:
                    coef = classifier_coef
                else:
                    raise ValueError(f"Unknown layer type in parameter name: {name}")

                # Rescale weights
                curr_norm = param.data.norm().item()
                c = coef * init_weight_norm[name] / curr_norm + (1 - coef)
                param.data.mul_(c)
                cs.append(c)
            elif 'bias' in name:
                # Rescale bias with cumulative scaler
                cum_c = np.prod(cs).item()
                param.data.mul_(cum_c)
    elif isinstance(model, CNN):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'fc' in name:
                    coef = feature_extractor_coef
                elif 'conv' in name:
                    coef = classifier_coef
                else:
                    raise ValueError(f"Unknown layer type in parameter name: {name}")

                # Rescale weights
                curr_norm = param.data.norm().item()
                c = coef * init_weight_norm[name] / curr_norm + (1 - coef)
                param.data.mul_(c)
                cs.append(c)
            elif 'bias' in name:
                # Rescale bias with cumulative scaler
                cum_c = np.prod(cs).item()
                param.data.mul_(cum_c)
    elif isinstance(model, CNN_BN):
        for name, param in model.named_parameters():
            if 'weight' in name:
                if 'fc' in name:
                    coef = feature_extractor_coef
                elif 'conv' in name:
                    coef = classifier_coef
                elif 'bn' in name:
                    coef = classifier_coef
                else:
                    raise ValueError(f"Unknown layer type in parameter name: {name}")

                # Rescale weights
                curr_norm = param.data.norm().item()
                c = coef * init_weight_norm[name] / curr_norm + (1 - coef)
                param.data.mul_(c)

                # Cumulate scalers from last normalization layer
                if 'bn2' in name or 'fc' in name:
                    cs.append(c)
                else:
                    cs = [c]
            elif 'bias' in name:
                # Rescale bias with cumulative scaler
                cum_c = np.prod(cs).item()
                param.data.mul_(cum_c)
    elif isinstance(model, VGG16):
        # TODO: implement here
        pass
    else:
        raise ValueError(f"Unsupported model: {type(model)}")