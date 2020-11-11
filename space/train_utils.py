import warnings
import torch
from common.utils.scheduler import *


def build_optimizer(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.OPTIMIZER.LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError(f'Unsupported optimizer: {name}.')


def build_lr_scheduler(cfg, optimizer):
    name = cfg.LR_SCHEDULER.TYPE
    if name == '':
        warnings.warn('No lr_scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        lr_scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.LR_SCHEDULER.get(name, dict()),
        )
        return lr_scheduler
    else:
        raise ValueError(f'Unsupported lr_scheduler: {name}.')


def get_priors(cfg, verbose=True):
    """Get priors for VAE."""
    prior_dict = dict()
    for name in cfg.PRIOR:
        if isinstance(cfg.PRIOR[name], dict):  # scheduler
            # Note that CfgNode inherits dict
            scheduler_kwargs = {k: eval(v) if isinstance(v, str) else v for k, v in cfg.PRIOR[name].items()}
            scheduler_type = scheduler_kwargs.pop('type', LinearScheduler)
            prior_dict[name + '_prior'] = scheduler_type(**scheduler_kwargs)
        else:  # constant
            value = cfg.PRIOR[name]
            prior_dict[name + '_prior'] = eval(value) if isinstance(value, str) else value
        if verbose:
            print(name, prior_dict[name + '_prior'])
    return prior_dict


def get_weights(cfg, verbose=True):
    """Get weights for losses or others"""
    weight_dict = dict()
    for name in cfg.WEIGHT:
        if isinstance(cfg.WEIGHT[name], dict):
            # Note that CfgNode inherits dict
            scheduler_kwargs = {k: eval(v) if isinstance(v, str) else v for k, v in cfg.WEIGHT[name].items()}
            scheduler_type = scheduler_kwargs.pop('type', LinearScheduler)
            weight_dict[name] = scheduler_type(**scheduler_kwargs)
        else:
            value = cfg.WEIGHT[name]
            weight_dict[name] = eval(value) if isinstance(value, str) else value
        if verbose:
            print(name, weight_dict[name])
    return weight_dict
