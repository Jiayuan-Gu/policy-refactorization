import numpy as np
import torch


def print_dict(d):
    for k, v in d.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, v.shape)
        else:
            print(k, v)


def add_prefix(tag, prefix):
    if prefix:
        return prefix + '/' + tag
    else:
        return tag
