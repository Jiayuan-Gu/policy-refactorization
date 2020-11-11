import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence, Union, Optional


# ---------------------------------------------------------------------------- #
# Layers
# ---------------------------------------------------------------------------- #
def make_conv1x1_gn(in_channels, conv_channels: Sequence[int], last_channels: Optional[int] = None,
                    num_groups: Union[int, Sequence[int]] = 8, act_builder=nn.ReLU):
    c_in = in_channels
    module_list = []
    if isinstance(num_groups, (list, tuple)):
        assert len(num_groups) == len(conv_channels)
    else:
        num_groups = [num_groups] * len(conv_channels)

    for idx, c_out in enumerate(conv_channels):
        module_list.append(nn.Conv2d(c_in, c_out, 1, bias=False))
        module_list.append(nn.GroupNorm(num_groups[idx], c_out))
        module_list.append(act_builder())
        c_in = c_out

    if last_channels is not None:
        module_list.append(nn.Conv2d(c_in, last_channels, 1, bias=True))

    return nn.Sequential(*module_list)


def make_conv1x1_bn(in_channels, conv_channels: Sequence[int], last_channels: Optional[int] = None,
                    act_builder=nn.ReLU):
    c_in = in_channels
    module_list = []

    for idx, c_out in enumerate(conv_channels):
        module_list.append(nn.Conv2d(c_in, c_out, 1, bias=False))
        module_list.append(nn.BatchNorm2d(c_out))
        module_list.append(act_builder())
        c_in = c_out

    if last_channels is not None:
        module_list.append(nn.Conv2d(c_in, last_channels, 1, bias=True))

    return nn.Sequential(*module_list)


def make_mlp_gn(in_channels, mlp_channels: Sequence[int], last_channels: Optional[int] = None,
                num_groups: Union[int, Sequence[int]] = 8, act_builder=nn.ReLU):
    c_in = in_channels
    module_list = []
    if isinstance(num_groups, (list, tuple)):
        assert len(num_groups) == len(mlp_channels)
    else:
        num_groups = [num_groups] * len(mlp_channels)

    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out, bias=False))
        module_list.append(nn.GroupNorm(num_groups[idx], c_out))
        module_list.append(act_builder())
        c_in = c_out

    if last_channels is not None:
        module_list.append(nn.Linear(c_in, last_channels, bias=True))

    return nn.Sequential(*module_list)


def make_mlp_bn(in_channels, mlp_channels: Sequence[int], last_channels: Optional[int] = None,
                act_builder=nn.ReLU):
    c_in = in_channels
    module_list = []

    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out, bias=False))
        module_list.append(nn.BatchNorm1d(c_out))
        module_list.append(act_builder())
        c_in = c_out

    if last_channels is not None:
        module_list.append(nn.Linear(c_in, last_channels, bias=True))

    return nn.Sequential(*module_list)


class SpatialBroadcast(nn.Module):
    def __init__(self, image_shape, stride=1):
        super(SpatialBroadcast, self).__init__()
        self.image_shape = image_shape
        self.stride = stride
        self.grid_shape = (int(image_shape[0] / stride), int(image_shape[1] / stride))

        identity = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=torch.float32)
        # (1, h, w, 2)
        grid = F.affine_grid(identity, [1, 1, self.grid_shape[0], self.grid_shape[1]])
        grid = grid.permute(0, 3, 1, 2).contiguous()
        self.register_buffer('grid', grid)

    def forward(self, x):
        x_reshape = x.reshape(x.size(0), x.size(1), 1, 1).expand(-1, -1, self.grid_shape[0], self.grid_shape[1])
        grid_reshape = self.grid.expand(x.size(0), -1, -1, -1)
        return torch.cat([x_reshape, grid_reshape], dim=1)


class Flatten(nn.Module):
    r"""
    Flattens a contiguous range of dims into a tensor. For use with :class:`~nn.Sequential`.
    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     nn.Flatten()
        >>> )
    """
    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
