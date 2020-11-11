import torch
import torch.nn.functional as F

from typing import Sequence, Union


def generate_anchors(scales: Sequence[Union[int, float]],
                     aspect_ratios: Sequence[float],
                     dtype=torch.float32,
                     device="cpu"):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/detection/anchor_utils.py"""
    scales = torch.as_tensor(scales, dtype=dtype, device=device)
    aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
    h_ratios = torch.sqrt(aspect_ratios)
    w_ratios = 1 / h_ratios

    ws = (w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h_ratios[:, None] * scales[None, :]).view(-1)

    # (num_scales * num_ratios, 4)
    base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
    return base_anchors.round()


def grid_anchors(base_anchors: torch.Tensor,
                 grid_size: Sequence[int],
                 stride: Sequence[int],
                 center=True):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/detection/anchor_utils.py"""
    grid_height, grid_width = grid_size
    stride_height, stride_width = stride
    device = base_anchors.device
    shifts_x = torch.arange(
        0, grid_width, dtype=torch.float32, device=device
    ) * stride_width
    shifts_y = torch.arange(
        0, grid_height, dtype=torch.float32, device=device
    ) * stride_height
    if center:
        shifts_x = shifts_x + 0.5 * stride_width
        shifts_y = shifts_y + 0.5 * stride_height
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)  # (grid_height * grid_width,)
    shift_y = shift_y.reshape(-1)  # (grid_height * grid_width,)
    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # (grid_height * grid_width, 4)
    # each item is (num_anchors_per_cell * grid_height * grid_width, 4)
    anchors = (shifts.view(1, -1, 4) + base_anchors.view(-1, 1, 4)).reshape(-1, 4)
    return anchors
