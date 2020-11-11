import torch
import torch.nn.functional as F

from typing import Tuple, Optional


def decode_boxes(anchors, delta, clip_delta=5.0, image_shape: Optional[Tuple[int, int]] = None):
    """Decode boxes according to z_where.

    Notes:
        Make use of broadcast to deal with batch dimension.

    References: https://github.com/pytorch/vision/blob/master/torchvision/models/detection/_utils.py

    Args:
        anchors (torch.Tensor): (..., 4). [x_min, y_min, x_max, y_max]
        delta (torch.Tensor): (..., 4)
        clip_delta (float): clip dw and dh
        image_shape (optional, tuple): [H, W]

    Returns:
        boxes (torch.Tensor): (..., 4). [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, x_max, y_max = torch.split(anchors, 1, dim=-1)
    ctr_x = 0.5 * (x_min + x_max)
    ctr_y = 0.5 * (y_min + y_max)
    w = x_max - x_min
    h = y_max - y_min

    # decode z_where
    dx, dy, dw, dh = torch.split(delta, 1, dim=-1)
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, min=-clip_delta, max=clip_delta)
    dh = torch.clamp(dh, min=-clip_delta, max=clip_delta)

    pred_x_ctr = ctr_x + dx * w
    pred_y_ctr = ctr_y + dy * h
    pred_w = torch.exp(dw) * w
    pred_h = torch.exp(dh) * h

    # decode new boxes
    x_min = (pred_x_ctr - 0.5 * pred_w)
    x_max = (pred_x_ctr + 0.5 * pred_w)
    y_min = (pred_y_ctr - 0.5 * pred_h)
    y_max = (pred_y_ctr + 0.5 * pred_h)

    # clip to image
    if image_shape is not None:
        x_min = x_min.clamp(min=0.0)
        x_max = x_max.clamp(max=image_shape[1])
        y_min = y_min.clamp(min=0.0)
        y_max = y_max.clamp(max=image_shape[0])

    return torch.cat([x_min, y_min, x_max, y_max], dim=-1)


def boxes_xyxy2xywh(boxes):
    """Convert boxes from xyxy format to xywh format."""
    x_min, y_min, x_max, y_max = torch.split(boxes, 1, dim=-1)
    ctr_x = 0.5 * (x_min + x_max)
    ctr_y = 0.5 * (y_min + y_max)
    w = x_max - x_min
    h = y_max - y_min
    return torch.cat([ctr_x, ctr_y, w, h], dim=-1)


def image_to_glimpse(image, boxes, glimpse_shape):
    """Crop glimpses from images

    Args:
        image (torch.Tensor): (b, c0, h0, w0)
        boxes (torch.Tensor): (b, N, 4).
            Assume already normalized to [0, 1]. xywh format
        glimpse_shape (tuple): (h2, w2)

    Returns:
        glimpses (torch.Tensor): (b * N, c0, h0, w0)

    """
    b, c0, h0, w0 = image.size()
    N = boxes.size(1)

    # reshape
    # (b * N, c0, h0, w0)
    image_reshape = image.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(-1, c0, h0, w0)
    # (b * N, 4)
    boxes = boxes.reshape(-1, 4)

    # generate affine transformation
    zeros = boxes.new_zeros([boxes.size(0)])
    scale_x = boxes[:, 2]
    scale_y = boxes[:, 3]
    shift_x = 2 * boxes[:, 0] - 1
    shift_y = 2 * boxes[:, 1] - 1
    affine_transform = torch.stack([scale_x, zeros, shift_x, zeros, scale_y, shift_y], dim=1)
    affine_transform = affine_transform.reshape(boxes.size(0), 2, 3)  # (b * N, 2, 3)
    grid = F.affine_grid(affine_transform, [boxes.size(0), c0, glimpse_shape[0], glimpse_shape[1]])
    glimpses = F.grid_sample(image_reshape, grid)
    return glimpses


def glimpse_to_image(glimpses, boxes, image_shape, eps=1e-6):
    """Revert glimpses to images

    Args:
        glimpses (torch.Tensor): (b * N, c0, h2, w2)
        boxes (torch.Tensor): (b, N, 4).
            Assume already normalized to [0, 1]. xywh format
        image_shape (tuple): (h0, w0)
        eps (float): prevent zero-division for small bounding boxes

    Returns:
        image (torch.Tensor): (b * N, c0, h0, w0)

    """
    boxes = boxes.reshape(-1, 4)  # (b * N, 4)
    zeros = boxes.new_zeros([boxes.size(0)])
    scale_x = 1.0 / boxes[:, 2].clamp(min=eps)
    scale_y = 1.0 / boxes[:, 3].clamp(min=eps)
    shift_x = - (2 * boxes[:, 0] - 1) * scale_x
    shift_y = - (2 * boxes[:, 1] - 1) * scale_y
    affine_transform = torch.stack([scale_x, zeros, shift_x, zeros, scale_y, shift_y], dim=1)
    affine_transform = affine_transform.reshape(boxes.size(0), 2, 3)
    grid = F.affine_grid(affine_transform, [boxes.size(0), glimpses.size(1), image_shape[0], image_shape[1]])
    image = F.grid_sample(glimpses, grid)
    return image
