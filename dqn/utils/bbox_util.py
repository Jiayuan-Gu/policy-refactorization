import numpy as np


def iou_2d(bbox1, bbox2):
    """Only support axis-aligned xywh style"""
    assert bbox1.shape[0] == bbox2.shape[0] == 4
    inner = np.minimum(bbox1[:2] + 0.5 * bbox1[2:], bbox2[:2] + 0.5 * bbox2[2:]) - \
            np.maximum(bbox1[:2] - 0.5 * bbox1[2:], bbox2[:2] - 0.5 * bbox2[2:])
    inner = np.maximum(inner, 0.0)
    intersection = np.prod(inner)
    union = np.prod(bbox1[2:]) + np.prod(bbox2[2:]) - intersection
    return intersection / union


def nms_2d(bboxes, scores, iou_thresh):
    if len(bboxes) == 0:
        return bboxes
    sorted_indices = np.argsort(-scores)
    bboxes = bboxes[sorted_indices]
    nms_indices = [0]
    for i in np.arange(1, len(bboxes)):
        bbox = bboxes[i]
        flag = True
        for j in nms_indices:
            if iou_2d(bbox, bboxes[j]) >= iou_thresh:
                flag = False
                break
        if flag:
            nms_indices.append(i)
    # remapping
    nms_indices = sorted_indices[nms_indices]
    return nms_indices


def generate_sliding_windows(window_shape, window_strides, image_shape):
    """Generate sliding windows

    Args:
        window_shape (2-tuple): (h, w)
        window_strides (2-tuple): (sy, sx)
        image_shape (2-tuple): (H, W)

    Returns:
        boxes (list of list): (N, 4), where N = (H-h+1)//sy * (W-w+1)/sx;
            boxes in image
        indices (list of list): (N, h*w)
            flatten coordinates in image
        num_cells (tuple): number of cells in y-axis and x-axis

    """
    h, w = window_shape
    sy, sx = window_strides
    H, W = image_shape

    ys_start = list(range(0, H - h + 1, sy))
    xs_start = list(range(0, W - w + 1, sx))

    boxes = []
    indices = []
    num_cells = (len(ys_start), len(xs_start))
    for y_start in ys_start:
        for x_start in xs_start:
            index = []
            for y in range(y_start, y_start + h):
                for x in range(x_start, x_start + w):
                    index.append(y * W + x)
            boxes.append([x_start, y_start, w, h])
            indices.append(index)
    return boxes, indices, num_cells
