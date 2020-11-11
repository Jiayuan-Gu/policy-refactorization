import math
import numpy as np
import matplotlib.pyplot as plt


def show_image(image):
    plt.cla()
    # plt.axis("off")
    if image.ndim == 2:
        plt.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(image)
    plt.show()


def show_boxes(image, boxes, boxes2=None, scores=None, show_axis=False):
    plt.cla()
    if not show_axis:
        plt.axis("off")

    if image.ndim == 2:
        plt.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    elif image.shape[2] == 1:
        plt.imshow(image.squeeze(-1), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(image)

    for i, bbox in enumerate(boxes):
        # (x_min, y_min, w, h)
        rect = plt.Rectangle((bbox[0] - 0.5, bbox[1] - 0.5), bbox[2], bbox[3],
                             fill=False, edgecolor='red', linewidth=1.0)
        plt.gca().add_patch(rect)
        if scores is not None:
            plt.gca().text(bbox[0] - 0.5, bbox[1] - 0.5, str(scores[i]), color='red')

    # additional boxes, maybe GT
    if boxes2 is not None:
        for bbox2 in boxes2:
            rect2 = plt.Rectangle((bbox2[0] - 0.5, bbox2[1] - 0.5), bbox2[2], bbox2[3],
                                  fill=False, edgecolor='green', linewidth=1.0)
            plt.gca().add_patch(rect2)

    plt.show()


def draw_boxes(image, boxes, scores=None):
    """Draw bounding boxes on a gray image. Usually for tensorboard.

    Args:
        image (np.ndarray): (H, W, C)
        boxes (np.ndarray): (N, 4), represented as [x_min, y_min, w, h]
        scores (np.ndarray or list, optional): (N,)

    Returns:
        fig (plt.figure.Figure)

    """
    plt.close('all')
    fig = plt.figure()
    plt.axis("off")
    if image.ndim == 2:
        plt.imshow(image, vmin=0.0, vmax=1.0)
    elif image.shape[2] == 1:
        plt.imshow(image.squeeze(-1), cmap="gray", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(image)
    for i, bbox in enumerate(boxes):
        rect = plt.Rectangle((bbox[0] - 0.5, bbox[1] - 0.5), bbox[2], bbox[3],
                             fill=False, edgecolor='red', linewidth=1.0)
        plt.gca().add_patch(rect)
        if scores is not None:
            plt.gca().text(bbox[0] - 0.5, bbox[1] - 0.5, '{:.2f}'.format(scores[i]), color='red')
    plt.tight_layout(0.02)
    return fig


def draw_images(images):
    """Draw gray images in a close-to-square layout

    Args:
        images (np.ndarray): (N, H, W) or (N, H, W, C)

    Returns:
        fig (plt.figure.Figure)

    """
    N, H, W = images.shape[:3]
    num_rows = int(math.sqrt(N))
    num_cols = (N + num_rows - 1) // num_rows

    plt.close('all')
    fig = plt.figure()
    for i in range(N):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        image = images[i]
        if image.ndim == 2:
            ax.imshow(image, vmin=0.0, vmax=1.0)
        elif image.shape[2] == 1:
            ax.imshow(image.squeeze(-1), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(image)
        ax.set_axis_off()
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)
    return fig
