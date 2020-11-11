import numpy as np
import cv2
import torch
import warnings
from space.datasets.falling_digit import FallingDigit as BaseFallingDigit


class FallingDigit(BaseFallingDigit):
    def __getitem__(self, index):
        data = self.data[index]
        if 'image' in data:
            image = data['image']
        else:
            image = data['original_image']
        assert image.dtype == np.uint8

        # resize if necessary
        if image.shape[:2] != self.image_size:
            warnings.warn('Resize image from {} to {}'.format(image.shape, self.image_size))
            image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_AREA)

        if self.transform is not None:
            image_pil = self.transform(image)
            image = np.asarray(image_pil)

        assert image.dtype == np.uint8
        image = np.asarray(image, dtype=np.float32) / 255.

        if 'action' in data:
            action = data['action']  # scalar
        else:
            action = data['q'].argmax()

        if self.to_tensor:
            image = np.transpose(image, [2, 0, 1])  # (c, h, w)
            image = torch.tensor(image, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)

        out_dict = {
            'image': image,
            'action': action,
        }
        return out_dict


def test():
    import os.path as osp
    from space.utils.plt_utils import show_image

    _ROOT_DIR = osp.join(osp.dirname(__file__), '../..')
    path = osp.join(_ROOT_DIR, 'data/falling_digit/FallingDigit_3-v0_n_72000_lv_0_to_3000_from_gt_policy.pkl')
    dataset = FallingDigit(path, to_tensor=False)

    for i in range(0, 10):
        data = dataset[i]
        image = data['image']
        show_image(image)
