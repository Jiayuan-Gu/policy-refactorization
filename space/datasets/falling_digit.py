import pickle
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
import warnings


class FallingDigit(Dataset):
    def __init__(self, path, start=0, end=-1, image_size=(128, 128), transform=None, to_tensor=True):
        self.path = path
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.data[start:(None if end == -1 else end)]

        self.image_size = image_size
        assert isinstance(image_size, tuple)
        self.transform = transform  # data augmentation
        self.to_tensor = to_tensor

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

        if self.to_tensor:
            image = np.transpose(image, [2, 0, 1])  # (c, h, w)
            image = torch.tensor(image, dtype=torch.float32)

        out_dict = {
            'image': image,
        }
        return out_dict

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return '{:s}: {:d} images.'.format(self.__class__.__name__, len(self))


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
