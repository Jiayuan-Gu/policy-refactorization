import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch_geometric.data import Data
from torch.distributions import Normal
from space.utils.box_utils import image_to_glimpse, boxes_xyxy2xywh


class FallingDigitPatch(Dataset):
    """Provide patches according to GT boxes or proposals."""

    def __init__(self, data_path, proposals_path, start=0, end=-1,
                 size=(16, 16), fixed_crop=False, std=None,
                 ):
        self.data_path = data_path
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            self.data = self.data[start:(None if end == -1 else end)]

        self.proposals_path = proposals_path
        with open(proposals_path, 'rb') as f:
            self.proposals = pickle.load(f)
            self.proposals = self.proposals[start:(None if end == -1 else end)]

        self.size = size
        self.fixed_crop = fixed_crop
        self.std = std

    def __getitem__(self, index):
        data = self.data[index]

        if 'image' in data:
            image = data['image'].copy()
        elif 'original_image' in data:
            image = data['original_image'].copy()
        else:
            raise RuntimeError(data.keys())
        image = torch.tensor(image.transpose([2, 0, 1]) / 255.0, dtype=torch.float32)

        boxes_xyxy = np.array(self.proposals[index]['boxes'], dtype=np.float32)

        # # visualization
        # import matplotlib.pyplot as plt
        # from space.utils.plt_utils import draw_boxes
        # draw_boxes(data['image'], boxes_xyxy)
        # plt.show()

        if len(boxes_xyxy) == 0:
            patches = torch.empty((0, 3) + self.size, dtype=torch.float32)
            boxes_xywh_tensor = torch.empty((0, 4), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.int64)
            # print(f'Data point {index} has empty detection.')
        else:
            # Normalize boxes
            boxes_xyxy_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            boxes_xyxy_tensor[:, [0, 2]] /= image.shape[2]
            boxes_xyxy_tensor[:, [1, 3]] /= image.shape[1]
            boxes_xywh_tensor = boxes_xyxy2xywh(boxes_xyxy_tensor)

            if self.fixed_crop:
                boxes_xywh_tensor[:, 2] = self.size[1] / image.shape[2]
                boxes_xywh_tensor[:, 3] = self.size[0] / image.shape[1]

            # add augmentation
            if self.std:
                std_tensor = boxes_xywh_tensor.new_tensor(self.std)
                boxes_xywh_tensor = Normal(boxes_xywh_tensor, std_tensor).sample()
            patches = image_to_glimpse(image.unsqueeze(0), boxes_xywh_tensor.unsqueeze(0), self.size)

            n = boxes_xywh_tensor.size(0)
            edge_index = torch.tensor([[i, j] for i in range(n) for j in range(n)], dtype=torch.int64).transpose(0, 1)

        # get target
        if 'action' in data:
            action = data['action']  # scalar
        else:
            action = data['q'].argmax()

        out = Data(
            x=patches,
            action=torch.tensor([action], dtype=torch.int64),
            edge_index=edge_index.long(),
            pos=boxes_xywh_tensor.float(),
            idx=torch.tensor([index], dtype=torch.int64),  # for visualization and dp
            size=torch.tensor([1], dtype=torch.int64),  # indicate batch size
        )
        return out

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return '{:s}: {:d} images.'.format(self.__class__.__name__, len(self))


def test():
    import os.path as osp
    import matplotlib.pyplot as plt
    from space.utils.plt_utils import draw_images

    _ROOT_DIR = osp.join(osp.dirname(__file__), '../..')
    data_path = osp.join(_ROOT_DIR,
                         'data/falling_digit/FallingDigit_3_digit_inst_1-v0_n_72000_lv_0_to_3000_from_gt_policy.pkl')
    proposals_path = osp.join(_ROOT_DIR,
                              'outputs/falling_digit/black_space_v0_fg_0.15_bg_0.1/1/proposals_FallingDigit_3_digit_inst_1-v0_n_72000_lv_0_to_3000_from_gt_policy.pkl')
    dataset = FallingDigitPatch(data_path, proposals_path)

    for i in range(0, 10):
        data = dataset[i]
        print(data)
        # patches = data.x  # [N, C, H, W]
        # patches = patches.cpu().numpy()
        # patches = np.transpose(patches, (0, 2, 3, 1))
        # draw_images(patches)
        # plt.show()
