import torch
import torch.nn as nn
from refactorization.models_cnn.base import BaseCNN


class PlainCNN(BaseCNN):
    """Plain CNN (for 128x128 images)"""

    def __init__(self,
                 in_channels=3,
                 output_dim=3,
                 max_pooling=True,
                 ):
        super().__init__()

        self.output_dim = output_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [64, 64]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(64, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(128, 128, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(128, 128, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        if max_pooling:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Linear(128, 256, bias=True), nn.ReLU(inplace=True),
                nn.Linear(256, output_dim)
            )
        else:
            self.pool = None
            self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256, bias=True), nn.ReLU(inplace=True),
                nn.Linear(256, output_dim)
            )

        self.reset_parameters()

    def forward(self, data_batch: dict) -> dict:
        image = data_batch['image']
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        logits = self.fc(x)

        pd_dict = {
            'logits': logits,
        }
        return pd_dict


def test():
    from common.utils.misc import print_dict

    model = PlainCNN()
    print(model)
    data_batch = {
        'image': torch.rand(4, 3, 128, 128),
        'action': torch.randint(3, [4])
    }
    pd_dict = model(data_batch)
    print_dict(pd_dict)
    loss_dict = model.compute_losses(pd_dict, data_batch)
    print_dict(loss_dict)
