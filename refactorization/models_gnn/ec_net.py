import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn import EdgeConv

from refactorization.models_gnn.base import BaseGNN


class EdgeConvNet(BaseGNN):
    def __init__(self, global_aggr='max', output_dim=3):
        super(BaseGNN, self).__init__()

        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=True), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.GroupNorm(4, 64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [2, 2]
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # [1, 1]
        )

        local_nn = nn.Sequential(
            nn.Linear((128 + 4) * 2, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
        )
        self.gnn = EdgeConv(local_nn, aggr='max')

        self.encoder2 = nn.Sequential(
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False), nn.GroupNorm(8, 128), nn.ReLU(),
        )

        self.global_aggr = global_aggr

        self.fc = nn.Sequential(
            nn.Linear(128, 128, bias=True), nn.ReLU(),
            nn.Linear(128, 128, bias=True), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.reset_parameters()

    def forward(self, data, batch_size=None, **kwargs):
        x = data.x
        batch = data.batch
        edge_index = data.edge_index
        pos = data.pos

        # infer real batch size, in case empty sample
        if batch_size is None:
            batch_size = data['size'].sum().item()

        img_feature = self.encoder(x).flatten(1)
        x = torch.cat([img_feature, pos], dim=1)

        x = self.gnn(x=x, edge_index=edge_index)
        x = self.encoder2(x)

        if self.global_aggr == 'max':
            global_feature = gnn.global_max_pool(x, batch, size=batch_size)
        elif self.global_aggr == 'sum':
            global_feature = gnn.global_add_pool(x, batch, size=batch_size)
        else:
            raise NotImplementedError()

        logits = self.fc(global_feature)

        out_dict = {
            'logits': logits,
        }
        return out_dict


def test():
    from common.utils.misc import print_dict
    from torch_geometric.data import Data, Batch

    model = EdgeConvNet()
    print(model)
    data = Data(
        x=torch.rand(4, 3, 16, 16),
        action=torch.randint(3, [1]),
        pos=torch.rand(4, 4),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.int64),
        size=torch.tensor([1], dtype=torch.int64),
    )
    data_batch = Batch.from_data_list([data])
    pd_dict = model(data_batch)
    print_dict(pd_dict)
    loss_dict = model.compute_losses(pd_dict, data_batch)
    print_dict(loss_dict)
