import torch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import ASAPooling
from MLP import MLP
from torch import nn
from models.Encoder import GIN
from models.Gaussian import DiagonalGaussianDistribution
import GCL.augmentors as A


class Config():
    def __init__(self):
        self.hidden = 128
        self.pooling_ratio = 0.8
        self.dropout = 0.5
        self.add_to_edge = 0.5


def add_noise(x, noise, tradeoff):
    if tradeoff > 0:
        if tradeoff <= 1:
            x = (1 - tradeoff) * x + noise * tradeoff
        else:
            raise ValueError('tradeoff <= 1')
    return x


def innercl(proj_1, proj_2):
    from losses.infonce import infonce
    return infonce(proj_1, proj_2)


class ASAPNet(torch.nn.Module):
    def __init__(self, args, augmentor):
        super(ASAPNet, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.hidden = args.hidden
        self.pooling_ratio = args.pooling_ratio
        self.dropout = 0.5
        self.augmentor = augmentor

        if args.conv in ['GIN']:
            self.conv1 = GINConv(nn.Sequential(nn.Linear(self.num_features, self.hidden), nn.ReLU(),
                                               nn.Linear(self.hidden, self.hidden)))
            self.conv2 = GINConv(
                nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden)))
            self.conv3 = GINConv(
                nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden)))
        elif args.conv in ['SAGE']:
            self.conv1 = SAGEConv(self.num_features, self.hidden)
            self.conv2 = SAGEConv(self.hidden, self.hidden)
            self.conv3 = SAGEConv(self.hidden, self.hidden)
        elif args.conv in ['GCN']:
            self.conv1 = GCNConv(self.num_features, self.hidden)
            self.conv2 = GCNConv(self.hidden, self.hidden)
            self.conv3 = GCNConv(self.hidden, self.hidden)
        else:
            raise NotImplementedError
        self.add_noise = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=256)

        self.projection_head_1 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)
        self.projection_head_2 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)
        self.projection_head_3 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)

        self.pool1 = ASAPooling(self.hidden, ratio=self.pooling_ratio)
        self.pool2 = ASAPooling(self.hidden, ratio=self.pooling_ratio)

        # For classification
        self.linear1 = torch.nn.Linear(self.hidden * 2, self.hidden)
        self.linear2 = torch.nn.Linear(self.hidden, self.hidden // 2)
        self.linear3 = torch.nn.Linear(self.hidden // 2, self.num_classes)

    def forward(self, data, tradeoff):
        aug1, aug2 = self.augmentor
        x_0, edge_index_0, batch_0 = data.x, data.edge_index, data.batch
        edge_attr_0 = None

        # InnerCL augmentations of the firsrt Graph
        x_0_1, edge_index_0_1, _ = aug1(x_0, edge_index_0)
        x_0_2, edge_index_0_2, _ = aug2(x_0, edge_index_0)
        g0_1 = global_add_pool(self.projection_head_1(self.conv1(x_0_1, edge_index_0_1)), batch_0)
        g0_2 = global_add_pool(self.projection_head_1(self.conv1(x_0_2, edge_index_0_2)), batch_0)

        x = F.relu(x)
        g1 = torch.cat([gmp(x, batch_0), gap(x, batch_0)], dim=1)

        local_proj_1 = self.projection_head_1(x)
        proj_1 = global_add_pool(local_proj_1, batch_0)
        # proj_1 = torch.cat([gmp(local_proj_1, batch_0), gap(local_proj_1, batch_0)], dim=1)

        x_1, edge_index_1, edge_attr_1, batch_1, _, = self.pool1(x, edge_index_0, edge_attr_0, batch_0)
        # g1 = torch.cat([gmp(x_1, batch_1), gap(x_1, batch_1)], dim=1)
        x_1 = self.conv2(x_1, edge_index_1)

        noise_2 = self.add_noise(x_1)
        posterior_2 = DiagonalGaussianDistribution(noise_2)

        x_1_1 = add_noise(x_1, posterior_2.sample(), tradeoff)
        x_1_2 = add_noise(x_1, posterior_2.sample(), tradeoff)

        # InnerCL augmentations of the second Graph

        g1_1 = global_add_pool(self.projection_head_1(x_1_1), batch_1)
        g1_2 = global_add_pool(self.projection_head_1(x_1_2), batch_1)

        x_1 = F.relu(x_1)
        g2 = torch.cat([gmp(x_1, batch_1), gap(x_1, batch_1)], dim=1)
        local_proj_2 = self.projection_head_2(x_1)
        proj_2 = global_add_pool(local_proj_2, batch_1)

        x = g1 + g2

        return x, proj_1, proj_2, g0_1, g0_2, g1_1, g1_2

    def get_embeddings(self, device, loader, tradeoff=5e-4):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                if data.x is None:
                    data.x = torch.ones(data.num_nodes, 1).to(device)
                data.to(device)
                x, *_ = self(data, tradeoff)
                ret.append(x)
                y.append(data.y)
            ret = torch.cat(ret, dim=0)
            y = torch.cat(y, dim=0)

        return ret, y



