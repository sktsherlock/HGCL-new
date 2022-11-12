import torch
import torch as th
import os.path as osp
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.nn import GINConv, global_add_pool
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import SumPooling
from utils import local_global_loss_
from torch.nn import Sequential, Linear, ReLU


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GIN(nn.Module):
    """
    init: input_dim, hidden_dim, num_layers
    forward: x, edge_index, batch
    """

    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

    def get_embeddings(self, device, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                # data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                _, x = self.forward(x, edge_index, batch)
                ret.append(x)
                y.append(data.y)
            ret = torch.cat(ret, dim=0)
            y = torch.cat(y, dim=0)

        return ret, y


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.PReLU(),
            nn.Linear(out_dim, out_dim),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fcs(x) + self.linear_shortcut(x)


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers, norm):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(GraphConv(in_dim, out_dim, bias=False, norm=norm, activation=nn.PReLU()))
        self.pooling = SumPooling()

        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(out_dim, out_dim, bias=False, norm=norm, activation=nn.PReLU()))

    def forward(self, graph, feat, edge_weight=None):
        h = self.layers[0](graph, feat, edge_weight=edge_weight)
        hg = self.pooling(graph, h)

        for idx in range(self.num_layers - 1):
            h = self.layers[idx + 1](graph, h, edge_weight=edge_weight)
            hg = th.cat((hg, self.pooling(graph, h)), -1)

        return h, hg


class MVGRL(nn.Module):
    r"""
        mvgrl model
    Parameters
    -----------
    in_dim: int
        Input feature size.
    out_dim: int
        Output feature size.
    num_layers: int
        Number of the GNN encoder layers.
    Functions
    -----------
    forward(graph1, graph2, feat, edge_weight):
        graph1: DGLGraph
            The original graph
        graph2: DGLGraph
            The diffusion graph
        feat: tensor
            Node features
        edge_weight: tensor
            Edge weight of the diffusion graph
    """

    def __init__(self, in_dim, out_dim, num_layers):
        super(MVGRL, self).__init__()
        self.local_mlp = MLP(out_dim, out_dim)
        self.global_mlp = MLP(num_layers * out_dim, out_dim)
        self.encoder1 = GCN(in_dim, out_dim, num_layers, norm='both')
        self.encoder2 = GCN(in_dim, out_dim, num_layers, norm='none')

    def get_embedding(self, graph1, graph2, feat, edge_weight):
        local_v1, global_v1 = self.encoder1(graph1, feat)
        local_v2, global_v2 = self.encoder2(graph2, feat, edge_weight=edge_weight)

        global_v1 = self.global_mlp(global_v1)
        global_v2 = self.global_mlp(global_v2)

        return (global_v1 + global_v2).detach()

    def forward(self, graph1, graph2, feat, edge_weight, graph_id):
        # calculate node embeddings and graph embeddings
        local_v1, global_v1 = self.encoder1(graph1, feat)
        local_v2, global_v2 = self.encoder2(graph2, feat, edge_weight=edge_weight)

        local_v1 = self.local_mlp(local_v1)
        local_v2 = self.local_mlp(local_v2)

        global_v1 = self.global_mlp(global_v1)
        global_v2 = self.global_mlp(global_v2)

        # calculate loss
        loss1 = local_global_loss_(local_v1, global_v2, graph_id)
        loss2 = local_global_loss_(local_v2, global_v1, graph_id)

        loss = loss1 + loss2

        return loss


class HGCL(nn.Module):
    def __init__(self, out_dim, encoder, augmentor):
        super(HGCL, self).__init__()
        self.local_mlp = MLP(out_dim, out_dim)
        self.global_mlp = MLP(out_dim, out_dim)
        self.augmentor = augmentor
        self.encoder1 = encoder.conv1
        self.encoder2 = encoder.conv2
        self.encoder3 = encoder.conv3

    def forward(self, x1, x2, x3, edge_index1, edge_index2, edge_index3, batch1, batch2, batch3):
        aug1, aug2 = self.augmentor

        x1_1, edge_index1_1, edge_weight1_1 = aug1(x1, edge_index1)
        x1_2, edge_index1_2, edge_weight1_2 = aug2(x1, edge_index1)
        z1_1, g1_1 = sefl.encoder1(x1_1, edge_index1_1, edge_weight1_1, batch1)
        z1_2, g1_2 = sefl.encoder1(x1_2, edge_index1_2, edge_weight1_2, batch1)
        z1_1 = self.local_mlp(z1_1)
        z1_2 = self.local_mlp(z1_2)
        g1_1 = self.global_mlp(g1_1)
        g1_2 = self.global_mlp(g1_2)
        loss1 = local_global_loss_(z1_1, g1_2, batch1) + local_global_loss_(z1_2, g1_1, batch1)

        x2_1, edge_index2_1, edge_weight2_1 = aug1(x2, edge_index2)
        x2_2, edge_index2_2, edge_weight2_2 = aug2(x2, edge_index2)
        z2_1, g2_1 = sefl.encoder2(x2_1, edge_index2_1, edge_weight2_1, batch2)
        z2_2, g2_2 = sefl.encoder2(x2_2, edge_index2_2, edge_weight2_2, batch2)
        z2_1 = self.local_mlp(z2_1)
        z2_2 = self.local_mlp(z2_2)
        g2_1 = self.global_mlp(g2_1)
        g2_2 = self.global_mlp(g2_2)
        loss2 = local_global_loss_(z2_1, g2_2, batch2) + local_global_loss_(z2_2, g2_1, batch2)

        x3_1, edge_index3_1, edge_weight3_1 = aug1(x3, edge_index3)
        x3_2, edge_index3_2, edge_weight3_2 = aug2(x3, edge_index3)
        z3_1, g3_1 = sefl.encoder3(x3_1, edge_index3_1, edge_weight3_1, batch3)
        z3_2, g3_2 = sefl.encoder3(x3_2, edge_index3_2, edge_weight3_2, batch3)
        z3_1 = self.local_mlp(z3_1)
        z3_2 = self.local_mlp(z3_2)
        g3_1 = self.global_mlp(g3_1)
        g3_2 = self.global_mlp(g3_2)
        loss3 = local_global_loss_(z3_1, g3_2, batch3) + local_global_loss_(z3_2, g3_1, batch3)

        loss = loss1 + loss2 + loss3

        return loss
