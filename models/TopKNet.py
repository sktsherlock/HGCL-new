import torch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling, SAGPooling, EdgePooling, ASAPooling
from MLP import MLP
from torch import nn
from models.Encoder import GIN
import GCL.augmentors as A

class TopKNet(torch.nn.Module):
    def __init__(self, args, augmentor):
        super(TopKNet, self).__init__()
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        self.hidden = args.hidden
        self.pooling_ratio = args.pooling_ratio
        self.augmentor = augmentor

        if args.conv in ['GIN']:
            self.conv1 = GINConv(nn.Sequential(nn.Linear(self.num_features, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden)))
            self.conv2 = GINConv(nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden)))
            self.conv3 = GINConv(nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.hidden)))
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

        self.projection_head_1 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)
        self.projection_head_2 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)
        self.projection_head_3 = MLP(in_channels=self.hidden, hidden_channels=self.hidden, out_channels=128)

        self.pool1 = TopKPooling(self.hidden, ratio=self.pooling_ratio)
        self.pool2 = TopKPooling(self.hidden, ratio=self.pooling_ratio)


    def forward(self, data):
        aug1, aug2 = self.augmentor
        x_0, edge_index_0, batch_0 = data.x, data.edge_index, data.batch
        edge_attr_0 = None

        # InnerCL augmentations of the firsrt Graph 
        x_0_1, edge_index_0_1, _ = aug1(x_0, edge_index_0)
        x_0_2, edge_index_0_2, _ = aug2(x_0, edge_index_0)
        g0_1 = global_add_pool(self.projection_head_1(self.conv1(x_0_1, edge_index_0_1)),batch_0)
        g0_2 = global_add_pool(self.projection_head_1(self.conv1(x_0_2, edge_index_0_2)),batch_0)

        x = F.relu(self.conv1(x_0, edge_index_0))
        x_first = x 
        g1 = torch.cat([gmp(x_first, batch_0), gap(x_first, batch_0)], dim=1)

        local_proj_1 = self.projection_head_1(x)
        proj_1 = global_add_pool(local_proj_1, batch_0)
        #proj_1 = torch.cat([gmp(local_proj_1, batch_0), gap(local_proj_1, batch_0)], dim=1)

        x_1, edge_index_1, edge_attr_1, batch_1, _, _  = self.pool1(x, edge_index_0, edge_attr_0, batch_0)
        #g1 = torch.cat([gmp(x_1, batch_1), gap(x_1, batch_1)], dim=1)
        

        # InnerCL augmentations of the second Graph
        x_1_1, edge_index_1_1, _ = aug1(x_1, edge_index_1)
        x_1_2, edge_index_1_2, _ = aug2(x_1, edge_index_1)
        g1_1 = global_add_pool(self.projection_head_2(self.conv2(x_1_1, edge_index_1_1)),batch_1)
        g1_2 = global_add_pool(self.projection_head_2(self.conv2(x_1_2, edge_index_1_2)),batch_1)

        x = F.relu(self.conv2(x_1, edge_index_1))
        x_second = x 
        g2 = torch.cat([gmp(x_second, batch_1), gap(x_second, batch_1)], dim=1)
        local_proj_2 = self.projection_head_2(x)
        proj_2 = global_add_pool(local_proj_2, batch_1)
        #proj_2 = torch.cat([gmp(local_proj_2, batch_1), gap(local_proj_2, batch_1)], dim=1)
        x_2, edge_index_2, edge_attr_2, batch_2, _, _  = self.pool2(x, edge_index_1, edge_attr_1, batch_1)
        #g2 = torch.cat([gmp(x_2, batch_2), gap(x_2, batch_2)], dim=1)
        

        # InnerCL augmentations of the third Graph
        x_2_1, edge_index_2_1, _ = aug1(x_2, edge_index_2)
        x_2_2, edge_index_2_2, _ = aug2(x_2, edge_index_2)
        g2_1 = global_add_pool(self.projection_head_3(self.conv3(x_2_1, edge_index_2_1)),batch_2)
        g2_2 = global_add_pool(self.projection_head_3(self.conv3(x_2_2, edge_index_2_2)),batch_2)

        x = F.relu(self.conv3(x_2, edge_index_2))
        x_third = x 
        g3 = torch.cat([gmp(x_third, batch_2), gap(x_third, batch_2)], dim=1)
        local_proj_3 = self.projection_head_3(x)
        proj_3 = global_add_pool(local_proj_3, batch_2)
        #proj_3 = torch.cat([gmp(local_proj_3, batch_2), gap(local_proj_3, batch_2)], dim=1)
        #g3 = torch.cat([gmp(x_3, batch_3), gap(x_3, batch_3)], dim=1)
        x = g1 + g2 + g3

        return x, proj_1, proj_2, proj_3, g0_1, g0_2, g1_1, g1_2, g2_1, g2_2  

    def get_embeddings(self, device, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                # data = data[0]
                data.to(device)
                x, *_ = self(data)
                ret.append(x)
                y.append(data.y)
            ret = torch.cat(ret, dim=0)
            y = torch.cat(y, dim=0)

        return ret, y
