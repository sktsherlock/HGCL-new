import torch
import os.path as osp
from torch import nn
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from torch_scatter import scatter_add

def degree_as_tag(dataset):
    all_features = []
    all_degrees = []
    tagset = set([])

    for i in range(len(dataset)):
        edge_weight = torch.ones((dataset[i].edge_index.size(1),))
        degree = scatter_add(edge_weight, dataset[i].edge_index[0], dim=0)
        degree = degree.detach().numpy().tolist()
        tagset = tagset.union(set(degree))
        all_degrees.append(degree)
    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for i in range(len(dataset)):
        node_features = torch.zeros(len(all_degrees[i]), len(tagset))
        node_features[range(len(all_degrees[i])), [tag2index[tag] for tag in all_degrees[i]]] = 1
        all_features.append(node_features)
    return all_features, len(tagset)


def build_data_list(dataset):
    node_features, num_features = degree_as_tag(dataset)  

    data_list = []
    for i in range(len(dataset)):
        old_data = dataset[i]
        new_data = Data(x=node_features[i], edge_index=old_data.edge_index, y=old_data.y)
        data_list.append(new_data)
    return data_list, num_features


def build_loader(args, batch_size, Data):
    dataset = TUDataset(osp.join('data', Data), name=Data)
    args.num_classes = dataset.num_classes
    loader = DataLoader(dataset, batch_size = batch_size,shuffle = True,  num_workers = 20)
    loader_test = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 20)
    args.num_features = max(dataset.num_features, 1)

    return  loader, loader_test
