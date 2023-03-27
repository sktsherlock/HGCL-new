import os
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
import json
from build_data import *
import random
import wandb
import GCL.augmentors as A
from eval import *
from models.Encoder import GConv
from torch_geometric.data import DataLoader
from GCL.models import DualBranchContrast
import GCL.losses as L
from torch_geometric.nn import TopKPooling, SAGPooling, ASAPooling, EdgePooling

# ! wandb
WANDB_API_KEY = '9e4f340d3a081dd1d047686edb29d362c8933632'
torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dataset', type=str, default='PTC_MR',
                    help='MUTAG/DD/COLLAB/PTC_MR/IMDB-BINARY/REDDIT-BINARY/REDDIT-MULTI-5K/NCI1/PROTEINS')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--seed', type=int, default=0, help='random seeds')
parser.add_argument('--hidden', type=int, default=128, help='hidden layers')
parser.add_argument('--layers', type=int, default=2, help='layers')
parser.add_argument('--tradeoff', type=float, default=0.8, help='权重')
parser.add_argument('--mix', type=float, default=0.8, help='权重')
parser.add_argument('--mixup', type=float, default=0.8, help='权重')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio')
parser.add_argument('--min_score', type=float, default=None, help='pooling min_score')
parser.add_argument('--edge_drop', type=float, default=0.2, help='pooling ratio')
parser.add_argument('--feature_mask', type=float, default=0.2, help='pooling ratio')
parser.add_argument('--up', type=float, default=0.4, help='the threshold of the tradeoff')
parser.add_argument('--eval_patience', type=int, default=10, help='the patience of evaluate')
parser.add_argument('--num_runs', type=int, default=5, help='the patience of evaluate')
parser.add_argument('--warmup_epochs', type=int, default=100, help='the number of warmup_epochs')
parser.add_argument('--test_init', type=bool, default=False, help='whether test the initial state')
parser.add_argument('--add_to_edge_score', type=float, default=0.5, help='add_to_edge_score')
parser.add_argument('--noise', type=float, default=0.1, help='noise 程度')
parser.add_argument('--pooling', type=str, default='ASAP', help='Different pooling methods')
parser.add_argument('--augment', type=str, default='FE', help='Select Augment Way')

args = parser.parse_args()
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
wandb.config = args
wandb.init(project="HGCL", entity="sher-hao", config=args, reinit=True)


def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(args.device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss

def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(args.device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, *_ = encoder_model(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    acc_mean, acc_std = svc(x, y)

    return acc_mean

class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, noise=None):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        self.noise = noise

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        Noise_x, Noise_A, _ = self.noise(x, edge_index)
        z, g = self.encoder(Noise_x, Noise_A, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


def main():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    Acc_Mean = []
    for i in range(args.num_runs):
        set_seed(i)

        dataset = TUDataset(osp.join('data', args.dataset), name=args.dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        input_dim = max(dataset.num_features, 1)

        # test the randomization model
        print('--------------------------------')
        print('randomization model')
        aug1 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                               A.NodeDropping(pn=0.1),
                               A.FeatureMasking(pf=0.1),
                               A.EdgeRemoving(pe=0.1)], 1)
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                               A.NodeDropping(pn=0.1),
                               A.FeatureMasking(pf=0.1),
                               A.EdgeRemoving(pe=0.1)], 1)
        Noise = A.Compose([A.NodeDropping(pn=args.noise),
                           A.EdgeRemoving(pe=args.noise)])

        gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden, num_layers=args.layers).to(args.device)


        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), noise=Noise).to(args.device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(args.device)
        optimizer = Adam(encoder_model.parameters(), lr=args.lr)

        log_interval = args.eval_patience
        Accuracy = []
        with tqdm(total=args.epochs, desc=f'(T){i}') as pbar:
            for epoch in range(1, args.epochs + 1):
                loss = train(encoder_model, contrast_model, dataloader, optimizer)
                if epoch % log_interval == 0:
                    encoder_model.eval()
                    acc_mean = test(encoder_model, dataloader)
                    Accuracy.append(acc_mean)
                    wandb.log({'Acc': acc_mean})

                pbar.set_postfix({'loss': loss})
                wandb.log({"loss": loss})
                pbar.update()

        wandb.log({'Acc': acc_mean})

        wandb.log({'Best Acc': max(Accuracy)})
        Acc_Mean.append(max(Accuracy))
    print('Run 5, the mean accuracy is {}'.format(np.mean(Acc_Mean)))
    wandb.log({'Mean Acc': np.mean(Acc_Mean), 'Std': np.std(Acc_Mean)})



if __name__ == '__main__':
    main()

