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
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch.nn import Sequential, Linear, ReLU

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
parser.add_argument('--pooling', type=str, default='ASAP', help='Different pooling methods')
parser.add_argument('--augment', type=str, default='FE', help='Select Augment Way')

args = parser.parse_args()
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
wandb.config = args
wandb.init(project="HGCL", entity="sher-hao", config=args, reinit=True)


def gen_ran_output(data, model, vice_model, args):
    for (adv_name,adv_param), (name,param) in zip(vice_model.named_parameters(), model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            adv_param.data = param.data + args.eta * torch.normal(0,torch.ones_like(param.data)*param.data.std()).to(device)
    z2 = vice_model(data.x, data.edge_index, data.batch, data.num_graphs)
    return z2


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


def train(model, vice_model, dataloader, optimizer):
    model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(args.device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        x2 = gen_ran_output(data, model, vice_model, args)
        x1 = model(data.x, data.edge_index, data.batch, data.num_graphs)
        loss_aug = model.loss_cal(x2, x1)
        loss = loss_aug
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
        #! Add Noise or attack for the graphs
        g1, g2 = encoder_model.get_embedding(data.x, data.edge_index, data.batch)
        x.append(g1 + g2)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    acc_mean, acc_std = svc(x, y)

    return acc_mean


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(args.device)

        xs = []
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        xpool = [global_add_pool(x, batch) for x in xs]
        x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(args.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(args.device)
                x, _ = self.forward(x, edge_index, batch)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader):

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(args.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(args.device)
                x_g, x = self.forward(x, edge_index, batch)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):
        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss



def main():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:2"
    else:
        args.device = "cpu"
    Acc_Mean = []
    for i in range(args.num_runs):
        set_seed(i)

        dataset = TUDataset(osp.join('data', args.dataset), name=args.dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        dataset_num_features = max(dataset.num_features, 1)

        # test the randomization model
        print('--------------------------------')
        print('randomization model')
        aug1 = A.Identity()
        aug2 = A.PPRDiffusion(alpha=0.2, use_cache=False)
        Noise = A.Compose([A.NodeDropping(pn=0.2),
                           A.EdgeRemoving(pe=0.2)])

        model = simclr(args.hidden_dim, args.num_gc_layers).to(args.device)
        vice_model = simclr(args.hidden_dim, args.num_gc_layers).to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr)

        log_interval = args.eval_patience
        Accuracy = []
        with tqdm(total=args.epochs, desc=f'(T){i}') as pbar:
            for epoch in range(1, args.epochs + 1):
                loss = train(model, vice_model, dataloader, optimizer)
                if epoch % log_interval == 0:
                    model.eval()
                    acc_mean = test(model, dataloader)
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

