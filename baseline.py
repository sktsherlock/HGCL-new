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
from models.Encoder import GConv, Encoder
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
parser.add_argument('--dataset', type=str, default='PROTEINS',
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
parser.add_argument('--noise', type=float, default=0.1, help='noise 程度')
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


def add_noise(x, tradeoff):
    if tradeoff > 0:
        if tradeoff <= 1:
            x = (1 - tradeoff) * x + torch.randn_like(x) * tradeoff
        else:
            raise ValueError('tradeoff <= 1')
    return x


def train(encoder_model, contrast_model, dataloader, optimizer, tradeoff=0.1):
    encoder_model.train()
    epoch_loss = 0
    epoch_loss_0 = 0
    epoch_loss_1 = 0
    align_loss_0 = 0
    align_loss_1 = 0
    log_interval = 10
    for data in dataloader:
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, g_0, g_1, g1, g2, g3, g4, batch_1 = encoder_model(data.x, data.edge_index, data.batch)
        # Mixup the Graph-level and Subgraph-level embeddings
        mix_g1 = args.mix * g1 + (1 - args.mix * g4) # alpha
        mix_g2 = args.mix * g2 + (1 - args.mix * g3)

        g1, g2 = [encoder_model.encoder.project(g) for g in [mix_g1, mix_g2]]
        g3, g4 = [encoder_model.encoder.project(g) for g in [g_0, g_1]]

        loss_0 = contrast_model(g1=g1, g2=g2)
        loss_1 = contrast_model(g1=g3, g2=g4)

        loss = loss_0 + args.tradeoff * loss_1
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_0 += loss_0.item()
        epoch_loss_1 += loss_1.item()

    return epoch_loss, epoch_loss_0, epoch_loss_1


def test(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g, g1, *_ = encoder_model(data.x, data.edge_index, data.batch)
        g = args.mixup * g + (1 - args.mixup) * g1
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    acc_mean, acc_std = svc(x, y)

    return acc_mean

def test_robost(encoder_model, dataloader):
    encoder_model.eval()
    x = []
    y = []
    for data in dataloader:
        data = data.to(args.device)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        #! Add Noise or attack for the graphs
        _, g = encoder_model.get_embedding(data.x, data.edge_index, data.batch)
        x.append(g)
        y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    acc_mean, acc_std = svc(x, y)

    return acc_mean

def test_0(dataloader):
    from torch_geometric.nn import global_add_pool,global_mean_pool
    x = []
    y = []
    gs = []
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        x.append(data.x)
        y.append(data.y)
        gs.append(global_add_pool(data.x, data.batch))
    x = torch.cat(gs, dim=0)
    y = torch.cat(y, dim=0)

    acc_mean, acc_std = svc(x, y)

    return acc_mean


def diffusion(args, epoch, way='stage'):
    if way == 'stage':
        if epoch <= 10:
            tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.01, 0.05).float()
        elif epoch <= 20:
            tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.1, 0.2).float()
        else:
            tradeoff = args.up
    else:
        raise ValueError('the other way is not implement')
    return tradeoff


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
        if args.augment in {'Random'}:
            aug1 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
            aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
        elif args.augment in {'FE'}:
            aug1 = A.Compose([A.FeatureMasking(pf=args.feature_mask)])
            aug2 = A.Compose([A.EdgeRemoving(pe=args.edge_drop)])
        elif args.augment in {'SNE'}:
            aug1 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
            aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)
        else:
            raise ValueError('Not implement for the Augmented way!')
        Noise = A.Compose([A.NodeDropping(pn=args.noise),
                           A.EdgeRemoving(pe=args.noise)])
        gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden, num_layers=args.layers).to(args.device)
        gconv2 = GConv(input_dim=args.hidden * args.layers, hidden_dim=args.hidden, num_layers=args.layers).to(
            args.device)
        if args.pooling in {'topk', 'TopK'}:
            pooling = TopKPooling(args.hidden * args.layers, ratio=args.pooling_ratio, min_score=args.min_score)
        elif args.pooling in {'SAGPooling', 'SAG'}:
            pooling = SAGPooling(args.hidden * args.layers, ratio=args.pooling_ratio, min_score=args.min_score)
        elif args.pooling in {'EdgePooling', 'Edge'}:
            pooling = EdgePooling(args.hidden * args.layers, edge_score_method= EdgePooling.compute_edge_score_softmax, dropout=args.dropout, add_to_edge_score = 0.0)
        elif args.pooling in {'ASAPooling', 'ASAP'}:
            pooling = ASAPooling(args.hidden * args.layers, ratio=args.pooling_ratio)
        else:
            raise ValueError('Not implement')
        encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), pooling=pooling, encoder2=gconv2, pool_way=args.pooling, noise=Noise).to(args.device)
        contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(args.device)
        optimizer = Adam(encoder_model.parameters(), lr=args.lr)

        log_interval = args.eval_patience
        Accuracy = []
        with tqdm(total=args.epochs, desc=f'(T){i}') as pbar:
            for epoch in range(1, args.epochs + 1):
                loss, loss_0, loss_1 = train(encoder_model, contrast_model, dataloader, optimizer)
                if epoch % log_interval == 0:
                    encoder_model.eval()
                    acc_mean = test_robost(encoder_model, dataloader)
                    Accuracy.append(acc_mean)
                    wandb.log({'Acc': acc_mean})

                pbar.set_postfix({'loss': loss})
                wandb.log({"loss": loss, "Mix_CL_loss": loss_0, "Hierarchical_loss": loss_1})
                pbar.update()

        wandb.log({'Acc': acc_mean})

        wandb.log({'Best Acc': max(Accuracy)})
        Acc_Mean.append(max(Accuracy))
    print('Run 5, the mean accuracy is {}'.format(np.mean(Acc_Mean)))
    wandb.log({'Mean Acc': np.mean(Acc_Mean), 'Std': np.std(Acc_Mean)})


def test_init():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    Acc = []
    with tqdm(total=5, desc='(T)') as pbar:
        for i in range(5):
            set_seed(i)

            dataset = TUDataset(osp.join('data', args.dataset), name=args.dataset)
            dataloader = DataLoader(dataset, batch_size=args.batch_size)
            input_dim = max(dataset.num_features, 1)

            # test the randomization model
            # aug1 = A.Identity()
            # aug2 = A.Identity()
            #
            # gconv = GConv(input_dim=input_dim, hidden_dim=args.hidden, num_layers=2).to(args.device)
            # encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(args.device)
            # optimizer = Adam(encoder_model.parameters(), lr=0.01)
            # initial_acc_mean = test(encoder_model, dataloader)
            initial_acc_mean = test_0(dataloader)
            Acc.append(initial_acc_mean)
            pbar.set_postfix({'Acc': initial_acc_mean})
            pbar.update()
        wandb.log({"Accuracy": np.mean(Acc), "Std": np.std(Acc)})


if __name__ == '__main__':
    if args.test_init == False:
        main()
    else:
        test_init()
