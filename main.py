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
from models.Encoder import HGCL
from eval import *
from utils import generate_split

# ! wandb
WANDB_API_KEY = '9e4f340d3a081dd1d047686edb29d362c8933632'
torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dataset', type=str, default='DD',
                    help='MUTAG/DD/COLLAB/PTC_MR/IMDB-BINARY/REDDIT-BINARY/REDDIT-MULTI-5K/NCI1/PROTEINS')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--seed', type=int, default=0, help='random seeds')
parser.add_argument('--pooling', type=str, default='ASAP', help='Different pooling methods')
parser.add_argument('--conv', type=str, default='GIN', help='Graph conv methods')
parser.add_argument('--hidden', type=int, default=128, help='hidden layers')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--add_to_edge_score', type=float, default=0.5, help='add_to_edge_score')
parser.add_argument('--alpha', type=float, default=0.2, help='control the weight of the hierarcial cl')
parser.add_argument('--bleta', type=float, default=0.2, help='control the weight of the innercl')
parser.add_argument('--task', type=str, default='Graph', help='Graph classification or Node classification')
parser.add_argument('--up', type=float, default=0.3, help='the threshold of the tradeoff')
parser.add_argument('--warmup_epochs', type=int, default=100, help='the number of warmup_epochs')

args = parser.parse_args()
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
wandb.config = args
wandb.init(project="HGCL", entity="sher-hao", config=args, reinit=True)


def innercl(proj_1, proj_2):
    from losses.infonce import infonce
    return infonce(proj_1, proj_2)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train(cf, model, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.lr, weight_decay=5e-4)
    log_interval = 2
    best_epochs = 0
    best_acc = 0
    best_std = 0
    pbar = tqdm(range(1, cf.epochs + 1), ncols=100)
    for epoch in pbar:
        model.train()
        loss_all = 0.0
        tradeoff = diffusion(cf, epoch=epoch)
        for i, data in enumerate(dataloader):
            optimizer.zero_grad()
            data = data.to(cf.device)

            if data.x is None:
                data.x = torch.ones(data.num_nodes, 1).to(cf.device)
            _, p1, p2, g0_1, g0_2, g1_1, g1_2 = model(data, tradeoff)

            loss = innercl(g0_1, g0_2) + cf.alpha * (innercl(p1, p2) + innercl(g1_1, g1_2))
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all))
        wandb.log({"Loss": loss_all, "p1_p2_loss": innercl(p1, p2), "g0_loss": innercl(g0_1, g0_2),
                   "g1_loss": innercl(g1_1, g1_2)})
        if epoch % log_interval == 0:
            model.eval()
            x, y = model.get_embeddings(cf.device, dataloader)
            acc_mean, acc_std = svc(x, y)
            wandb.log({"Acc": acc_mean, "Acc_std": acc_std})
            print('acc_mean = ', acc_mean, '  acc_std = ', acc_std)
            if acc_mean > best_acc:
                best_acc = acc_mean
                best_epochs = epoch
                best_std = acc_std
                wandb.run.summary["best_acc"] = acc_mean
                wandb.run.summary["best_epoch"] = epoch
    wandb.log({'Best Epochs': best_epochs, 'Best Acc': best_acc, 'Best Acc Std': best_std})


def train_node(cf, model, dataset):
    # load data
    data = dataset[0]
    data = data.to(cf.device)
    log_interval = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.lr)
    best_epochs = 0
    best_acc = 0
    pbar = tqdm(range(1, cf.epochs + 1), ncols=100)
    for epoch in pbar:
        model.train()
        loss_all = 0.0
        optimizer.zero_grad()
        data = data.to(cf.device)

        _, p1, p2, g0_1, g0_2, g1_1, g1_2 = model(data)

        loss = cf.alpha * innercl(p1, p2) + innercl(g0_1, g0_2) + cf.bleta * innercl(g1_1, g1_2)
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
        print('Epoch {}, Loss {}'.format(epoch, loss_all))
        wandb.log({"Loss": loss_all, "p1_p2_loss": innercl(p1, p2), "g0_loss": innercl(g0_1, g0_2),
                   "g1_loss": innercl(g1_1, g1_2)})
        if epoch % log_interval == 0:
            acc = test(cf, model, data)
            wandb.log({"Acc": acc})
            print('Acc = ', acc)
            if acc > best_acc:
                best_acc = acc
                best_epochs = epoch
    wandb.log({'Best Epochs': best_epochs, 'Best Acc': best_acc})


def test(cf, model, data):
    model.eval()
    data = data.to(cf.device)
    z = model(data)
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, data, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, data, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    return


def diffusion(args, epoch, way='linear'):
    if way == 'linear':
        tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.02, args.up).float()
    else:
        raise ValueError('the other way is not implement')
    return tradeoff
    # Adamw


def main():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    # 设置随机种子
    set_seed(args.seed)
    # 加载数据增强策略
    import GCL.augmentors as A
    aug1 = A.Identity()
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.3)])

    # 加载模型; 加载数据
    # For the Graph Classification
    if args.task == 'Graph':
        loader = build_loader(args)
        # 训练图级别任务 模型

    elif args.task == 'Node':
        path = osp.expanduser('./dataset')
        path = osp.join(path, args.dataset)
        dataset = get_dataset(path, args.dataset)
        args.num_classes = dataset[0]['y'].unique().shape[0]
        args.num_features = dataset[0]['x'].shape[1]

    if args.task == 'Graph':
        if args.pooling in {'topk', 'TopK'}:
            from models.TopKNet import TopKNet
            model = TopKNet(args=args, augmentor=(aug1, aug2))
        elif args.pooling in {'SAGPooling', 'SAG'}:
            from models.SAGNet import SAGNet
            model = SAGNet(args=args, augmentor=(aug1, aug2))
        elif args.pooling in {'EdgePooling', 'Edge'}:
            from models.EdgeNet import EdgeNet
            model = EdgeNet(args=args, augmentor=(aug1, aug2))
        elif args.pooling in {'ASAPooling', 'ASAP'}:
            from models.ASAPNet import ASAPNet
            model = ASAPNet(args=args, augmentor=(aug1, aug2))
        else:
            raise ValueError('Invalid pooling type')
        model.to(args.device)
        train(args, model, loader)
    else:
        if args.pooling in {'topk', 'TopK'}:
            from models.TopKNet import TopKNet
            model = TopKNet(args=args, augmentor=(aug1, aug2))
        elif args.pooling in {'SAGPooling', 'SAG'}:
            from models.SAGNet import SAGNet
            model = SAGNet(args=args, augmentor=(aug1, aug2))
        elif args.pooling in {'EdgePooling', 'Edge'}:
            from models.EdgeNet import EdgeNet
            model = EdgeNet(args=args, augmentor=(aug1, aug2))
        elif args.pooling in {'ASAPooling', 'ASAP'}:
            from models.ASAPNet import ASAPNet
            model = ASAPNet(args=args, augmentor=(aug1, aug2))
        else:
            raise ValueError('Invalid pooling type')
        model.to(args.device)
        train_node(args, model, dataset)


if __name__ == '__main__':
    main()
