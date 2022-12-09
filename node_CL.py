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
import torch.nn.functional as F
from eval import *
from models.Encoder import GCN_Conv, Node_Encoder
from torch_geometric.data import DataLoader
from GCL.models import DualBranchContrast
import GCL.losses as L

# ! wandb
WANDB_API_KEY = '9e4f340d3a081dd1d047686edb29d362c8933632'
torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--dataset', type=str, default='WikiCS',
                    help='WikiCS')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--seed', type=int, default=0, help='random seeds')
parser.add_argument('--hidden', type=int, default=256, help='hidden layers')
parser.add_argument('--proj_hidden', type=int, default=256, help='proj hidden layers')
parser.add_argument('--activation', type=str, default='prelu', help='activation function')
parser.add_argument('--tau', type=float, default=0.4, help='the tau of the infoNce loss')
parser.add_argument('--up', type=float, default=0.25, help='the threshold of the tradeoff')
parser.add_argument('--warmup_epochs', type=int, default=1000, help='the number of warmup_epochs')
parser.add_argument('--test_init', type=bool, default=False, help='whether test the initial state')

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


def add_noise(x, tradeoff):
    if tradeoff > 0:
        if tradeoff <= 1:
            x = (1 - tradeoff) * x + torch.randn_like(x) * tradeoff
        else:
            raise ValueError('tradeoff <= 1')
    return x


def train(encoder_model, contrast_model, data, optimizer, tradeoff):
    encoder_model.train()
    optimizer.zero_grad()
    z, z1, z2 = encoder_model(data.x, data.edge_index, data.edge_attr, tradeoff=tradeoff)
    h1, h2 = [encoder_model.project(x) for x in [z1, z2]]
    loss = contrast_model(h1, h2)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(args, encoder_model, data, dataset, split):
    encoder_model.eval()
    z, _, _ = encoder_model(data.x, data.edge_index, data.edge_attr)

    evaluator = MulticlassEvaluator()
    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}', num_epochs=800)['acc']
            accs.append(acc)
        acc = sum(accs) / len(accs)
    else:
        acc = log_regression(z, dataset, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    return acc


def diffusion(args, epoch, way='stage'):
    if way == 'stage':
        if epoch <= 400:
            tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.01, 0.05).float()
        elif epoch <= 1000:
            tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.1, 0.15).float()
        else:
            tradeoff = args.up
    else:
        raise ValueError('the other way is not implement')
    return tradeoff


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU,
        'rrelu': F.rrelu
    }

    return activations[name]


def main():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    set_seed(args.seed)
    device = args.device
    dataset = get_dataset(osp.join('data', args.dataset), args.dataset)
    data = dataset[0].to(device)
    # generate split
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.3), A.FeatureMasking(pf=0.1)])
    act = get_activation(args.activation)

    gconv = GCN_Conv(input_dim=dataset.num_features, hidden_dim=args.hidden, activation=act, num_layers=2).to(device)
    encoder_model = Node_Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden,
                                 proj_dim=args.proj_hidden).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=args.tau), mode='L2L', intraview_negs=True).to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)
    log_interval = 500
    Accuracy = []
    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(1, args.epochs + 1):
            tradeoff = torch.clamp(torch.tensor(epoch / args.warmup_epochs), 0.1, args.up).float()
            loss = train(encoder_model, contrast_model, data, optimizer, tradeoff=tradeoff)
            if epoch % log_interval == 0:
                encoder_model.eval()
                test_result = test(args, encoder_model, data, dataset, split)
                print('Accuracy: {}'.format(test_result))
                Accuracy.append(test_result)

            pbar.set_postfix({'loss': loss})
            pbar.update()

    test_result = test(args, encoder_model, data, dataset, split)
    Accuracy.append(test_result)
    print(f'(E): The Best test result is={max(Accuracy)}')


def test_init():
    # 检测是否有可用GPU
    if torch.cuda.is_available():
        args.device = "cuda:0"
    else:
        args.device = "cpu"
    Acc = []
    with tqdm(total=20, desc='(T)') as pbar:
        for i in range(20):
            set_seed(i)
            device = args.device
            dataset = get_dataset(osp.join('data', args.dataset), args.dataset)
            data = dataset[0].to(device)
            # generate split
            split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)

            aug1 = A.Identity()
            aug2 = A.Identity()

            gconv = GCN_Conv(input_dim=dataset.num_features, hidden_dim=args.hidden, activation=torch.nn.ReLU,
                             num_layers=2).to(device)
            encoder_model = Node_Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=args.hidden, proj_dim=32).to(
                device)
            test_result = test(args, encoder_model, data, dataset, split)

            Acc.append(test_result)
            pbar.set_postfix({'Acc': test_result})
            pbar.update()
        wandb.log({"Accuracy": np.mean(Acc), "Std": np.std(Acc)})


if __name__ == '__main__':
    if args.test_init == False:
        main()
    else:
        test_init()
