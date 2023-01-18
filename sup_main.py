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
import torch.nn.functional as F
# ! wandb
WANDB_API_KEY = '9e4f340d3a081dd1d047686edb29d362c8933632'
torch.set_printoptions(threshold=np.inf)

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description="Graph Pooling")
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='MUTAG/DD/COLLAB/PTC_MR/IMDB-BINARY/REDDIT-BINARY/REDDIT-MULTI-5K/NCI1/PROTEINS')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--seed', type=int, default=0, help='random seeds')
parser.add_argument('--pooling', type=str, default='ASAP', help='Different pooling methods')
parser.add_argument('--conv', type=str, default='GCN', help='Graph conv methods')
parser.add_argument('--hidden', type=int, default=128, help='hidden layers')
parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
parser.add_argument('--add_to_edge_score', type=float, default=0.5, help='add_to_edge_score')
parser.add_argument('--alpha', type=float, default=0.8, help='control the weight of the hierarcial cl')
parser.add_argument('--bleta', type=float, default=0.5, help='control the weight of the innercl')
parser.add_argument('--task', type=str, default='Graph', help='Graph classification or Node classification')
parser.add_argument('--patience', type=int, default=100, help='path to save result')

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

# def train(cf, model, dataloader):
#     optimizer = torch.optim.Adam(model.parameters(), lr=cf.lr, weight_decay=5e-4)
#     log_interval = 1
#     best_epochs = 0
#     best_acc = 0
#     best_std = 0
#     pbar = tqdm(range(1, cf.epochs + 1), ncols=100)
#     for epoch in pbar:
#         model.train()
#         loss_all = 0.0
#         for i, data in enumerate(dataloader):
#             optimizer.zero_grad()
#             data = data.to(cf.device)
#
#             if data.x is None:
#                 data.x = torch.ones(data.num_nodes, 1).to(cf.device)
#             cls, _, p1, p2, p3, g0_1, g0_2, g1_1, g1_2, g2_1, g2_2 = model(data)
#             cls_loss = F.nll_loss(cls, data.y)
#             loss = cf.alpha * innercl(p1, p2) + (1 - cf.alpha) * (innercl(p2, p3) + innercl(p1, p3)) + cf.bleta * innercl(g0_1,g0_2) + (1 - cf.bleta) * (
#                                innercl(g1_1, g1_2) + innercl(g2_1, g2_2))
#             Loss_all = loss + cls_loss
#             Loss_all.backward()
#             optimizer.step()
#             loss_all += Loss_all.item()
#         print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
#         wandb.log({"Loss": loss_all, "p1_p2_loss": innercl(p1, p2), "g0_loss": innercl(g0_1, g0_2),
#                    "g1_loss": innercl(g1_1, g1_2), "g2_loss": innercl(g2_1, g2_2)})
#         if epoch % log_interval == 0:
#             model.eval()
#             x, y = model.get_embeddings(cf.device, dataloader)
#             acc_mean, acc_std = svc(x, y)
#             wandb.log({"Acc": acc_mean, "Acc_std": acc_std})
#             print('acc_mean = ', acc_mean, '  acc_std = ', acc_std)
#             if acc_mean > best_acc:
#                 best_acc = acc_mean
#                 best_epochs = epoch
#                 best_std = acc_std
#                 wandb.run.summary["best_acc"] = acc_mean
#                 wandb.run.summary["best_epoch"] = epoch
#     wandb.log({'Best Epochs': best_epochs, 'Best Acc': best_acc, 'Best Acc Std': best_std})


def train(args, model, train_loader, val_loader, dir, save):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    min_loss = 1e10
    max_acc = 0
    patience_cnt = 0
    best_epoch = 0
    val_loss_values = []

    for epoch in range(args.epochs):
        loss_train = 0.0
        loss_dis = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            # output = model(data)
            output,  p1, p2, p3, g0_1, g0_2, g1_1, g1_2, g2_1, g2_2 = model(data)
            cls_loss = F.nll_loss(output, data.y)
            dis_loss = args.alpha * innercl(p1, p2) + (1 - args.alpha) * (innercl(p2, p3) + innercl(p1, p3)) #+ args.bleta * innercl(g0_1,g0_2) + (1 - args.bleta) * (innercl(g1_1, g1_2) + innercl(g2_1, g2_2))
            loss =  cls_loss +  0.01 * dis_loss
            loss.backward()
            optimizer.step()
            loss_train += cls_loss.item()
            loss_dis += dis_loss.item()
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
            for name, parms in model.named_parameters():
                print('-->name:', name,
                      '\n ->grad_value:', parms.grad),

        acc_train = correct / len(train_loader.dataset)
        val_acc, val_loss = test(args, model, val_loader)
        if val_acc > max_acc:
            max_acc = val_acc
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train), 'loss_dis: {:.6f}'.format(loss_dis),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(val_loss),
              'acc_val: {:.6f}'.format(val_acc), 'max_acc: {:.6f}'.format(max_acc))

        val_loss_values.append(val_loss)
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
            if not os.path.exists(dir):
                os.makedirs(dir)

            torch.save(model.state_dict(), osp.join(dir, save))
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

    print('Optimization Finished!')



def test(args, model, loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        output, *_ = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(output, data.y, reduction='sum').item()
    return correct / len(loader.dataset),loss / len(loader.dataset)

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
    dir = osp.join('~/HGCL/output/save/',f'{args.dataset}')
    save = str(args.seed) + '.pth'

    if args.task == 'Graph':
        train_loader, val_loader, test_loader = build_train_val_test_loader(args)
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
        train(args, model, train_loader, val_loader, dir, save)

        model.load_state_dict(torch.load(osp.join(dir, save)))
        test_acc, test_loss = test(args, model, test_loader)
        print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
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
