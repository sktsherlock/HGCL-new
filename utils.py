import torch as th
import torch.nn.functional as F
import math

from sklearn.svm import LinearSVC, SVC
from losses.losses import Loss
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV, StratifiedKFold
from torch_geometric.utils.num_nodes import maybe_num_nodes


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = th.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }

def linearsvc(embeds, labels):
    x = embeds.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies), np.std(accuracies)

def f_adj(edge_index, edge_attr, perm, num_nodes=None):
    if num_nodes != None:
        num_nodes = num_nodes
    else:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0

    #num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0

    mask = perm.new_full((num_nodes,), -1) 
    i = th.arange(perm.size(0), dtype=th.long, device=perm.device)

    mask[perm] = i 
    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    
    row, col = row[mask], col[mask]
    

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return th.stack([row, col], dim=0), edge_attr


def from_predefined_split(data):
    assert all([mask is not None for mask in [data.train_mask, data.test_mask, data.val_mask]])
    num_samples = data.num_nodes
    indices = th.arange(num_samples)
    return {
        'train': indices[data.train_mask],
        'valid': indices[data.val_mask],
        'test': indices[data.test_mask]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: th.FloatTensor, y: th.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: th.FloatTensor, y: th.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result


class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        test_macro = f1_score(y_test, classifier.predict(x_test), average='macro')
        test_micro = f1_score(y_test, classifier.predict(x_test), average='micro')

        return {
            'micro_f1': test_micro,
            'macro_f1': test_macro,
        }


class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None):
        if linear:
            self.evaluator = LinearSVC()
        else:
            self.evaluator = SVC()
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, params)


def get_positive_expectation(p_samples, average=True):
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Ep = log_2- F.softplus(- p_samples)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, average=True):
    """Computes the negative part of a JS Divergence.
    Args:
        q_samples: Negative samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Eq = F.softplus(- q_samples) + q_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq

def get_diffcult_expectation(p_samples, average=True):
    """Computes the positive part of a JS Divergence.
    Args:
        p_samples: Positive samples.
        average: Average the result over samples.
    Returns:
        th.Tensor
    """
    log_2 = math.log(2.)
    Eq = F.softplus(-p_samples) + p_samples
    # Eq = F.softplus(-p_samples) + p_samples - log_2

    if average:
        return Eq.mean()
    else:
        return Eq

def global_global_loss_(con0, con1):
    num_graphs = con0.shape[0]
    device = con0.device

    pos_mask = th.eye(num_graphs).to(device)
    neg_mask = 1 - pos_mask

    res = th.mm(con0, con1.t())


    E_pos = get_positive_expectation(res * pos_mask, average=False)
    E_pos = (E_pos * pos_mask).sum() / pos_mask.sum()
    E_neg = get_negative_expectation(res * neg_mask, average=False)
    E_neg = (E_neg * neg_mask).sum() / neg_mask.sum()

    return E_neg - E_pos

def global_global_negative_loss_(con0, con1):
    num_graphs = con0.shape[0]
    device = con0.device

    pos_mask = th.eye(num_graphs).to(device)
    neg_mask = 1 - pos_mask

    res = th.mm(con0, con1.t())

    E_neg_diffcult = get_diffcult_expectation(res * pos_mask, average=False)
    E_neg_diffcult = (E_neg_diffcult * pos_mask).sum() / pos_mask.sum()

    return E_neg_diffcult

def local_global_loss_(l_enc, g_enc, graph_id):

    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    device = g_enc.device

    pos_mask = th.zeros((num_nodes, num_graphs)).to(device)
    neg_mask = th.ones((num_nodes, num_graphs)).to(device)

    for nodeidx, graphidx in enumerate(graph_id):

        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = th.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos
    #return  -E_pos


class PAPloss():
    def __init__(self) -> None:
        super(PAPloss, self).__init__()

    def compute(self, anchor, sample1, sample2, *args, **kwargs) -> th.FloatTensor:
        num_graphs = sample1.shape[0]
        device = sample1.device
        pos_mask = th.eye(num_graphs).to(device)
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample1 = F.normalize(sample1, dim=-1, p=2)
        sample2 = F.normalize(sample2, dim=-1, p=2)

        similarity1 = anchor @ sample1.t()
        similarity2 = anchor @ sample2.t()
        similarity3 = anchor @ anchor.t()
        far =    sample1 @ sample2.t()
        loss1 = (similarity1 * pos_mask).sum() / pos_mask.sum()
        loss2 = (similarity2 * pos_mask).sum() / pos_mask.sum()
        loss3 = (far * pos_mask).sum() / pos_mask.sum()

        #loss = (similarity1 * pos_mask + similarity2 * pos_mask + far * pos_mask).sum(dim=-1)
        loss =   loss3 -(loss1 + loss2)/2 #loss3 - loss1 - loss2 
        return loss / 2

class PAP1Loss():
    def __init__(self) -> None:
        super(PAP1Loss, self).__init__()

    def compute(self, anchor, sample1, sample2, *args, **kwargs) -> th.FloatTensor:
        num_graphs = sample1.shape[0]
        device = sample1.device
        pos_mask = th.eye(num_graphs).to(device)
        anchor = F.normalize(anchor, dim=-1, p=2)
        sample1 = F.normalize(sample1, dim=-1, p=2)
        sample2 = F.normalize(sample2, dim=-1, p=2)

        # similarity1 = F.softplus(- anchor @ sample1.t())
        # similarity2 = F.softplus(- anchor @ sample2.t())
        similarity1 = anchor @ sample1.t()
        similarity2 = anchor @ sample2.t()
        #far = - sample1 @ sample2.t()
        loss1 = (similarity1 * pos_mask).sum() / pos_mask.sum()
        loss2 = (similarity2 * pos_mask).sum() / pos_mask.sum()
        #loss3 = (far * pos_mask).sum() / pos_mask.sum()
        
        #loss = (similarity1 * pos_mask + similarity2 * pos_mask + far * pos_mask).sum(dim=-1)
        loss = -loss1 - loss2 #+loss3
        return loss / 2

class Far():
    def __init__(self) -> None:
        super(Far, self).__init__()

    def compute(self, sample1, sample2, *args, **kwargs) -> th.FloatTensor:
        num_graphs = sample1.shape[0]
        device = sample1.device
        pos_mask = th.eye(num_graphs).to(device)
        sample1 = F.normalize(sample1, dim=-1, p=2)
        sample2 = F.normalize(sample2, dim=-1, p=2)
        far =  sample1 @ sample2.t()
        loss3 = (far * pos_mask).sum() / pos_mask.sum()
        
        #loss = (similarity1 * pos_mask + similarity2 * pos_mask + far * pos_mask).sum(dim=-1)
        return loss3

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask