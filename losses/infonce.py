import torch
import numpy as np
import torch.nn.functional as F

from .losses import Loss


def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1, dim=-1, p=2)
    h2 = F.normalize(h2, dim= -1, p=2)
    return h1 @ h2.t()


def infonce(anchor, sample):
    tau = 1
    f = lambda x: torch.exp(x / tau)
    sim = f(_similarity(anchor, sample))  # anchor x sample
    num_graphs = anchor.shape[0]
    device = anchor.device
    pos_mask = torch.eye(num_graphs).to(device)
    neg_mask = 1 - pos_mask
    assert sim.size() == pos_mask.size()  # sanity check
    pos = (sim * pos_mask).sum(dim=1)
    neg = (sim * neg_mask).sum(dim=1)

    loss = pos / (pos + neg)
    loss = -torch.log(loss)

    return loss.mean()



class InfoNCE():
    """
    InfoNCE loss for single positive.
    """
    def __init__(self):
        super(InfoNCE, self).__init__()


    def compute(self, anchor, sample, *args, **kwargs):
        tau = 1
        f = lambda x: torch.exp(x / tau)
        sim = f(_similarity(anchor, sample))  # anchor x sample
        num_graphs = anchor.shape[0]
        device = anchor.device
        pos_mask = torch.eye(num_graphs).to(device)
        neg_mask = 1 - pos_mask
        assert sim.size() == pos_mask.size()  # sanity check
        pos = (sim * pos_mask).sum(dim=1)
        neg = (sim * neg_mask).sum(dim=1)

        loss = pos / (pos + neg)
        loss = -torch.log(loss)

        return loss.mean()
