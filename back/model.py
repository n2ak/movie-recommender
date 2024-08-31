import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import tqdm
from dataclasses import dataclass, asdict


@dataclass(eq=True)
class Params:
    nu: int
    nm: int
    ubs: int
    mbs: int
    lr: float
    embd: int = 32
    use_gelu: bool = False
    init_weights: bool = False

    asdict = asdict


class MF(nn.Module):
    def __init__(self, params: Params) -> None:
        super().__init__()
        self.params = params
        self.emb1 = nn.Embedding(params.nu, params.embd)
        self.emb2 = nn.Embedding(params.nm, params.embd)
        if params.init_weights:
            self.emb1.weight.data.uniform_(0, 0.05)
            self.emb2.weight.data.uniform_(0, 0.05)

    def forward(self, u, m):
        u = (self.emb1(u))
        m = (self.emb2(m))
        if self.params.use_gelu:
            u = F.gelu(u)
            m = F.gelu(m)
        res = u@m.T
        return torch.sigmoid(res)*5, u, m

    def save(self, filename="models/model.pt", optimizer=None):
        torch.save({
            "model": self.state_dict(),
            "params": self.params.asdict(),
            "optim": optimizer.state_dict() if optimizer is not None else None,
        }, filename)
        print("Model saved to:", filename)

    @staticmethod
    def load(filename="models/model.pt", opt=False):
        state = torch.load(filename)
        model = MF(Params(**state["params"]))
        model.load_state_dict(state["model"])
        print("Model loaded from:", filename)
        if not opt:
            return model
        optimizer = optim.Adam(model.parameters(), lr=model.params.lr)
        optimizer.load_state_dict(state["optim"])
        return model, optimizer

    def predict(self, userIds, movieIds, start=0, count=None, device="cpu"):
        assert len(userIds) != 0
        if len(movieIds) == 0:
            movieIds = torch.arange(0, self.params.nm).to(device)
        else:
            movieIds = torch.tensor(movieIds).to(device)
        userIds = torch.tensor(userIds).to(device)

        with torch.no_grad():
            res, _, _ = self.forward(userIds, movieIds)
        ratings, ids = res.squeeze().sort()
        end = start + count if count is not None else None
        ratings = ratings[start:end].tolist()
        ids = ids[start:end].tolist()
        return ids, ratings


def loss_fn(pred, real):
    loss = ((pred-real)**2 * (real != 0)).mean()
    loss = loss + (pred**2 * (real == 0)).mean() * .1
    return loss


def compute_f1(real, pred):
    from sklearn.metrics import f1_score
    real = (real.flatten().detach()*2).round().int().numpy()
    pred = (pred.flatten().detach()*2).round().int().numpy()
    return f1_score(real, pred, average="weighted")


def random_range(l, h, len):
    start = torch.randint(l, h-len, (1,)).item()
    return torch.arange(start, start+len)


def train1(model, optimizer, loss_fn, real_matrix, epochs, params: Params):
    losses = []
    stds = []
    f1 = []
    for _ in (bar := tqdm.trange(epochs)):
        ur = random_range(0, params.nu, params.ubs)
        mr = random_range(0, params.nm, params.mbs)
        _train(model, optimizer, loss_fn, real_matrix, ur, mr, losses, stds, f1)
        bar.set_description(f"Loss : {losses[-1]:.4f}")

    return losses, stds, f1


def train2(model, optimizer, loss_fn, real_matrix, epochs, params: Params):
    losses = []
    stds = []
    f1 = []
    for _ in (bar := tqdm.trange(epochs)):
        for i in range(0, params.nu-params.ubs, params.ubs):
            ur = torch.arange(i, i+params.ubs).int()
            for j in range(0, params.nm-params.mbs, params.mbs):
                mr = torch.arange(j, j+params.mbs).int()
                _train(model, optimizer, loss_fn,
                       real_matrix, ur, mr, losses, stds, f1)
                bar.set_description(f"Loss : {losses[-1]:.4f}")

    return losses, stds, f1


def _train(model, optimizer, loss_fn, real_matrix, ur, mr, losses, stds, f1):

    matrix = real_matrix[ur.tolist()][:, mr.tolist()]
    res, u, m = model.forward(ur, mr)
    loss = loss_fn(res, matrix)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    stds.append(res.std().item())
    f1.append(compute_f1(matrix, res))
