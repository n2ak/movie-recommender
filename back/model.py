from base import BaseModel
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
class MFParams:
    nu: int
    nm: int
    lr: float
    embd: int = 32
    use_bias: bool = False
    init_weights: bool = False
    usePQ: bool = False

    asdict = asdict


class MF(BaseModel):
    def __init__(self, params: MFParams, seed=None) -> None:
        super().__init__(params)
        if seed is not None:
            from train import set_seed
            set_seed(seed)
        self.emb1 = nn.Embedding(params.nu, params.embd)
        self.emb2 = nn.Embedding(params.nm, params.embd)
        self.emb3 = nn.Embedding(params.nu, 1)
        self.emb4 = nn.Embedding(params.nm, 1)
        if params.init_weights:
            self.emb1.weight.data.uniform_(0, 0.05)
            self.emb2.weight.data.uniform_(0, 0.05)

    def forward(self, u, m):
        ub = self.emb3(u)
        mb = self.emb4(m)
        P = u = (self.emb1(u))
        Q = m = (self.emb2(m))
        # if self.params.use_bias:
        #     u = F.gelu(u)
        #     m = F.gelu(m)
        # + ub.squeeze() + mb.squeeze())*5, (P, Q, ub, mb)
        return torch.sigmoid((u*m).sum(1))*5, None

    def device_(self): return self.emb1.weight.device

    @classmethod
    def param_cls(self): return MFParams
