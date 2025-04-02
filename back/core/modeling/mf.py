from .base import BaseModel, BaseParams
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from dataclasses import dataclass


@dataclass(eq=True)
class MFParams(BaseParams):
    nu: int  # number of users
    nm: int  # number of moveis
    embd: int = 32
    init_weights: bool = False


class MF(BaseModel[MFParams]):
    """
    Matrix factorization.
    """

    def init(self) -> None:
        params = self.config.params
        self.user_emb = nn.Embedding(params.nu, params.embd)
        self.movie_emb = nn.Embedding(params.nm, params.embd)
        self.user_bias = nn.Embedding(params.nu, 1)
        self.movie_bias = nn.Embedding(params.nm, 1)
        if params.init_weights:
            self.user_emb.weight.data.uniform_(0, 0.05)
            self.movie_emb.weight.data.uniform_(0, 0.05)

    def forward(self, users, movies):
        u = self.user_emb(users)
        m = self.movie_emb(movies)
        ub = self.user_bias(users)
        mb = self.movie_bias(movies)
        return torch.sigmoid(u*m + ub + mb)
