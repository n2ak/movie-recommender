from base import BaseModel
import torch
import torch.nn.functional as F
from torch import nn, optim
from dataclasses import dataclass, asdict


@dataclass(eq=True)
class NCFParams:
    nu: int
    nm: int
    u_embd: int
    m_embd: int
    lr: float
    hidden_layers: list[int]
    asdict = asdict


class NCF(BaseModel):
    def __init__(self, params: NCFParams) -> None:
        super().__init__(params)
        self.u_emb = nn.Embedding(params.nu, params.u_embd)
        self.m_emb = nn.Embedding(params.nm, params.m_embd)

        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.LazyLinear(i),
                    nn.ReLU(),
                    nn.Dropout(.25),
                )
                for i in params.hidden_layers
            ]
        )
        self.final = nn.LazyLinear(1, bias=False)

    def device_(self): return self.u_emb.weight.device

    def forward(self, u, m):
        u_emb = self.u_emb(u)
        m_emb = self.m_emb(m)
        hidden = self.mlp(torch.concat([u_emb, m_emb], dim=1))
        res = self.final(hidden)
        res = F.sigmoid(res)*5
        return res.squeeze(), None

    def train_tick(self, lossfn, metric, batch):
        self.train()
        device = self.device_()
        u, m, r = [b.to(device) for b in batch]
        pred, embeddings = self.forward(u, m)
        loss = lossfn(pred, r, embeddings, self.params)
        with torch.no_grad():
            m = metric(r.cpu().detach(), pred.cpu().detach()).item()
        return loss, m

    @classmethod
    def param_cls(self): return NCFParams


def default_ncf(nu, nm, device):
    ncfparams = NCFParams(
        nu=nu,
        nm=nm,
        u_embd=32,
        m_embd=32,
        lr=4e-3,
        hidden_layers=[32, 64, 32]
    )
    ncf = NCF(ncfparams).to(device)
    ncf_opt = torch.optim.AdamW(
        ncf.parameters(), lr=ncfparams.lr, weight_decay=.0001)
    last_state = {}
    return ncf, ncf_opt, last_state
