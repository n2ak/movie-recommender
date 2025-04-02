from .base import BaseModel, BaseParams, MLP
import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass


@dataclass(eq=True)
class VAEParams(BaseParams):
    encoder_dims: list[int]
    encoder_activations: list[nn.Module]
    decoder_dims: list[int]
    decoder_activations: list[nn.Module]
    latent_dim: int
    input_dim: int


class VAE(BaseModel[VAEParams]):

    def init(self) -> None:
        params = self.config.params
        self.encoder = MLP(params.encoder_dims)
        self.decoder = MLP(params.decoder_dims, final_act=nn.Sigmoid)
        self.mu = nn.Linear(params.encoder_dims[-1], params.latent_dim)
        self.var = nn.Linear(params.encoder_dims[-1], params.latent_dim)

    def forward(self, u):
        enc = self.encoder.forward(u)
        mu: torch.Tensor = self.mu(enc)
        var: torch.Tensor = self.var(enc)
        std = (var**.5).exp()
        eps = torch.randn_like(std)
        latent = eps * std + mu
        res = self.decoder.forward(latent)
        return res, None

    def train_tick(self, lossfn, metric, batch):
        self.train()
        device = self.device_()
        r = batch.float().to(device)
        pred, embeddings = self.forward(r)

        loss = lossfn(pred, r, embeddings, self.config)
        with torch.no_grad():
            m = metric(r.detach().cpu(), pred.detach().cpu()).item()
        return loss, m
