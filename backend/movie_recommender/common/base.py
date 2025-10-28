import torch
from torch import nn
from typing import Type


class MLP(nn.Module):
    def __init__(self, layer_neurons: list[int], act: Type[nn.Module] = nn.ReLU, p=.0, final_act=None, norm=False):
        super().__init__()
        layers = []
        for i, (inc, outc) in enumerate(zip(layer_neurons[:-1], layer_neurons[1:])):
            last = i == len(layer_neurons)-2
            layers.append(nn.Linear(inc, outc, bias=not last))
            if not last:
                if norm:
                    layers.append(nn.BatchNorm1d(outc))
                layers.append(act())
                if p:
                    layers.append(nn.Dropout(p))
        if final_act:
            layers.append(final_act())
        self.seq = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.seq(x)
