from typing import Type
import torch
import numpy as np
from torch import nn
from typing import TypeVar
from dataclasses import dataclass,  field


T = TypeVar('T')


class List(list[T]):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name


@dataclass
class TrainingResult:
    train_losses: List[float] = field(
        default_factory=lambda: List("train loss"))
    valid_losses: List[float] = field(
        default_factory=lambda: List("valid loss"))

    # f1/rmse for example
    train_scores: List[float] = field(
        default_factory=lambda: List("train score"))
    valid_scores: List[float] = field(
        default_factory=lambda: List("valid score"))

    # asdict = asdict

    def asdict(self):
        return dict(
            train_losses=self.train_losses,
            valid_losses=self.valid_losses,
            train_scores=self.train_scores,
            valid_scores=self.valid_scores,
        )

    @staticmethod
    def fromdict(d: dict):
        return TrainingResult(
            List("train loss", d["train_losses"]),
            List("valid loss", d["valid_losses"]),
            List("train score", d["train_scores"]),
            List("valid score", d["valid_scores"]),
        )

    @staticmethod
    def empty(): return TrainingResult()

    def plot(self, ax1_ylim=[0, None], ax2_ylim=[0, None], score_name=""):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        def plot_ax(ax: plt.Axes, lists: list[List], ylim, label=""):
            for arr in lists:
                if arr:
                    sns.lineplot(arr, label=arr.name + label, ax=ax)
            ax.set_ylim(ylim)
            ax.legend()

        plot_ax(ax1, [self.train_losses, self.valid_losses], ax1_ylim)
        plot_ax(ax2, [self.train_scores, self.valid_scores],
                ax2_ylim, label=f": {score_name}")


class OverfittingError(RuntimeError):
    def __init__(self, n, patience, *args: object) -> None:
        super().__init__(*args)
        self.epoch: int = n
        self.patience: int = patience


@torch.no_grad()
def eval(model, dl, loss_fn, metric):
    model.eval()
    losses = []
    ms = []
    for batch in dl:
        loss, m = model.train_step(loss_fn, metric, batch)
        ms.append(m)
        losses.append(loss.item())
    return np.mean(losses), np.mean(ms)


class MLP(nn.Module):
    def __init__(self, layers: list[int], act: Type[nn.Module] = nn.ReLU, p=.0, final_act=None, norm=False):
        super().__init__()
        self.layers = []
        for i, (inc, outc) in enumerate(zip(layers[:-1], layers[1:])):
            last = i == len(layers)-2
            self.layers.append(nn.Linear(inc, outc, bias=not last))
            if not last:
                if norm:
                    self.layers.append(nn.LayerNorm(outc))
                self.layers.append(act())
                self.layers.append(nn.Dropout(p))
        if final_act:
            self.layers.append(final_act())
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.seq(x)
