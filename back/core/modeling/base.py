import tqdm
import torch
import pathlib
import numpy as np
from torch import nn
from typing import Generic, TypeVar
from typing import Callable
from dataclasses import dataclass, asdict, field

T = TypeVar('T')


def get_and_del(obj: dict, k: str):
    o = obj[k]
    del obj[k]
    return o


class BaseParams:
    asdict = asdict

    @staticmethod
    def fromdict(d: dict):
        raise NotImplementedError("")


@dataclass(eq=True)
class Config(Generic[T], BaseParams):
    model_name: str
    lr: float
    params: T
    n_users: int
    n_movies: int

    @staticmethod
    def fromdict(d: dict):
        from .mf import MFParams
        from .dlrm import DLRMParams
        from .ncf import NCFParams
        PARAMS = classes_dict(
            [MFParams, DLRMParams, NCFParams],
            names=["MF", "DLRM", "NCF"],
        )
        lr = d["lr"]
        params = d["params"]
        model_name = d["model_name"]
        ParamsCls = PARAMS[model_name]
        # print(ParamsCls, model_name)
        return Config(
            lr=lr,
            model_name=model_name,
            params=ParamsCls(**params),
            n_movies=d["n_movies"],
            n_users=d["n_users"],
        )


@dataclass(eq=True)
class Recommendation:
    userId: int
    movieId: int
    predicted_rating: float


class L(list[T]):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name


@dataclass
class TrainingResult:
    train_losses: L[int] = field(default_factory=lambda: L("train loss"))
    valid_losses: L[int] = field(default_factory=lambda: L("valid loss"))

    # f1/rmse for example
    train_scores: L[float] = field(default_factory=lambda: L("train score"))
    valid_scores: L[float] = field(default_factory=lambda: L("valid score"))

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
            L("train loss", d["train_losses"]),
            L("valid loss", d["valid_losses"]),
            L("train score", d["train_scores"]),
            L("valid score", d["valid_scores"]),
        )

    @staticmethod
    def empty(): return TrainingResult()

    def plot(self, ax1_ylim=[0, None], ax2_ylim=[0, None], score_name=""):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        def plot_ax(ax: plt.Axes, lists: list[L], ylim, label=""):
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


class BaseModel(nn.Module, Generic[T]):
    ds_state = None

    def __init__(self, params: Config[T]) -> None:
        super().__init__()
        self.config = params
        self.init()

    def save(
        self,
        filename: str,
        optimizer=None,
        last_training_result: TrainingResult = None,
    ):
        assert "models" not in filename
        from ..utils import get_models_path
        path = get_models_path()/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.state_dict(),
            "config": self.config.asdict(),
            "optim": optimizer.state_dict() if optimizer is not None else None,
            "last_training_result": last_training_result.asdict() if last_training_result is not None else None,
        }
        torch.save(state, path)
        print("Model saved.")

    @classmethod
    def load(cls, filename, model_only=False):
        assert "models" not in filename, filename
        from ..utils import get_models_path
        filename = get_models_path()/filename
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state: dict = torch.load(
            filename, map_location=torch.device(device), weights_only=False)
        model, optimizer = BaseModel.fromconfig(
            state["config"],
            load_optim=(state["optim"] is not None or (not model_only)),
        )
        model.load_state_dict(state["model"])
        if model_only:
            return model
        if optimizer is not None and state["optim"]:
            optimizer.load_state_dict(state["optim"])
        if state.get("last_training_result", None) is not None:
            last_training_result = TrainingResult.fromdict(
                state["last_training_result"]
            )
        else:
            last_training_result = TrainingResult.empty()
        print(f"Model loaded.")
        return model, optimizer, last_training_result

    @property
    def device_(self): return next(self.parameters()).device

    def _predict(self, *_, **__) -> torch.Tensor: raise NotImplementedError("")

    def predict(self, sort=True, clamp=None, **input) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            # TODO add batches
            res = self._predict(**input) * 5
            if clamp:
                min, max = clamp
                res = torch.clamp(res, min, max)
            if sort:
                return res.cpu().sort(descending=True)
            return res.cpu(), torch.arange(res.shape[0])

    def recommend_for_user(
        self,
        userId,
        movieIds,
        start=0,
        count=None,
        **other,
    ) -> list[Recommendation]:
        count = count or len(movieIds)
        return self.recommend_for_users(
            users=[userId]*count,
            movies=movieIds,
            start=start,
            count=count
            ** other,
        )

    def prepare(self, *_, **__) -> dict: raise NotImplementedError("")

    def recommend_for_users(
        self,
        start=0,
        count=None,
        wrap=True,
        sort=False,
        **other,
    ) -> list[Recommendation]:
        userIds = other["users"]
        movieIds = other["movies"]
        count = count or len(movieIds)
        ratings, ids = self.predict(**other, sort=sort)
        end = start + count if count is not None else None
        assert ratings.ndim == 1, ratings.shape
        ratings = ratings[start:end].flatten().tolist()
        movieIds = movieIds[ids[start:end].numpy()].tolist()
        # print(sort, movieIds)

        if not wrap:
            return ratings
        return self.to_pred(
            userIds.tolist(),
            movieIds,
            ratings,
        )

    def to_pred(self, userIds, movieIds, preds):
        assert len(userIds) == len(movieIds) == len(
            preds), (len(userIds), len(movieIds), len(preds))
        return [Recommendation(u, m, p) for u, m, p in zip(userIds, movieIds, preds)]

    def train_step(self, lossfn, metric, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        self.train()

        def split():
            y = batch["y"]
            del batch["y"]
            return batch, y

        X, y = split()
        y = y.to(self.device_)
        for key in X:
            X[key] = X[key].to(self.device_)
        pred = self.forward(**X)
        loss = lossfn(pred.squeeze(), y)
        with torch.no_grad():
            score = metric(y.cpu(), pred.detach().cpu())
        return loss, score

    def fit(
        self,
        optimizer: torch.optim.Optimizer,
        dl,
        loss_fn: Callable,
        metric: Callable,
        epochs: int,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        val_ld=None,
        last_training_result: TrainingResult = None,
        patience=None,  # patience for monitoring overfitting
    ):
        if val_ld is None:
            val_loss = np.nan
        if last_training_result is None:
            last_training_result = TrainingResult.empty()
        lr = [group['lr'] for group in optimizer.param_groups]
        last_val_loss = np.finfo(np.float32).max
        n_steps_since_last_improv = 0
        try:
            lowest_l = 9999999999
            lowest_e = 9999999999
            for epoch in (_ := range(
                    len(last_training_result.train_losses),
                    len(last_training_result.train_losses)+epochs,
            )):
                l, scores = [], []

                bar = tqdm.tqdm(dl)
                for i, batch in enumerate(bar):
                    last_step = i == (len(bar)-1)

                    self.train()
                    loss, score = self.train_step(loss_fn, metric, batch)

                    l.append(loss.item())
                    scores.append(score)
                    optimizer.zero_grad()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)

                    loss.backward()
                    optimizer.step()

                    if last_step:
                        self.evaluate(val_ld, loss_fn, metric,
                                      last_training_result, epoch, l, scores, scheduler, bar)
                    else:
                        bar.set_description(
                            f"Epoch:{epoch+1:3} Loss : {l[-1]:.4f}")

                last_training_result.train_losses.append(np.mean(l))
                last_training_result.train_scores.append(np.mean(scores))

                # check for overfitting
                if val_ld:
                    val_loss = last_training_result.valid_losses[-1]
                    if patience is not None:
                        if val_loss < last_val_loss:
                            last_val_loss = val_loss
                            n_steps_since_last_improv = 0
                        else:
                            n_steps_since_last_improv += 1
                        if n_steps_since_last_improv > patience:
                            raise OverfittingError(epoch+1, patience)

        except KeyboardInterrupt:
            pass
        except OverfittingError as e:
            print(
                f"No improvements on validation in the last {e.patience} epochs, "
                f"Stopping at epoch {e.epoch}."
            )
        return last_training_result

    def evaluate(self, val_ld, loss_fn, metric, last_training_result, epoch, l, scores, scheduler, bar: tqdm.tqdm):
        if val_ld is not None:
            val_loss, val_m = eval(self, val_ld, loss_fn, metric)
            last_training_result.valid_losses.append(val_loss)
            last_training_result.valid_scores.append(val_m)

        # if last_training_result.train_losses[-1] < lowest_l:
        #     lowest_l = last_training_result.train_losses[-1]
        #     lowest_e = epoch

        # scheduler.step(val_loss if val_ld is not None else losses[-1])
        if scheduler is not None:
            scheduler.step(epoch)
            lr = scheduler.get_last_lr()
        # bar.set_description(
        #     f"Epoch:{epoch+1:3},LR:{lr[0]:.6f}/{epoch-lowest_e},Loss : {last_training_result.train_losses[-1]:.4f}/{lowest_l:.4f}, "
        #     f"Val loss: {(val_loss):.4f}")
        bar.set_description(
            f"Epoch:{epoch+1:3}, loss : {l[-1]:.4f}, "
            f"val_loss: {(val_loss):.4f}")

    @classmethod
    def init_model(cls, config: Config) -> "BaseModel":
        from .mf import MF
        from .ncf import NCF
        from .dlrm import DLRM
        MODELS = classes_dict([MF, DLRM, NCF])
        ModelClass = MODELS[config.model_name]
        return ModelClass(config)

    @staticmethod
    def fromconfig(conf: dict, load_optim=False):
        config = Config.fromdict(conf)
        model = BaseModel.init_model(config)
        optimizer = None
        if load_optim:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        return model, optimizer

    def numel(self): return sum([p.numel() for p in self.parameters()])


def classes_dict(classes: list, names: list = None):
    if names is None:
        names = [c.__name__ for c in classes]
    assert len(names) == len(classes), (classes, names)
    return dict(zip(names, classes))


@torch.no_grad()
def eval(model: BaseModel, dl, loss_fn, metric):
    model.eval()
    losses = []
    ms = []
    for batch in dl:
        loss, m = model.train_step(loss_fn, metric, batch)
        ms.append(m)
        losses.append(loss.item())
    return np.mean(losses), np.mean(ms)


class MLP(nn.Module):
    def __init__(self, layers: list[int], act=nn.ReLU, p=0, final_act=None, norm=False):
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
