import os
import torch
import mlflow
import numpy as np
import pandas as pd
from torch import nn
import lightning as L
from functools import cached_property
from dataclasses import dataclass, asdict
from typing import Self, Type, Optional, Type, Literal, Optional, Callable

from .base import MLP,  MovieRecommender
from ..logging import Logger
from ..workflow import log_temp_artifacts, register_last_model_and_try_promote, model_uri

registered_name = os.environ["DLRM_REGISTERED_NAME"]


@dataclass(eq=True)
class DLRMParams:
    add_interactions: bool
    model_name: str
    lr: float
    n_users: int
    n_movies: int
    n_num_cols: int

    bot_layers: list[int]  # for numerical features.
    embds: list[tuple[int, int]]  # for categorical features.
    top_layers: list[int]  # classifier/regressor.

    cat_cols: list[str]
    num_cols: list[str]

    bot_p: float = .0  # dropput for bot MLp
    top_p: float = .0  # dropput for top MLp

    bot_use_batchnorm: bool = True
    top_use_batchnorm: bool = True

    bot_act: str = "LeakyReLU"
    top_act: str = "LeakyReLU"
    final_act: Optional[str] = "Sigmoid"

    as_regression: bool = True

    @classmethod
    def fromdict(cls, d: dict):
        return cls(**d)
    asdict = asdict


class FeatureInteraction(nn.Module):

    def forward(self, dense: list[torch.Tensor], sparse: list[torch.Tensor]) -> torch.Tensor:
        # inputs.shape = b,n,embd
        inputs = torch.stack(dense + sparse, dim=1)
        B, N, E = inputs.shape

        dot_products = torch.matmul(inputs, inputs.transpose(1, 2))
        indices = torch.triu_indices(N, N, offset=1, device=inputs.device)
        interactions = dot_products[:, indices[0], indices[1]]
        return interactions


class EmbeddingWithProjection(nn.Sequential):
    def __init__(self, i, o, proj, mode="mean"):
        layers = [nn.EmbeddingBag(i, o, mode=mode)]
        if proj is not None:
            layers += [nn.Linear(o, proj)]
        super().__init__(
            *layers
        )


def str_to_act_class(name) -> Type[nn.Module]:
    if name is None:
        return None  # type: ignore
    return {
        None: None,
        "silu": nn.SiLU,
        "LeakyReLU".lower(): nn.LeakyReLU,
        "ReLU".lower(): nn.ReLU,
        "Sigmoid".lower(): nn.Sigmoid,
    }[name.lower()]


class DLRM(nn.Module, MovieRecommender[dict[str, torch.Tensor]]):
    """
    Deep Learning Recommendation Model.
    """
    MAX_BATCH_SIZE = 256*8*4

    def __init__(self, params: DLRMParams) -> None:
        super().__init__()
        self.params = params
        n_cat_cols = len(params.embds)
        self.bot_mlp = MLP(
            [params.n_num_cols] + params.bot_layers,
            act=str_to_act_class(params.bot_act),
            p=params.bot_p,
            final_act=str_to_act_class(params.bot_act),
            norm=params.bot_use_batchnorm,
        )  # numerical features
        self.embds = nn.ModuleList([
            EmbeddingWithProjection(
                i,
                o,
                params.bot_layers[-1] if params.add_interactions else None,
                mode="mean",
            )
            for i, o in params.embds
        ])  # categorical features
        if params.add_interactions:
            self.interactions = FeatureInteraction()
            N = n_cat_cols + 1  # TODO
            dim = params.n_num_cols
            dim += N*N//2 - N//2
        else:
            dim = sum([o for _, o in params.embds]) + params.bot_layers[-1]
        self.top_mlp = MLP(
            [dim] + params.top_layers + [
                1 if params.as_regression else 11
            ],
            act=str_to_act_class(params.top_act),
            final_act=str_to_act_class(params.final_act),
            p=params.top_p,
            norm=params.top_use_batchnorm,
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        """
        num_f: numerical features.
        cat_f: categorical features.
        """
        num, cat = batch["num"], batch["cat"]
        x = self.bot_mlp.forward(num)
        y = []
        for i, embd, f in zip(range(len(self.embds)), self.embds, torch.unbind(cat, dim=1)):
            # (i, f.min(), f.max(), f.device)
            y.append(embd.forward(f.view(-1, 1)))
        if self.params.add_interactions:
            interactions = self.interactions.forward([x], y)
            out = torch.cat([interactions, num], dim=1)
        else:
            out = torch.cat([x]+y, dim=1)
        out = self.top_mlp.forward(out)
        return out.squeeze(1)

    def set_data(self, movies: pd.DataFrame, users: pd.DataFrame):
        self.movies = movies
        self.users = users

    @property
    def device_(self): return next(self.parameters()).device

    @classmethod
    def load(cls, champion=True, device="cpu") -> "DLRM":
        Logger.info("Loading dlrm on device: %s", device)
        return TrainableModule.load(champion=champion).model.to(device)

    def predict(self, batch: dict[str, torch.Tensor], max_rating):
        self.eval()
        # Logger.info("Dlrm batch length:", batch["num"].shape)
        with torch.no_grad():
            res = self.forward(batch).cpu()
            res *= max_rating
        return res.numpy()

    def _prepare_batch(
        self,
        user_ids: list[int],
        movie_ids_list: list[list[int]],
    ) -> list[dict[str, torch.Tensor]]:

        def prepare(user_id: int, movieIds: list[int]) -> pd.DataFrame:
            movie_data = self.movies.iloc[movieIds]
            user_data = self.users.iloc[[user_id]]
            movie_data.reset_index(inplace=True, drop=True)
            user_data.reset_index(inplace=True, drop=True)
            all = user_data.merge(movie_data, how="cross")
            return all

        num_cols = self.params.num_cols
        cat_cols = self.params.cat_cols

        datas = []
        for i, (user_id, movie_ids) in enumerate(zip(user_ids, movie_ids_list)):
            b = prepare(user_id, movie_ids)
            datas.append(b)
            # Logger.info(f"{i+1} request batch shape: {b.shape}")
        batch = pd.concat(datas, axis=0)
        num = torch.from_numpy(batch[num_cols].values).float()
        cat = torch.from_numpy(batch[cat_cols].values)

        batch_size = self.MAX_BATCH_SIZE
        return [dict(num=n.to(self.device_), cat=c.to(self.device_)) for n, c in zip(num.split(batch_size), cat.split(batch_size))]

    def log_artifacts(self):
        Logger.info("Logging artifacts.")

        def save(dir):
            self.movies.to_parquet(f"{dir}/movies.parquet")
            self.users.to_parquet(f"{dir}/users.parquet")
        log_temp_artifacts(save, artifact_path="resources")

    def _prepare_simple(self, user_ids, movie_ids):
        movie_data = self.movies.iloc[movie_ids]
        user_data = self.users.iloc[user_ids]
        movie_data.reset_index(inplace=True, drop=True)
        user_data.reset_index(inplace=True, drop=True)

        batch = pd.concat([user_data, movie_data], axis=1)

        num_cols = self.params.num_cols
        cat_cols = self.params.cat_cols

        num = torch.from_numpy(batch[num_cols].values).float().to(self.device_)
        cat = torch.from_numpy(batch[cat_cols].values).to(self.device_)
        return dict(num=num, cat=cat)


class TrainableModule(L.LightningModule):
    def __init__(self, model: DLRM, criterion: nn.Module | Callable, metrics: list[Callable[[np.ndarray, np.ndarray], tuple[str, float]]] = []):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.params_dict = self.model.params.asdict()  # type: ignore

        self.metrics__ = metrics

    @cached_property
    def numel(self): return sum([p.numel() for p in self.parameters()])

    @cached_property
    def device_(self): return next(self.parameters()).device

    def training_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        # self.log(
        #     "train_loss",
        # )
        with torch.no_grad():
            self.log_dict(
                {
                    "train_loss": loss,
                } | self.run_metrics_(logits, y, "train"),
                prog_bar=True,
                on_epoch=True,
                on_step=True,
            )
        return loss

    def run_metrics_(self, y_pred, y, tag):
        metrics = [metric(y_pred.cpu(), y.cpu()) for metric in self.metrics__]
        return {f"{tag}-{name}": value for name, value in metrics}

    def validation_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        self.log_dict({
            "val_loss": loss,
        } | self.run_metrics_(logits, y, "val"),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self):  # type: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.model.params.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def fit(
        self,
        dataloader,
        valloader,
        epochs,
        exp_name: str,
        seed=0,
        early_stopping=None,
        ModelCheckpoint_monitor=None,
        acc: Literal["gpu", "cpu", "auto"] = "gpu"
    ):
        L.seed_everything(seed)
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
        callbacks: list = [
            ModelCheckpoint(
                dirpath="./lightning_runs",
                monitor=ModelCheckpoint_monitor,
                filename="{epoch}-{val_loss:.2f}"
            ),
            LearningRateMonitor(),
            TQDMProgressBar(leave=True),
        ]

        if early_stopping is not None:
            monitor, patience = early_stopping
            callbacks.append(EarlyStopping(monitor, patience=patience))

        trainer = L.Trainer(
            callbacks=callbacks,
            accelerator=acc,
            deterministic=True,
            max_epochs=epochs,
        )
        mlflow.set_experiment(exp_name)
        mlflow.pytorch.autolog(log_every_n_step=1)  # type: ignore
        with mlflow.start_run(tags={"model_type": "DLRM"}) as run:
            import json
            run_id: str = run.info.run_id
            Logger.info("Run id: %s", run_id)
            mlflow.log_params({"model_params": json.dumps(self.params_dict)})
            trainer.fit(
                model=self,
                train_dataloaders=dataloader,
                val_dataloaders=valloader,
            )

        register_last_model_and_try_promote(
            registered_name=registered_name,
            metric_name="val-mae",
        )
        return run_id

    @classmethod
    def load(cls, champion=True) -> Self:
        # mlflow.artifacts.list_artifacts(f"models:/{model_name}@champion")
        return mlflow.pytorch.load_model(  # type: ignore
            model_uri(registered_name, champion)
        )
