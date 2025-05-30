import tqdm
from dataclasses import dataclass, asdict
from typing import Callable
from core.data import MovieLens
import torch
import numpy as np
import pandas as pd
from torch import nn
from .base import MLP, OverfittingError, TrainingResult, eval
from dataclasses import dataclass
from typing import Optional


@dataclass(eq=True)
class DLRMParams:
    model_name: str
    lr: float
    n_users: int
    n_movies: int

    bot_layers: list[int]  # for numerical features.
    embds: list[tuple[int, int]]  # for categorical features.
    top_layers: list[int]  # classifier/regressor.
    bot_p: float = .0  # dropput for bot MLp
    top_p: float = .0  # dropput for top MLp
    asdict = asdict

    @classmethod
    def fromdict(cls, d: dict):
        return cls(**d)


class DLRM(nn.Module):
    """
    Deep Learning Recommendation Model.
    """

    def __init__(self, params: DLRMParams) -> None:
        super().__init__()
        self.params = params
        self.bot_mlp = MLP(
            params.bot_layers,
            act=nn.LeakyReLU,
            p=params.bot_p,
        )  # numerical features
        self.embds = nn.ModuleList([
            nn.EmbeddingBag(i, o, mode="sum") for i, o in params.embds
        ])  # categorical features
        dim = sum([o for _, o in params.embds]) + params.bot_layers[-1]
        self.top_mlp = MLP(
            [dim] + params.top_layers + [1],
            act=nn.LeakyReLU,
            final_act=nn.Sigmoid,
            p=params.top_p,
        )  # output

    def forward(self, num, cat):
        """
        num_f: numerical features.
        cat_f: categorical features.
        """
        x = self.bot_mlp.forward(num)
        y = []
        for i, embd, f in zip(range(len(self.embds)), self.embds, torch.unbind(cat, dim=1)):
            # print(i, f.min(), f.max())
            y.append(embd.forward(f.view(-1, 1)))
        out = torch.concat([x] + y, dim=1)
        out: torch.Tensor = self.top_mlp.forward(out)

        return out.squeeze(1)

    def _predict(self, num, cat, **_) -> torch.Tensor:
        num = torch.tensor(num, device=self.device_)
        cat = torch.tensor(cat, device=self.device_, dtype=torch.long)
        res = self.forward(num, cat)
        return res

    def prepare(self, userIds, movieIds, cross: bool):
        assert isinstance(userIds, (list, np.ndarray)), type(userIds)
        assert isinstance(movieIds, (list, np.ndarray)), type(movieIds)
        if not cross:
            assert len(userIds) == len(movieIds)
        movielens = MovieLens.get_instance()
        movies = movielens.movies
        users = movielens.users
        num_cols = movielens.num_cols
        cat_cols = movielens.cat_cols
        movie_data = movies.iloc[movieIds]
        user_data = users.iloc[userIds]
        if not cross:
            assert len(user_data) == len(
                movie_data), np.setxor1d(movie_data.movieId.to_list(), movieIds)

        movie_data.reset_index(inplace=True, drop=True)
        user_data.reset_index(inplace=True, drop=True)
        if cross:
            all = user_data.merge(movie_data, how="cross")
        else:
            all = pd.concat([user_data, movie_data], axis=1)
        if all.isna().values.mean() != 0:
            from core.utils import display
            display(all)
            assert False
        return dict(
            num=all[num_cols].values,
            cat=all[cat_cols].values,
        )

    def save(
        self,
        filename: str,
        optimizer=None,
        last_training_result: Optional[TrainingResult] = None,
    ):
        assert "models" not in filename
        from ..utils import get_models_path
        path = get_models_path()/filename
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.state_dict(),
            "params": self.params.asdict(),
            "optim": optimizer.state_dict() if optimizer is not None else None,
            "last_training_result": last_training_result.asdict() if last_training_result is not None else None,
        }
        torch.save(state, path)
        print("Model saved.")

    @classmethod
    def load(cls, filename, model_only=False, device="cpu"):

        assert "models" not in filename, filename
        from ..utils import get_models_path
        filename = get_models_path()/filename
        state = torch.load(filename, weights_only=False)
        # print("Kyes", list(state.keys()))
        config = DLRMParams.fromdict(state["params"])
        model = cls(config)
        model.load_state_dict(state["model"])
        model = model.to(device=device)
        if model_only:
            return model
        last_training_result = TrainingResult.fromdict(
            state["last_training_result"])
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(state["optim"])
        return model, optimizer, last_training_result

    @property
    def device_(self): return next(self.parameters()).device

    def predict(self, sort=True, clamp=None, max_rating=10, **input):
        self.eval()
        with torch.no_grad():
            # TODO add batches
            res = self._predict(**input) * max_rating
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
        max_rating,
        wrap=False,
    ) -> list["Recommendation"]:
        assert isinstance(userId, int), type(userId)
        return self.recommend_for_users(
            userIds=[userId],
            movieIds=movieIds,
            cross=True,
            wrap=wrap,
            max_rating=max_rating,
        )

    def recommend_for_users(
        self,
        userIds,
        movieIds,
        max_rating,
        cross=False,
        wrap=False,
    ) -> list["Recommendation"]:
        input = self.prepare(userIds=userIds, movieIds=movieIds, cross=cross)
        ratings, ids = self.predict(sort=False, max_rating=max_rating, **input)
        ratings = ratings.tolist()
        if cross and len(userIds) == 1:
            userIds = userIds * len(movieIds)
        userIds, movieIds = MovieLens.get_instance().map_ids(
            userIds, movieIds, reverse=True)
        if not wrap:
            return ratings
        return self.to_pred(
            userIds,
            movieIds,
            ratings,
        )

    def to_pred(self, userIds, movieIds, preds):
        assert isinstance(userIds, list)
        assert isinstance(movieIds, list)
        assert isinstance(preds, list)
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
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        val_ld=None,
        last_training_result: Optional[TrainingResult] = None,
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

    def numel(self): return sum([p.numel() for p in self.parameters()])


@dataclass(eq=True)
class Recommendation:
    userId: int
    movieId: int
    predicted_rating: float
    asdict = asdict
