import torch
import numpy as np
import pandas as pd
from torch import nn
from .base import BaseModel, BaseParams, MLP
from dataclasses import dataclass


@dataclass(eq=True)
class DLRMParams(BaseParams):
    bot_layers: list[int]  # for numerical features.
    embds: list[tuple[int, int]]  # for categorical features.
    top_layers: list[int]  # classifier/regressor.
    bot_p: float = .0  # dropput for bot MLp
    top_p: float = .0  # dropput for top MLp


class DLRM(BaseModel[DLRMParams]):
    """
    Deep Learning Recommendation Model.
    """

    def init(self) -> None:
        params = self.config.params
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
        cat = torch.tensor(cat, device=self.device_)
        res = self.forward(num, cat)
        return res

    def prepare(
        self,
        user,
        movieIds=None,
    ):
        from ..data import MovieLens
        from ..utils import exclude
        movie_lens = MovieLens.get_instance()
        assert movie_lens.n_users == self.config.n_users
        assert user < movie_lens.n_users
        if movieIds is None:
            movieIds = np.arange(movie_lens.n_movies)
        else:
            movieIds = np.array(movieIds)

        movie_data = movie_lens.movies.loc[movie_lens.movies.movie.isin(
            movieIds)]
        users_data = movie_lens.users.loc[movie_lens.users.user == user]
        nmovies = movie_data.shape[0]
        userIds = np.array([user] * nmovies)

        def dup(user_cols, movie_cols):
            a = users_data[user_cols]
            a = np.repeat(a, nmovies, axis=0)
            return np.concatenate((a, movie_data[movie_cols].values), axis=1)
        user_cat_cols = exclude(movie_lens.user_cat_cols, "usedr")
        cat = dup(movie_lens.user_cat_cols,
                  movie_lens.movie_cat_cols).astype(np.int32)
        num = dup(movie_lens.user_num_cols,
                  movie_lens.movie_num_cols).astype(np.float32)
        return dict(
            users=userIds,
            movies=movieIds,
            num=num,
            cat=cat,
        )
