import torch
import numpy as np
import pandas as pd
from torch import nn
from dataclasses import dataclass
from .base import BaseModel, BaseParams, MLP
from ..utils import exclude


@dataclass(eq=True)
class NCFParams(BaseParams):
    nu: int  # number of users
    nm: int  # number of movies
    u_embd: int  # user embeddings
    m_embd: int  # movie embeddings
    genres_embd: int
    hidden_layers: list[int]  # layer
    p: float = 0  # dropput used for mlp


class NCF(BaseModel[NCFParams]):
    """
    Neural Collaborative Filtering
    """

    def init(self) -> None:
        params = self.config.params
        self.u_emb = nn.Embedding(params.nu, params.u_embd)
        self.m_emb = nn.Embedding(params.nm, params.m_embd)
        self.genres_emb = nn.Embedding(2, params.genres_embd)
        embed_dim = params.u_embd + params.m_embd + params.genres_embd
        self.mlp = MLP(
            [embed_dim] + params.hidden_layers + [1],
            act=nn.LeakyReLU,
            final_act=nn.Sigmoid,
            p=params.p,
        )

    def forward(self, users, movies, genres):
        u_emb = self.u_emb(users)
        m_emb = self.m_emb(movies)
        genres_emb = self.genres_emb.forward(genres).sum(1)
        emb = torch.concat([u_emb, m_emb, genres_emb], dim=1)
        res: torch.Tensor = self.mlp(emb)
        return res.squeeze()

    def _predict(self, users, movies, genres, **_) -> torch.Tensor:
        users = torch.tensor(users, device=self.device_)
        movies = torch.tensor(movies, device=self.device_)
        genres = torch.tensor(genres, device=self.device_)
        res = self.forward(users, movies, genres)
        return res

    def prepare(self, user, movieIds=None):
        if self.ds_state is None:
            self.ds_state = NCFState(**self.config.extra)
        ds_state = self.ds_state
        assert user < ds_state.n_users
        if movieIds is None:
            movieIds = np.arange(ds_state.n_movies)
        else:
            movieIds = np.array(movieIds)

        movie_data = ds_state.movies.loc[ds_state.movies.movie.isin(movieIds)]
        nmovies = movie_data.shape[0]
        userIds = np.array([user] * nmovies)

        genres = movie_data[exclude(
            ds_state.movie_cat_cols, "movie", "year_bin")
        ].values.astype(np.int32)
        return dict(
            users=userIds,
            movies=movieIds,
            genres=genres,
        )


@dataclass
class NCFState:
    n_users: int
    n_movies: int
    # user_cat_cols: list[str]
    movie_cat_cols: list[str]
    movies_path: str
    users_path: str

    __movies: pd.DataFrame = None
    __users: pd.DataFrame = None

    @property
    def movies(self):
        if self.__movies is None:
            from ..utils import read_csv
            self.__movies = read_csv(self.movies_path)
        return self.__movies

    @property
    def users(self):
        if self.__users is None:
            from ..utils import read_csv
            self.__users = read_csv(self.users_path)
        return self.__users

    def to_dict(self):
        return dict(
            n_users=self.n_users,
            n_movies=self.n_movies,
            movie_cat_cols=self.movie_cat_cols,
            movies_path=self.movies_path,
            users_path=self.users_path,
            # user_cat_cols=self.user_cat_cols,
            # user_num_cols=self.user_num_cols,
            # movie_num_cols=self.movie_num_cols,
        )
