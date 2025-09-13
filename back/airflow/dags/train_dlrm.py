from functools import partial
from movie_recommender.modeling.workflow import save_plots
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from movie_recommender.modeling.dlrm import DLRM, TrainableModule, DLRMParams
from movie_recommender.data import MovieLens, get_cols

try:
    from .utils import Env, read_parquet, mae, rmse
except ImportError:
    from utils import Env, read_parquet, mae, rmse  # type: ignore


def create_data_sampler(y_train: np.ndarray):
    class_counts = np.bincount((y_train*10).astype(int), minlength=11)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights[0] = 0
    sampler = torch.utils.data.WeightedRandomSampler(
        class_weights[(y_train*10).astype(int)].tolist(), y_train.size,
    )
    return sampler


def process_data(train: pd.DataFrame, test: pd.DataFrame):
    cat_cols, num_cols = get_cols(train)
    unique = pd.concat([train, test], axis=0)[cat_cols].nunique()
    nmovies = unique[["movie_id"]].item()
    nusers = unique[["user_id"]].item()
    embds = np.ceil(np.array(unique) ** .5)
    embds = np.clip(np.exp2(np.ceil(np.log2(embds))),
                    2, 64).astype(int).tolist()
    return unique, cat_cols, num_cols, embds, nmovies, nusers


def create_data_loaders(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str],
    num_cols: list[str],
    unique: pd.Series,
):
    batch_size = 64 * 4

    for c, u in zip(cat_cols, unique):
        assert train[c].max() < u, c
        assert test[c].max() < u, c
    train_ds = DS(
        train[num_cols].values,
        train[cat_cols].values,
        train["rating"].values,  # type: ignore
    )
    test_ds = DS(
        test[num_cols].values,
        test[cat_cols].values,
        test["rating"].values,  # type: ignore
    )
    # valid_ds, test_ds = torch.utils.data.random_split(DS(
    #     test[num_cols].values,
    #     test[cat_cols].values,
    #     test.rating.values,
    # ), [.5, .5], generator=torch.Generator().manual_seed(0))
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, valid_dl, train_ds, test_ds


class DS(torch.utils.data.Dataset):
    def __init__(
        self,
        num: np.ndarray,
        cat: np.ndarray,
        y: np.ndarray,
    ) -> None:
        super().__init__()
        assert len(num) == len(cat) == len(y)

        self.num = num
        self.cat = cat
        self.y = y

    def __len__(self):
        l = self.num.shape[0]
        return l

    def __getitem__(self, index):
        num = self.num[index].astype(np.float32)
        cat = self.cat[index]
        y = self.y[index].astype(np.float32)
        return dict(num=num, cat=cat), y


def create_model(
    cat_cols: list[str],
    num_cols: list[str],
    nusers: int,
    nmovies: int,
    unique: pd.Series,
    embds: list[int],
):
    criterion = F.mse_loss
    model = TrainableModule(
        DLRM(
            DLRMParams(
                add_interactions=True,
                model_name="DLRM",
                n_num_cols=len(num_cols),
                lr=.01,
                n_users=nusers,
                n_movies=nmovies,
                bot_layers=[128, 128],
                embds=[(n, e) for n, e in zip(unique, embds)],
                top_layers=[128, 128],
                bot_p=0.3,
                top_p=0.3,
                final_act=None,
                top_act="SiLU",
                bot_act="SiLU",
                top_use_batchnorm=False,
                as_regression=True,
                cat_cols=cat_cols,
                num_cols=num_cols,
            ),
        ),
        criterion,
        metrics=[mae, rmse]
    )
    return model


def split_(df: pd.DataFrame):
    movie_cols = MovieLens.movie_cols(df)
    user_cols = MovieLens.user_cols(df)
    movies = df[movie_cols].drop_duplicates("movie_id")
    users = df[user_cols].drop_duplicates("user_id")
    return users, movies


def train_dlrm(train_path: str, test_path: str):
    Env.dump()

    train, test = read_parquet(train_path, test_path)

    unique, cat_cols, num_cols, embds, nmovies, nusers = process_data(
        train, test
    )
    train_dl, valid_dl, train_ds, test_ds = create_data_loaders(
        train, test, cat_cols, num_cols, unique
    )
    model = create_model(
        cat_cols, num_cols, nusers, nmovies, unique, embds
    )
    users, movies = split_(pd.concat([train, test], axis=0))
    model.model.set_data(movies, users)
    run_id = model.fit(
        train_dl,
        valid_dl,
        epochs=Env.EPOCH,
        acc="auto",
        log_mlflow=(Env.exp_name, Env.MLFLOW_TRACKING_URI)
    )
    p = partial(prepare, cat_cols=cat_cols)
    # save_plots(model.model, p, train_ds[:], test_ds[:],
    #            Env.MAX_RATING, run_id=run_id)


def prepare(ds, cat_cols):
    batch, y = ds
    user_ids = batch["cat"][:, cat_cols.index("user_id")]
    movie_ids = batch["cat"][:, cat_cols.index("movie_id")]
    return np.array(user_ids), np.array(movie_ids), np.array(y)
