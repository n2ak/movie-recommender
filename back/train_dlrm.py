import mlflow
import typing
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sqlalchemy import create_engine
from movie_recommender.modeling.dlrm import DLRM, TrainableModule, DLRMParams
from movie_recommender.data import MovieLens, get_cols

# try:
#     from .utils import Env, read_parquet, mae, rmse
# except ImportError:
#     from utils import Env, read_parquet, mae, rmse  # type: ignore

MAX_RATING = 5


def mae(logits, y) -> tuple[str, float]:
    max_rating = MAX_RATING
    y = y * max_rating
    logits = logits * max_rating
    return "mae", np.abs(logits - y).mean().item()


def rmse(logits, y) -> tuple[str, float]:
    return "rmse", np.sqrt(np.power(logits - y, 2).mean()).item()


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


def train_dlrm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int,
    exp_name: str,
    tracking_uri: str
):
    print("Starting dlrm training...")
    print("Train ds shape:", train.shape)
    print("Test ds shape:", test.shape)
    print("Epochs:", epochs)
    print(f"Mlflow {exp_name=}, {tracking_uri=}")
    mlflow.set_tracking_uri(tracking_uri)

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
        epochs=epochs,
        acc="gpu",
        log_mlflow=(exp_name, tracking_uri)
    )


DB_URL = 'postgresql+psycopg2://admin:password@host.docker.internal:5432/mydb'

T = typing.TypeVar("T")


def get_env(name: str, default: T) -> T:
    import os
    val = str(os.environ.get(name, default))
    if isinstance(default, bool):
        return val.lower() in ["true", "1"]  # type: ignore
    return default.__class__(val)  # type: ignore


def read_ds(table_name: str):
    engine = create_engine(DB_URL)
    table = pd.read_sql_table(table_name, engine)
    return table


def prepare(ds, cat_cols):
    batch, y = ds
    user_ids = batch["cat"][:, cat_cols.index("user_id")]
    movie_ids = batch["cat"][:, cat_cols.index("movie_id")]
    return np.array(user_ids), np.array(movie_ids), np.array(y)


def test_dlrm_model():
    from movie_recommender.recommender import Recommender, Request
    Recommender.instance = None
    Recommender.single(Request(
        userId=0,
        genres=[],
        model="dlrm_cpu",
        temp=0,
        start=0,
        count=10,
    ))
    print("DLRM model test passed successfully")


if __name__ == "__main__":
    train, test = read_ds("dlrm_train_ds"), read_ds("dlrm_test_ds")

    train_dlrm(
        train,
        test,
        epochs=get_env("EPOCHS", 2),
        exp_name=get_env("EXP_NAME", "movie_recom"),
        tracking_uri=get_env("MLFLOW_TRACKING_URI",
                             "http://host.docker.internal:8081"),
    )
