import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from movie_recommender.common.logging import Logger
from training.data import movie_cols, user_cols, preprocess_data, movie_cols, user_cols
from training.train_utils import mae, rmse, get_env
from movie_recommender.dlrm.dlrm import DLRM, TrainableModule, DLRMParams
from movie_recommender.common.workflow import StorageClient


def create_data_sampler(y_train: np.ndarray):
    class_counts = np.bincount((y_train*10).astype(int), minlength=11)
    class_weights = 1.0 / class_counts
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights[0] = 0
    sampler = torch.utils.data.WeightedRandomSampler(
        class_weights[(y_train*10).astype(int)].tolist(), y_train.size,
    )
    return sampler


def get_cols(ratings: pd.DataFrame):
    cat_cols = ratings.iloc[:1].select_dtypes(
        ["int", "bool", "category"]).columns.to_list()
    num_cols = ratings.iloc[:1].select_dtypes("float").columns.to_list()
    if "rating" in num_cols:
        num_cols.remove("rating")
    ratings[cat_cols] = ratings[cat_cols]
    ratings[num_cols] = ratings[num_cols].astype("float32")
    return cat_cols, num_cols


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
    batch_size=64 * 4,
):

    for c, u in zip(cat_cols, unique):
        assert train[c].max() < u, (c, train[c].max(), u)
        assert test[c].max() < u,  (c, test[c].max(), u)
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
    movies = df[movie_cols(df)].drop_duplicates("movie_id")
    users = df[user_cols(df)].drop_duplicates("user_id")
    return users, movies


def train_dlrm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    epochs: int,
    exp_name: str,
    batch_size: int,
):
    Logger.info("****************Starting dlrm training...**************")
    Logger.info("Train ds shape: %s", train.shape)
    Logger.info("Test ds shape: %s", test.shape)
    Logger.info("Epochs: %s", epochs)
    Logger.info("Batch size: %s", batch_size)

    unique, cat_cols, num_cols, embds, nmovies, nusers = process_data(
        train, test
    )
    train_dl, valid_dl, train_ds, test_ds = create_data_loaders(
        train, test, cat_cols, num_cols, unique,
        batch_size=batch_size
    )
    model = create_model(
        cat_cols, num_cols, nusers, nmovies, unique, embds
    )
    users, movies = split_(pd.concat([train, test], axis=0))
    run_id = model.fit(
        train_dl,
        valid_dl,
        epochs=epochs,
        acc="auto",
        exp_name=exp_name,
    )
    Logger.info("****************Training is done*******************")
    model.model.eval()
    # save_plots(
    #     model.model,
    #     prepare(train_ds[:], cat_cols),
    #     prepare(test_ds[:], cat_cols),
    #     max_rating=5,
    #     run_id=run_id,
    # )
    Logger.info("****************Saved plots*******************")


def prepare(ds, cat_cols):
    batch, y = ds
    user_ids = batch["cat"][:, cat_cols.index("user_id")]
    movie_ids = batch["cat"][:, cat_cols.index("movie_id")]
    return np.array(user_ids), np.array(movie_ids), np.array(y)


# def test_dlrm_model():
#     DLRM.load(champion=False, device="cuda").recommend_for_users_batched(
#         [0, 1],
#         movieIds=[[0, 1], [10, 30]],
#         max_rating=5,
#         clamp=True,
#         temps=[0.1, 0.3]
#     )
#     Logger.info("DLRM model test passed successfully")


if __name__ == "__main__":
    import sys
    import os
    arg = sys.argv[1]
    bucket = os.environ["DB_MINIO_BUCKET"]

    if arg == "train":
        train, test = StorageClient.get_instance().download_parquet_from_bucket(
            bucket, "dlrm_train", "dlrm_test")

        train_dlrm(
            train,
            test,
            epochs=get_env("EPOCHS", 2),
            exp_name=get_env("EXP_NAME", "movie_recom"),
            batch_size=get_env("BATCH_SIZE", 64 * 4)
        )
    elif arg == "test":
        # test_dlrm_model()
        pass
    elif arg == "preprocess":
        ratings, movies = StorageClient.get_instance().download_parquet_from_bucket(
            bucket, "ratings", "movies")
        train, test = preprocess_data(
            ratings,
            movies,
            max_rating=int(os.environ["MAX_RATING"]),
            train_size=float(os.environ["TRAIN_SIZE"]),
        )
        StorageClient.get_instance().upload_parquet_to_bucket(
            bucket, dlrm_train=train, dlrm_test=test
        )
    else:
        Logger.error(f'Invalid arg {arg}')
        sys.exit(1)
