import pandas as pd
import numpy as np
from typing import Literal


def movie_cols(df: pd.DataFrame):
    return list(filter(lambda c: c.startswith("movie"), df.columns))


def user_cols(df: pd.DataFrame):
    return list(filter(lambda c: c.startswith("user"), df.columns))


def get_cols(ratings: pd.DataFrame):
    import numpy as np
    cat_cols = ratings[:1].select_dtypes(
        [np.integer, "bool", "category"]).columns.to_list()
    num_cols = ratings[:1].select_dtypes([np.floating]).columns.to_list()
    if "rating" in num_cols:
        num_cols.remove("rating")

    ratings[num_cols] = ratings[num_cols].astype("float32")
    return cat_cols, num_cols


def get_n_cat_col(users_features, movies_features):
    unique = pd.concat(
        [
            movies_features[get_cols(movies_features)[0]].max(),
            users_features[get_cols(users_features)[0]].max()
        ]
    )+1
    return unique


def preprocess_data(
    ratings: pd.DataFrame,
    user_features: pd.DataFrame,
    movie_features: pd.DataFrame,
    train_size: float,
):
    assert 0 < train_size <= 1
    ratings = ratings.merge(user_features, on="user_id").merge(
        movie_features, on="movie_id")
    ratings = ratings.sample(frac=1)

    train_size = int(ratings.shape[0] * train_size)
    train, test = ratings[:train_size], ratings[train_size:]

    unique = get_n_cat_col(user_features, movie_features)
    return train, test, unique
