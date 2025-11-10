import pandas as pd
import numpy as np
from typing import Literal


def movie_cols(df: pd.DataFrame):
    return list(filter(lambda c: c.startswith("movie"), df.columns))


def user_cols(df: pd.DataFrame):
    return list(filter(lambda c: c.startswith("user"), df.columns))


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
    return train, test
