import pandas as pd
from sklearn.model_selection import train_test_split
import typing
import numpy as np

MAX_RATING = 5

T = typing.TypeVar("T")


def mae(pred, y) -> tuple[str, float]:
    max_rating = MAX_RATING
    y = y * max_rating
    pred = pred * max_rating
    return "mae", np.abs(pred - y).mean().item()


def rmse(logits, y) -> tuple[str, float]:
    return "rmse", np.sqrt(np.power(logits - y, 2).mean()).item()


def get_env(name: str, default: T) -> T:
    import os
    val = str(os.environ.get(name, default))
    if isinstance(default, bool):
        return val.lower() in ["true", "1"]  # type: ignore
    return default.__class__(val)  # type: ignore


def fix_split(all, train, test):
    test = test[test['user_id'].isin(train['user_id'])]
    test = test[test['movie_id'].isin(train['movie_id'])]
    excluded = all.loc[~all.index.isin(
        train.index) & ~all.index.isin(test.index)]
    train = pd.concat([train, excluded])
    return train, test


def simple_split(ratings: pd.DataFrame, train_size):
    train, test = train_test_split(
        ratings, train_size=train_size, random_state=0)
    return fix_split(ratings, train, test)


def per_user_split(df: pd.DataFrame, train_size):
    train_list, test_list = [], []

    for uid, user_ratings in df.groupby("user_id"):
        if len(user_ratings) < 2:
            train_list.append(user_ratings)
            continue

        user_train, user_test = train_test_split(
            user_ratings, train_size=train_size, random_state=42)
        train_list.append(user_train)
        test_list.append(user_test)

    train = pd.concat(train_list)
    test = pd.concat(test_list)
    test = test[test['movie_id'].isin(train['movie_id'])]
    return train, test
