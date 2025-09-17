import pandas as pd


def movie_cols(df: pd.DataFrame):
    return list(filter(lambda c: c.startswith("movie"), df.columns))


def user_cols(df: pd.DataFrame):
    return list(filter(lambda c: c.startswith("user"), df.columns))
