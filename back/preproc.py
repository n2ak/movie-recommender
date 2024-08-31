import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict


def matrix_from_file(nrows, limit=None):
    movies = pd.read_csv("dataset/ml-32m/movies.csv", nrows=nrows)
    ratings = pd.read_csv("dataset/ml-32m/ratings.csv", nrows=nrows)
    links = pd.read_csv("dataset/ml-32m/links.csv", nrows=nrows)
    tags = pd.read_csv("dataset/ml-32m/tags.csv", nrows=nrows)
    ratings = ratings.merge(movies, on=["movieId"]).sort_values(
        "movieId").reset_index(drop=True)
    ratings = ratings.merge(links, on=["movieId"]).reset_index(drop=True)
    ratings = encode(ratings, ["userId", "movieId"], False)
    if limit is not None:
        ratings = ratings.iloc[:limit]
    rating_matrix = ratings.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)
    real_matrix = torch.from_numpy(rating_matrix.values)
    return real_matrix, ratings


def encode(df, cols, reset=False):
    lencoders = defaultdict(LabelEncoder)
    for c in cols:
        lencoders[c].fit(df[c].unique())
        df[c] = lencoders[c].transform(df[c])
    if reset:
        df = df.sort_values("userId").reset_index(drop=True)
    return df
