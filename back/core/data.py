import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import torch
import re


def preprocess_year(movies: pd.DataFrame, encode_year=True):
    exp = re.compile("^.+\((\d+)\)$")

    def get_year(title):
        matches = exp.match(title.strip())
        if matches is None:
            return np.nan
        groups = matches.groups()
        assert len(groups) == 1
        return int(groups[0])
    bins = 7
    year = movies.title.map(get_year)
    if not encode_year:
        edges = np.linspace(year.min(), year.max(), bins+1).astype(int)
        labels = [f'({edges[i]}, {edges[i+1]}]' for i in range(bins)]
    else:
        labels = False
    movies["year_bin"] = pd.cut(year, bins=bins, labels=labels).fillna(0)
    return movies


def preprocess_movies(movies: pd.DataFrame, ratings: pd.DataFrame, add_avg_rating, encode_year, keep_title, add_main_genre, keep_genres):
    """
    Adds movies' avg rating & splits genres
    """
    num_cols = []
    if add_avg_rating:
        movie_avg_rating = ratings.groupby("movie").mean().rating.reset_index()
        movie_avg_rating.rename(
            columns={"rating": "movie_avg_rating"}, inplace=True)
        movie_avg_rating = movies.merge(
            movie_avg_rating, on="movie", how="left")
        movie_avg_rating["movie_avg_rating"] = movie_avg_rating.movie_avg_rating.fillna(
            0)
        num_cols.append("movie_avg_rating")
    else:
        movie_avg_rating = movies
    movie_avg_rating, all_genres = preprocess_genres(
        movie_avg_rating, add_main_genre, keep_genres)
    movie_avg_rating = preprocess_year(
        movie_avg_rating, encode_year=encode_year)
    if not keep_title:
        movie_avg_rating.drop(columns=["title"], inplace=True)
    movie_avg_rating.drop_duplicates("movie", inplace=True)
    cat_cols = all_genres + ["year_bin", "movie"]
    return movie_avg_rating, (cat_cols, num_cols), all_genres


def preprocess_users(users: pd.DataFrame, columns: list[str]):
    """
    Adds users' avg rating per genre
    """
    def name_column(c: str):
        return f"{c}_user_rating"
    num_cols = [name_column(c) for c in columns]
    for genre, nc in zip(columns, num_cols):
        avg_genres_rating = users.loc[users[genre] == True][[
            "user", "rating"]].groupby("user").mean().reset_index()
        avg_genres_rating.rename(columns={"rating": nc}, inplace=True)
        users = users.merge(avg_genres_rating, on="user", how="left")
    users[num_cols] = users[num_cols].fillna(0)
    users.drop(columns + ["movie", "rating"], axis=1, inplace=True)
    cat_cols = ["user"]
    return users, (cat_cols, num_cols)


def preprocess(movies, ratings: pd.DataFrame, add_user_avg_rating=True, add_movie_avg_rating=True, encode_year=True, keep_title=False, keep_genres=False, add_main_genre=True):
    movies, movie_cols, all_genres = preprocess_movies(
        movies,
        ratings,
        add_movie_avg_rating,
        encode_year,
        keep_title,
        add_main_genre,
        keep_genres,
    )
    if add_user_avg_rating:
        users = ratings[["user", "movie", "rating"]].merge(
            movies[["movie"] + all_genres], how='left', on="movie")
        users, user_cols = preprocess_users(users, all_genres)
    else:
        users = ratings[["user"]].reset_index(drop=True)
        user_cols = (["user"], [])
    assert users.isna().values.mean() == 0
    assert movies.isna().values.mean() == 0
    encode_label(movies, users)
    users.drop_duplicates("user", inplace=True)
    users = users.reset_index(drop=True)
    return movies, users, movie_cols, user_cols


def encode_label(movies, users, encode_year=False):
    movies["__movieId"] = movies.movie.copy()
    users["__userId"] = users.user.copy()
    movies.movie = LabelEncoder().fit_transform(movies.movie)
    users.user = LabelEncoder().fit_transform(users.user)


def normalize(df: pd.DataFrame, cols: list[str]):
    df[cols] = df[cols] / 5
    return df


def preprocess_genres(movies: pd.DataFrame, add_main_genre, keep_genres):
    genres = []
    all_genres = set()
    main_genres = []
    for i in movies.genres.str.split("|"):
        for k in i:
            all_genres.add(k)
    if '(no genres listed)' in all_genres:
        all_genres.remove('(no genres listed)')
    for i in movies.genres.str.split("|"):
        if add_main_genre:
            main_genres.append(i[0])
        genres.append([])
        for k in all_genres:
            genres[-1].append(k in i)
    all_genres = list(all_genres)
    genres = pd.DataFrame(data=genres, columns=all_genres)
    movies = pd.concat([movies, genres], axis=1).reset_index(drop=True)
    if add_main_genre:
        movies["main_genre"] = (main_genres)
    if not keep_genres:
        movies.drop(["genres"], axis=1, inplace=True)
    return movies, all_genres


def load_ds(nrows_ratings=10_000, nmovies_ratings=10_000):
    name_map = dict(
        userId="user",
        movieId="movie",
    )
    from .utils import read_csv
    ratings = read_csv(
        "ml-32m/ratings.csv",
        nrows=nrows_ratings,
    ).rename(name_map, axis=1)
    movies = read_csv(
        "ml-32m/movies.csv",
        nrows=nmovies_ratings,
    ).rename(name_map, axis=1)

    ratings.drop("timestamp", axis=1, inplace=True)
    return movies, ratings
