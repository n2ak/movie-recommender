from .utils import read_csv, exclude, Instance
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import re


year_exp = re.compile("^.+\((\d+)\)$")


def get_year(title):
    matches = year_exp.match(title.strip())
    if matches is None:
        return np.nan
    groups = matches.groups()
    assert len(groups) == 1
    return int(groups[0])


def preprocess_year(df: pd.DataFrame, bins, encode_year=True):
    year = df.movie_title.map(get_year)
    if not encode_year:
        edges = np.linspace(year.min(), year.max(), bins+1).astype(int)
        labels = [f'({edges[i]}, {edges[i+1]}]' for i in range(bins)]
    else:
        labels = False
    df["movie_year"] = year.fillna(year.min())
    df["movie_year_bin"] = pd.cut(
        df.movie_year, bins=bins, labels=labels)
    if df["movie_year_bin"].min() == 1:
        df["movie_year_bin"] = df["movie_year_bin"]-1
    return df


def preprocess_movies(movies: pd.DataFrame, ratings: pd.DataFrame, add_avg_rating, encode_year, keep_title, split_genres, keep_genres):
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
        movie_avg_rating, split_genres, keep_genres)
    # movie_avg_rating = preprocess_year(movie_avg_rating, encode_year=encode_year)
    if not keep_title:
        movie_avg_rating.drop(columns=["title"], inplace=True)
    movie_avg_rating.drop_duplicates("movie", inplace=True)
    return movie_avg_rating, all_genres


def preprocess_users(users: pd.DataFrame, columns: list[str]):
    """
    Adds users' avg rating per genre
    """
    def name_column(c: str):
        return f"user_{c.replace('movie_','')}_rating"
    num_cols = [name_column(c) for c in columns]
    for genre, nc in zip(columns, num_cols):
        avg_genres_rating = users.loc[users[genre] == True][[
            "user", "rating"]].groupby("user").mean().reset_index()
        avg_genres_rating.rename(columns={"rating": nc}, inplace=True)
        users = users.merge(avg_genres_rating, on="user", how="left")
    users[num_cols] = users[num_cols].fillna(0)
    users.drop(columns + ["movie", "rating"], axis=1, inplace=True)
    return users


def preprocess(movies, ratings: pd.DataFrame, add_user_avg_rating=True, add_movie_avg_rating=True, encode_year=True, keep_title=False, keep_genres=False, split_genres=True):
    movies, all_genres = preprocess_movies(
        movies,
        ratings,
        add_movie_avg_rating,
        encode_year,
        keep_title,
        split_genres,
        keep_genres,
    )
    if add_user_avg_rating:
        users = ratings[["user", "movie", "rating"]].merge(
            movies[["movie"] + all_genres], how='left', on="movie")
        users = preprocess_users(users, all_genres)
    else:
        users = ratings[["user"]].reset_index(drop=True)
    assert users.isna().values.mean() == 0
    assert movies.isna().values.mean() == 0
    users.drop_duplicates("user", inplace=True)
    users = users.reset_index(drop=True)
    return movies, users


def encode_label(df, add_one=False):
    a = 0 if not add_one else 1
    df.movie = LabelEncoder().fit_transform(df.movie)+a
    df.user = LabelEncoder().fit_transform(df.user)+a


def normalize(df: pd.DataFrame, cols: list[str]):
    df[cols] = df[cols] / 5
    return df


def preprocess_genres(movies: pd.DataFrame, split_genres, keep_genres):
    genres = []
    all_genres = set()
    # main_genres = []
    for i in movies.movie_genres.str.split("|"):
        for k in i:
            if k != '(no genres listed)':
                all_genres.add("movie_" + k)
    for i in movies.movie_genres.str.split("|"):
        # if split_genres:
        #     main_genres.append(i[0])
        genres.append([])
        for k in all_genres:
            genres[-1].append(k in i)
    all_genres = list(all_genres)
    genres = pd.DataFrame(data=genres, columns=all_genres)
    movies = pd.concat([movies, genres], axis=1).reset_index(drop=True)
    # if split_genres:
    # movies["main_genre"] = (main_genres)
    if not keep_genres:
        movies.drop(["genres"], axis=1, inplace=True)
    if split_genres:
        movies["genres"] = movies.genres.str.split("|")
    return movies, all_genres


def _load_ds(nrows_ratings=10_000, nmovies_ratings=10_000):
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


class MovieLens(Instance):
    def load_ds(self):
        def get_cols(df: pd.DataFrame, prefix):
            return list(filter(lambda a: a.startswith(prefix), df.columns.to_list()))

        ratings = read_csv("ratings_full.csv")
        movie_cols = get_cols(ratings, "movie")
        user_cols = get_cols(ratings, "user")

        movies = ratings[movie_cols].drop_duplicates("movie")
        users = ratings[user_cols].drop_duplicates("user")
        return ratings, movies, users, movie_cols, user_cols

    def init(self):
        ratings, movies, users, movie_cols, user_cols = self.load_ds()
        movie_num_cols = ["movie_avg_rating"]
        movie_cat_cols = exclude(movie_cols, "movie_imdbId", "movie_href",
                                 "movie_year", "movie_title", "movie_genres", *movie_num_cols)
        user_cat_cols = ["user"]
        user_num_cols = exclude(
            user_cols, "user_imdbId", "user_href", "user_year", "user_title", *user_cat_cols)

        self.ratings = ratings
        self.movies = movies
        self.users = users

        self.movie_cat_cols = movie_cat_cols
        self.movie_num_cols = movie_num_cols
        self.user_num_cols = user_num_cols
        self.user_cat_cols = user_cat_cols

    @property
    def n_users(self): return self.users.user.nunique()+1
    @property
    def n_movies(self): return self.movies.movie.nunique()+1
