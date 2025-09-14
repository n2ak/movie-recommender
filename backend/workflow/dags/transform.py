import numpy as np
import pandas as pd
from typing import Literal
try:
    from .utils import Env, read_parquet
except ImportError:
    from utils import Env, read_parquet  # type: ignore


def read_ds(table_name: str, db_url):
    import pandas as pd
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    table = pd.read_sql_table(table_name, engine)
    return table


def process_data(model: Literal["xgb", "dlrm"]):
    ratings, movies = read_ds(
        "ratings", Env.DB_URL), read_ds("movies", Env.DB_URL)
    # needed for somereason
    ratings = ratings.rename(str, axis="columns")
    movies = movies.rename(str, axis="columns")
    if isinstance(movies.movie_genres[0], str):
        movies.movie_genres = movies.movie_genres.apply(lambda x: x.split(","))

    from sqlalchemy import create_engine
    # Env.dump()
    print(f"Proccessing data for: {model=}")

    match model:
        case "dlrm":
            train, test = process_data_for_dlrm(ratings, movies)
        case "xgb":
            train, test = process_data_for_xgb(ratings, movies)

    conn = create_engine(Env.DB_URL)

    train.to_sql(f"{model}_train_ds", conn, index=False, if_exists="replace")
    test.to_sql(f"{model}_test_ds", conn, index=False, if_exists="replace")

    print("Data is written to db.")


def process_data_for_xgb(ratings: pd.DataFrame, movies: pd.DataFrame):
    movielens = MovieLens.preprocess(
        ratings.copy(), movies.copy(),
        Env.MAX_RATING,
        hot_encode_genres=True,
        # year_bin=10,
    )
    train, test = movielens.prepare_for_training(
        train_size=Env.TRAIN_SIZE,
        add_rating_features=True,
        columns2scale=[
            'movie_year', 'movie_mean_rating', 'movie_total_rating', 'user_mean_rating', 'user_total_rating'
        ]
    )
    train = train.sample(frac=1, random_state=0)

    movie_genres = list(filter(lambda c: c.startswith(
        "movie_genre_"), train.columns.tolist()))
    train.drop(columns=movie_genres, inplace=True)
    test.drop(columns=movie_genres, inplace=True)
    return train, test


def process_data_for_dlrm(ratings: pd.DataFrame, movies: pd.DataFrame):

    movielens = MovieLens.preprocess(
        ratings.copy(), movies.copy(),
        Env.MAX_RATING,
        hot_encode_genres=False,
        # year_bin=10,
    )
    train, test = movielens.prepare_for_training(
        train_size=Env.TRAIN_SIZE,
        add_rating_features=True,
        encode_year=False,
        columns2scale=[
            'movie_mean_rating', 'movie_total_rating', 'user_mean_rating', 'user_total_rating'
        ]
    )
    train[['movie_total_rating', 'user_total_rating']] = train[[
        'movie_total_rating', 'user_total_rating']].astype(float)

    if isinstance(train.movie_genres[0], np.ndarray):
        train["movie_genres"] = train["movie_genres"].apply(
            lambda x: x.tolist())
        test["movie_genres"] = test["movie_genres"].apply(lambda x: x.tolist())

    return train, test


class MovieLens():

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, max_rating):
        self.cat_cols, self.num_cols = get_cols(ratings)
        self.max_rating = max_rating
        self.ratings = ratings
        self.movies = movies

    @classmethod
    def split_genres(cls, movies: pd.DataFrame):
        import itertools
        genres = set(itertools.chain.from_iterable(movies.movie_genres))

        def handle_genre(genre):
            return [bool(g in genre) for g in genres]

        movies[["movie_genre_" + g for g in genres]
               ] = movies.movie_genres.apply(handle_genre).tolist()
        movies.drop(columns=["movie_genres"], inplace=True)
        return movies

    @classmethod
    def split(cls, ratings: pd.DataFrame):
        movie_cols = cls.movie_cols(ratings)
        user_cols = cls.user_cols(ratings)
        movies = ratings[movie_cols].drop_duplicates(
            "movie_id").sort_values("movie_id").set_index("movie_id", drop=False)
        users = ratings[user_cols].drop_duplicates(
            "user_id").sort_values("user_id").set_index("user_id", drop=False)
        return movies, users

    @classmethod
    def preprocess(cls, ratings: pd.DataFrame, movies: pd.DataFrame, max_rating, hot_encode_genres=False):
        if hot_encode_genres:
            movies = cls.split_genres(movies)

        return MovieLens(
            ratings,
            movies,
            max_rating
        )

    @classmethod
    def add_rating_feature(cls, *dfs: pd.DataFrame):
        train = dfs[0]
        movie_rating_features = train.groupby("movie_id").agg(
            movie_mean_rating=("rating", "mean"),
            movie_total_rating=("rating", "count"),
        )
        user_rating_features = train.groupby("user_id").agg(
            user_mean_rating=("rating", "mean"),
            user_total_rating=("rating", "count"),
        )

        def helper(df: pd.DataFrame):
            if not df.size:
                return df
            df = df.merge(movie_rating_features, on="movie_id", how="left")
            df = df.merge(user_rating_features, on="user_id", how="left")
            return df
        return [helper(df) for df in dfs]

    @classmethod
    def movie_cols(cls, df: pd.DataFrame): return list(
        filter(lambda c: c.startswith("movie"), df.columns))

    @classmethod
    def user_cols(cls, df: pd.DataFrame): return list(
        filter(lambda c: c.startswith("user"), df.columns))

    def prepare_for_training(
        self,
        train_size=.8,
        columns2scale=[],
        add_rating_features=False,
        encode_year=False,
    ):
        ratings = self.ratings
        movies = self.movies
        if encode_year:
            movies.movie_year = movies.movie_year.astype("category").cat.codes
        # return movies, ratings
        ratings = ratings.merge(movies, on="movie_id")

        assert ratings.isna().mean().mean() == 0

        # ratings.drop(columns=["time", "imdbId", "title"], inplace=True)
        train_size = int(len(ratings) * train_size)

        if add_rating_features:
            # splitting before add_rating_feature inroduces nan values
            # train, test = self.add_rating_feature(train, test)
            ratings, = self.add_rating_feature(ratings)
        train, test = ratings[:train_size], ratings[train_size:]

        if columns2scale:
            train, test = MovieLens.scale(train, test, columns2scale)
        return train, test

    @classmethod
    def scale(cls, train_X: pd.DataFrame, test_X: pd.DataFrame, columns):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler()
        train_X[columns] = scaler.fit_transform(train_X[columns])
        if test_X.size:
            test_X[columns] = scaler.transform(test_X[columns])
        return train_X, test_X


def get_cols(ratings: pd.DataFrame):
    cat_cols = ratings.iloc[:1].select_dtypes(
        ["int", "bool", "category"]).columns.to_list()
    num_cols = ratings.iloc[:1].select_dtypes("float").columns.to_list()
    if "rating" in num_cols:
        num_cols.remove("rating")
    ratings[cat_cols] = ratings[cat_cols]
    ratings[num_cols] = ratings[num_cols].astype("float32")
    return cat_cols, num_cols
