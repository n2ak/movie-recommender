from typing import Literal
from movie_recommender.data import MovieLens
from movie_recommender.logging import logger
try:
    from .utils import Env, read_parquet
except ImportError:
    from utils import Env, read_parquet  # type: ignore


def process_data(ratings_path: str, movies_path: str, model: Literal["xgb", "dlrm"]):
    Env.dump()
    logger.info(f"Proccessing data for: {model=}")
    match model:
        case "dlrm":
            train, test = process_data_for_dlrm(ratings_path, movies_path)
        case "xgb":
            train, test = process_data_for_xgb(ratings_path, movies_path)

    train_path = f"train_path_{model}.parquet"
    test_path = f"test_path_{model}.parquet"

    train.to_parquet(train_path)
    test.to_parquet(test_path)
    return dict(
        train_path=train_path,
        test_path=test_path,
    )


def process_data_for_xgb(ratings_path: str, movies_path: str):
    ratings, movies = read_parquet(ratings_path, movies_path)
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

    train = train.set_index(["user_id", "movie_id"])
    train.drop(columns=movie_genres, inplace=True)
    test = test.set_index(["user_id", "movie_id"])
    test.drop(columns=movie_genres, inplace=True)
    return train, test


def process_data_for_dlrm(ratings_path: str, movies_path: str):
    ratings, movies = read_parquet(ratings_path, movies_path)

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
    return train, test
