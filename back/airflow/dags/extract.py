from movie_recommender.logging import logger
from movie_recommender.data import read_db
max_rating = 5
try:
    from .utils import Env
except ImportError:
    from utils import Env  # type: ignore


def extract_data():
    Env.dump()

    ratings, movies = read_db(Env.DB_URL)
    logger.info("Done extracting data.")
    movies.drop(columns=["movie_title"], inplace=True)
    ratings["rating"] = ratings["rating"] / max_rating

    movies_path = "movies.parquet"
    ratings_path = "ratings.parquet"

    movies.to_parquet(movies_path)
    ratings.to_parquet(ratings_path)

    return dict(
        ratings_path=ratings_path,
        movies_path=movies_path,
    )
