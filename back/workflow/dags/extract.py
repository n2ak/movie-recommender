max_rating = 5
try:
    from .utils import Env
except ImportError:
    from utils import Env  # type: ignore


def read_db(db_url):
    import pandas as pd
    from sqlalchemy import create_engine
    print(f"Reading db from: '{db_url}'")
    engine = create_engine(db_url)
    ratings = pd.read_sql_table("movie_rating", engine, columns=[
                                "rating", "movieModelId", "userModelId"])
    ratings.rename(columns=dict(userModelId="user_id",
                   movieModelId="movie_id"), inplace=True)

    movies = pd.read_sql_table("movie", engine, columns=[
                               "id", "title", "genres", "year"])
    movies.rename(
        columns={c: f"movie_{c}" for c in movies.columns}, inplace=True)
    return ratings, movies


def extract_data():
    # Env.dump()

    ratings, movies = read_db(Env.DB_URL)
    print("Done extracting data.")
    movies.drop(columns=["movie_title"], inplace=True)
    ratings["rating"] = ratings["rating"] / max_rating

    movies_path = "movies.parquet"
    ratings_path = "ratings.parquet"

    movies.to_parquet(movies_path)
    ratings.to_parquet(ratings_path)
    return dict(
        movies_path=movies_path,
        ratings_path=ratings_path,
    )
