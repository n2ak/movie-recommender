max_rating = 5
try:
    from .utils import Env, upload_files, connect_minio
except ImportError:
    from utils import Env, upload_files, connect_minio  # type: ignore


def read_db(db_url, ratings_table: str, movies_table: str):
    import pandas as pd
    from sqlalchemy import create_engine
    print(f"Reading db from: '{db_url}'")
    engine = create_engine(db_url)
    ratings = pd.read_sql_table(ratings_table, engine, columns=[
                                "rating", "movieModelId", "userModelId"])
    ratings.rename(columns=dict(userModelId="user_id",
                   movieModelId="movie_id"), inplace=True)

    movies = pd.read_sql_table(movies_table, engine, columns=[
                               "id", "title", "genres", "year"])
    movies.rename(
        columns={c: f"movie_{c}" for c in movies.columns}, inplace=True)
    return ratings, movies


def extract_data():
    # from sqlalchemy import create_engine
    connect_minio()
    Env.dump()

    ratings, movies = read_db(Env.DB_URL, "movie_rating", "movie")
    print("Done extracting data.")
    movies.drop(columns=["movie_title"], inplace=True)
    ratings["rating"] = ratings["rating"] / max_rating

    # conn = create_engine(Env.DB_URL)
    movies.movie_genres = movies.movie_genres.apply(lambda x: ",".join(x))

    def save_fn(path):
        ratings.to_parquet(path / "ratings.parquet")
        movies.to_parquet(path / "movies.parquet")
    upload_files(save_fn, "trainingbucket")
