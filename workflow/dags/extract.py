import tempfile
max_rating = 5


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


def extract_data(db_url: str):
    # from sqlalchemy import create_engine
    # conn = create_engine(Env.DB_URL)

    ratings, movies = read_db(db_url, "movie_rating", "movie")
    print("Done extracting data.")
    movies.drop(columns=["movie_title"], inplace=True)
    ratings["rating"] = ratings["rating"] / max_rating

    movies.movie_genres = movies.movie_genres.apply(lambda x: ",".join(x))

    path = tempfile.mkdtemp()
    movies_path = f"{path}/movies.parquet"
    ratings_path = f"{path}/ratings.parquet"
    ratings.to_parquet(ratings_path)
    movies.to_parquet(movies_path)
    return path
