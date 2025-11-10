from movie_recommender.common.logging import Logger
from movie_recommender.common.env import DATABASE_URL, ARTIFACT_ROOT, BUCKET
from movie_recommender.common.workflow import StorageClient
import pandas as pd
import tempfile

max_rating = 5


def read_db(db_url, ratings_table: str, movies_table: str, users_table: str):
    import pandas as pd
    from sqlalchemy import create_engine
    Logger.info(f"Reading db from: '{db_url}'")
    engine = create_engine(db_url)
    ratings = pd.read_sql_table(ratings_table, engine, columns=[
                                "rating", "movieId", "userModelId"])
    movies = pd.read_sql_table(movies_table, engine, columns=[
                               "id", "title", "genres", "release_date"])
    users = pd.read_sql_table(users_table, engine, columns=[
        "id"])

    ratings.rename(columns=dict(userModelId="user_id",
                   movieId="movie_id"), inplace=True)

    movies["year"] = movies.release_date.dt.year
    movies.drop(columns=["release_date"], inplace=True)

    movies.rename(
        columns={c: f"movie_{c}" for c in movies.columns}, inplace=True)
    users.rename(columns={c: f"user_{c}" for c in users.columns}, inplace=True)
    return ratings, movies, users


def extract_features(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame):
    from movie_recommender.common.utils import movie_cols, user_cols

    ratings = ratings.merge(movies, on="movie_id", how="left")
    movie_agg = ratings.groupby("movie_id").agg(
        movie_mean_rating=("rating", "mean"),
        movie_total_rating=("rating", "count"),
    )

    user_agg = ratings.groupby("user_id").agg(
        user_mean_rating=("rating", "mean"),
        user_total_rating=("rating", "count"),
    )

    movie_features = movies.merge(movie_agg, on="movie_id", how="left")
    users_features = users.merge(user_agg, on="user_id", how="left")

    movie_na_cols = ["movie_mean_rating", "movie_total_rating"]
    user_na_cols = ["user_mean_rating", "user_total_rating"]
    movie_features[movie_na_cols] = movie_features[movie_na_cols].fillna(
        0).astype(float)
    users_features[user_na_cols] = users_features[user_na_cols].fillna(
        0).astype(float)

    movie_features["movie_genres"] = movie_features.movie_genres.str.split(",")
    movie_features["movie_year"] = movie_features["movie_year"].astype(
        "category").cat.codes
    movie_features["movie_main_genre"] = movie_features["movie_genres"].apply(
        lambda x: x[0] if len(x) else "")
    movie_features["movie_main_genre"] = movie_features["movie_main_genre"].astype(
        "category").cat.codes

    movie_features.drop(columns="movie_genres", inplace=True)

    StorageClient.get_instance().upload_parquet_to_bucket(
        BUCKET,
        ARTIFACT_ROOT,
        users_features=users_features,
        movies_features=movie_features,
    )
    return users_features, movie_features


def extract_data():
    ratings, movies, users = read_db(
        DATABASE_URL, "movie_rating", "movie", "user")
    movies.drop(columns=["movie_title"], inplace=True)
    ratings["rating"] = ratings["rating"] / max_rating

    movies.movie_genres = movies.movie_genres.apply(lambda x: ",".join(x))

    path = tempfile.mkdtemp()
    ratings.to_parquet(f"{path}/ratings.parquet")
    movies.to_parquet(f"{path}/movies.parquet")
    users.to_parquet(f"{path}/users.parquet")

    StorageClient.get_instance().upload_folder_to_bucket(
        path, bucket=BUCKET, root=ARTIFACT_ROOT)
    return ratings, movies, users


def main():
    ratings, movies, users = extract_data()
    Logger.info("Feature extraction...")
    users_features, movie_features = extract_features(ratings, movies, users)
    Logger.info("Data extraction is done.")
    return ratings, users_features, movie_features


if __name__ == "__main__":
    main()
