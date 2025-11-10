import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, load_from_disk


ds_path = "hf://datasets/ashraq/movielens_ratings/"
splits = {
    'train': 'data/train-00000-of-00001-8c8c7645a52d95e5.parquet',
    'val': 'data/validation-00000-of-00001-609ec132d91847f9.parquet',
}


def read_ds(limit=None):
    ratings = pd.read_parquet(ds_path + splits["val"])[:limit]
    movies = load_dataset(
        "wykonos/movies", split="train").shuffle(seed=0).take(1_000)
    movies = movies.filter(lambda m: m["overview"] and m["title"])

    movies_df = movies.to_pandas().rename(columns={"id": "tmdbId"})
    movies_df.drop_duplicates("tmdbId", inplace=True)
    ratings.dropna(inplace=True)
    ratings["tmdbId"] = ratings["tmdbId"].apply(int)
    ratings_cols = ["tmdbId", "user_id", "rating", "posters"]
    ratings = ratings[ratings_cols]
    movies_df["movie_id"] = LabelEncoder().fit_transform(movies_df.tmdbId)

    ratings = ratings.merge(
        movies_df[["movie_id", "tmdbId"]], on="tmdbId").drop(columns="tmdbId")
    movies_df.drop(columns="tmdbId", inplace=True)
    return ratings, movies_df


def process(ratings, movies):
    ratings = ratings.copy()
    ratings = ratings.drop_duplicates(["movie_id", "user_id"])
    movies = movies.drop_duplicates(["movie_id"])
    movies = movies.copy()

    ratings.user_id = ratings.user_id.astype("category").cat.codes

    ratings.sort_values("user_id", inplace=True)
    movies.sort_values("movie_id", inplace=True)
    ratings.reset_index(inplace=True, drop=True)
    movies.reset_index(inplace=True, drop=True)

    users = ratings.user_id.drop_duplicates().sort_values().to_frame()
    users["username"] = users["user_id"].apply(lambda id: f"user{id}")

    movies.drop(columns=["poster_path", "backdrop_path", "recommendations",

                         ], inplace=True)
    movies = movies.merge(
        ratings[["movie_id", "posters"]], on="movie_id", how="left")
    movies.release_date = pd.to_datetime(movies.release_date)

    cols = ["posters", "production_companies",
            "tagline", "genres", "credits", "keywords"]
    movies[cols] = movies[cols].fillna("")
    movies.release_date = movies.release_date.fillna(datetime.datetime.now())
    ratings = ratings[ratings.movie_id.isin(movies.movie_id)]
    movies = movies.drop_duplicates("movie_id")
    return ratings, movies, users


def save(**ds: pd.DataFrame):
    for name, d in ds.items():
        p = f"{name}.parquet"
        d.to_parquet(p)


def main(limit):
    ratings, movies = read_ds(limit)
    ratings, movies, users = process(ratings, movies)
    save(
        ratings=ratings,
        movies=movies,
        users=users,
    )


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    main(limit)
