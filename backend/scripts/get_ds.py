import pandas as pd


ds_path = "hf://datasets/ashraq/movielens_ratings/"
splits = {
    'train': 'data/train-00000-of-00001-8c8c7645a52d95e5.parquet',
    'val': 'data/validation-00000-of-00001-609ec132d91847f9.parquet',
}


def read_ds(limit=None):
    # df = pd.concat([
    #     pd.read_parquet(ds_path + splits["train"]),
    #     pd.read_parquet(ds_path + splits["val"]),
    # ],
    #     axis=0
    # )
    df = pd.read_parquet(ds_path + splits["val"])
    if limit is not None:
        df = df[:limit]
    return df


def split_title(title):
    title, year = title[:-7], title[-5:-1]
    year = int(year) if year.isdigit() else -1
    return title, year


def process(df: pd.DataFrame):
    df = df.drop_duplicates(["movie_id", "user_id"])
    ratings = df.copy()
    ratings = ratings[ratings.genres != "(no genres listed)"]
    ratings = ratings.merge(ratings.groupby("movie_id").agg(
        movie_avg_rating=("rating", "mean"),
        movie_total_rating=("rating", "count"),
    ), on="movie_id")
    ratings[["title", "year"]] = ratings.title.apply(split_title).tolist()
    ratings = ratings[ratings.year != -1]

    ratings.user_id = ratings.user_id.astype("category").cat.codes
    ratings.movie_id = ratings.movie_id.astype("category").cat.codes

    movies = ratings[["imdbId", "tmdbId", "movie_id", "title", "genres", "posters",
                      "movie_avg_rating", "movie_total_rating", "year"]].drop_duplicates("movie_id")
    ratings = ratings[["user_id", "movie_id", "rating"]]
    # movies.drop(columns=['imdbId', 'tmdbId'], inplace=True)

    ratings.sort_values("user_id", inplace=True)
    movies.sort_values("movie_id", inplace=True)
    ratings.reset_index(inplace=True, drop=True)
    movies.reset_index(inplace=True, drop=True)
    users = ratings.user_id.drop_duplicates().sort_values().to_frame()

    users["username"] = users["user_id"].apply(lambda id: f"user{id}")

    return ratings, movies, users


def save(**ds: pd.DataFrame):
    import pathlib
    import tempfile
    dir = tempfile.mkdtemp()
    paths = []
    for name, d in ds.items():
        p = pathlib.Path(dir) / f"{name}.csv"
        d.to_csv(p, index=False)
        paths.append(str(p))
    return paths


def main(limit):
    df = read_ds(limit)
    ratings, movies, users = process(df)
    paths = save(
        ratings=ratings,
        movies=movies,
        users=users,
    )
    print(" ".join(paths))


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(limit)
