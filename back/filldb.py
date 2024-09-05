import asyncio
try:
    from prisma import Prisma
except:
    print("No prisma")
    pass

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import numpy as np
from datetime import datetime


def encode(df, cols, reset=False):
    lencoders = defaultdict(LabelEncoder)
    for c in cols:
        lencoders[c].fit(df[c].unique())
        df[c] = lencoders[c].transform(df[c])
    if reset:
        df = df.sort_values("userId").reset_index(drop=True)
    return df


URL = "https://m.media-amazon.com/images/I/712+BCaQuzL._AC_UF894,1000_QL80_.jpg"
URL = "https://www.svgrepo.com/show/508699/landscape-placeholder.svg"


def read_df(
    def_href, nrows=200000,
):
    print("Reading data")
    movies = pd.read_csv("dataset/ml-32m/movies.csv", nrows=nrows)
    ratings = pd.read_csv("dataset/ml-32m/ratings.csv", nrows=nrows)
    links = pd.read_csv("dataset/ml-32m/links.csv", nrows=nrows)
    # tags = pd.read_csv("dataset/ml-32m/tags.csv", nrows=nrows)
    ratings = ratings.merge(movies, on=["movieId"]).sort_values(
        "movieId").reset_index(drop=True)
    ratings = ratings.merge(links, on=["movieId"]).reset_index(drop=True)
    ratings = encode(ratings, ["userId", "movieId"], False)
    covers = pd.read_csv("covers.csv", nrows=nrows).drop_duplicates("imdbId")
    ratings = ratings.merge(covers, on="imdbId", how="left")
    ratings.href[ratings.href.isin(["-1", "-2", "-3"])] = np.nan
    ratings.href = ratings.href.fillna(def_href)
    ratings = ratings.reset_index(drop=True)
    avg_rating = ratings[["imdbId", "rating"]].groupby(
        "imdbId").mean().reset_index()
    avg_rating["avg_rating"] = avg_rating.rating
    avg_rating = avg_rating.drop("rating", axis=1)
    ratings = ratings.merge(avg_rating, on="imdbId").reset_index(drop=True)
    total_ratings = ratings[["userId", "imdbId"]].groupby(
        "imdbId").count().reset_index().rename(
        columns={"userId": "total_ratings", "imdbId": "imdbId"},
    )

    ratings = ratings.merge(total_ratings, on="imdbId").reset_index(drop=True)
    ratings.timestamp = ratings.timestamp.map(datetime.fromtimestamp)
    return ratings


def user_to_dict(row):
    i, user = row
    return {
        "username": f"user{user.userId+1}",
        "password": "password",
        "email": f"user{user.userId+1}@gmail.com",
    }


def movie_to_dict(row):
    i, movie = row
    return {
        "title": movie.title,
        "genres": str(movie.genres).split("|"),
        "imdbId": movie.imdbId,
        "href": movie.href,
        "avg_rating": movie.avg_rating,
        "total_ratings": movie.total_ratings,
    }


def rating_to_dict(row):
    i, rating = row
    return {
        # data to create a UserMovieRating record
        'rating': rating.rating,
        'movieModelId': rating.movieId+1,
        'userModelId': rating.userId+1,
        'timestamp': rating.timestamp
    }


async def main() -> None:
    prisma = Prisma()
    await prisma.connect()
    df = read_df(URL, 200_000)
    movies = df.drop_duplicates(subset=["movieId"])[
        ["movieId", "rating", "title", "genres", "imdbId", "href", "avg_rating", "total_ratings"]]
    users = df.drop_duplicates(subset=["userId"])[
        ["userId"]].sort_values("userId")
    ratings = df[["userId", "movieId", "rating",
                  "timestamp"]].sort_values("userId")
    print("Filling db")
    total_users = await prisma.usermodel.create_many(
        data=list(map(user_to_dict, users.iterrows())),
    )
    print("Created", total_users, "users")
    total_movies = await prisma.moviemodel.create_many(
        data=list(map(movie_to_dict, movies.iterrows())),
    )
    print("Created", total_movies, "movies")
    total_ratings = await prisma.usermovierating.create_many(
        data=list(map(rating_to_dict, ratings.iterrows())),
    )
    print("Created", total_ratings, "ratings")
    print("Done")
    await prisma.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
