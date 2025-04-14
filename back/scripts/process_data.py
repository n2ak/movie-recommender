import pandas as pd
from core.data import preprocess, encode_label, preprocess_year
from core.utils import read_csv, to_csv

DEFAULT_URL = "https://www.svgrepo.com/show/508699/landscape-placeholder.svg"

name_map = dict(
    userId="user",
    movieId="movie",
    genres="movie_genres",
    title="movie_title",
    href="movie_href",
    imdbId="movie_imdbId",
)


def make_final_df(nrows, save=True, year_bins=7):
    def href(a):
        if a in ["-2", "-1"]:
            return DEFAULT_URL
        return a
    ratings = read_csv("ml-32m/ratings.csv",
                       nrows=nrows).rename(name_map, axis=1)
    movies = read_csv("ml-32m/movies.csv",
                      nrows=nrows).rename(name_map, axis=1)
    covers = read_csv("covers.csv", nrows=nrows).rename(name_map, axis=1)
    links = read_csv("ml-32m/links.csv", nrows=nrows).rename(name_map, axis=1)
    covers.movie_href = covers.movie_href.apply(href)

    # year
    movies, users = preprocess(
        movies, ratings, keep_title=True, split_genres=False, keep_genres=True)
    links = links.merge(covers, on="movie_imdbId")
    ratings = ratings.merge(links, on="movie")
    ratings = ratings.merge(movies, on="movie")
    preprocess_year(ratings, bins=year_bins)

    # avg and total ratings
    total_ratings = ratings.groupby("movie").rating.count().reset_index().rename(
        columns={"rating": "total_ratings"})
    avg_rating = ratings.groupby("movie").rating.mean().reset_index().rename(
        columns={"rating": "avg_rating"})

    ratings = ratings.merge(total_ratings, on="movie")
    ratings = ratings.merge(avg_rating, on="movie")
    ratings = ratings.merge(users, on="user")
    ratings.drop(columns=["tmdbId"], inplace=True)
    assert ratings.isna().values.mean() == 0
    encode_label(ratings, add_one=True)
    if save:
        to_csv(ratings, "ratings_full.csv")
        print("Saved")
    return ratings
