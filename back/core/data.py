from .utils import read_csv, exclude, Instance
import numpy as np
import pandas as pd


def preprocess(ratings: pd.DataFrame, max_rating, bin_movie_total_ratings=3):
    # TODO move the renaming to JS
    ratings.rename(columns={"total_ratings": "movie_total_ratings",
                            "avg_rating": "movie_avg_rating"}, inplace=True)

    from itertools import chain
    ratings.genres = ratings.genres.str.split(",")
    all_genres = sorted(list(set(chain.from_iterable(ratings.genres))))

    # User avg rating
    user_avg_rating = ratings.groupby("userId").rating.mean().rename(
        "user_avg_rating").reset_index()
    ratings = ratings.merge(user_avg_rating, on="userId", how="left")
    # process date
    ratings.movie_date = pd.to_datetime(ratings.movie_date, format="%d/%m/%Y")
    ratings["movie_year"] = pd.Categorical(ratings.movie_date.dt.year).codes

    # Genres to 0,1s
    def process_genres(a):
        ret = [0] * len(all_genres)
        for g in a:
            ret[all_genres.index(f"movie_{g}")] = 1
        return ret
    all_genres = [f"movie_{g}" for g in all_genres]
    ratings[all_genres] = pd.DataFrame(
        ratings.genres.map(process_genres).tolist())
    # User avg rating per genre
    for g in all_genres:
        a = ratings[["userId", "rating", g]]
        a = a.loc[a[g] == 1]
        name = f"user_{g}_avg_rating"
        a = a.groupby("userId").rating.mean().rename(name).reset_index()
        ratings = ratings.merge(a, on="userId", how="left")
        ratings[name] = ratings[name].fillna(0)
    # movie_total_ratings => bins
    if bin_movie_total_ratings is not None:
        ratings["movie_total_ratings"] = pd.cut(
            ratings.movie_total_ratings, bin_movie_total_ratings).cat.codes
    ratings["movie_avg_rating"] = ratings.movie_avg_rating.values.round()
    user_genre_avg_rating_cols = [f"user_{g}_avg_rating" for g in all_genres]
    num_cols = user_genre_avg_rating_cols + \
        ["movie_avg_rating", "user_avg_rating"]
    label = "rating"
    cat_cols = ["userId", "movieId", "movie_year",
                "movie_total_ratings"] + all_genres

    ratings = ratings[[label] + cat_cols + num_cols]
    ratings["rating"] = ratings.rating/max_rating

    ratings[cat_cols] = ratings[cat_cols].astype(np.int32)
    ratings[num_cols] = ratings[num_cols].astype(np.float32)
    ratings[label] = ratings[label].astype(np.float32)

    return ratings


def encode_ids(df):
    from sklearn.preprocessing import LabelEncoder

    def encode(name):
        encoder = LabelEncoder().fit(df[name])
        df[name] = encoder.transform(df[name])
        return encoder
    return encode("userId"), encode("movieId")


class MovieLens(Instance):
    def init(self):
        ratings = read_csv("ratings_processed.csv")
        self.user_encoder, self.movie_encoder = encode_ids(ratings)
        movie_cols = [c for c in ratings.columns.to_list()
                      if c.startswith("movie")]
        user_cols = [c for c in ratings.columns.to_list()
                     if c.startswith("user")]
        self.cat_cols, self.num_cols = get_cols(ratings)
        self.movies = ratings[movie_cols].drop_duplicates("movieId")
        self.users = ratings[user_cols].drop_duplicates("userId")
        self._ratings = ratings

    def realids_to_ids(self, encoder, ids):
        return encoder.transform(ids).tolist()

    def ids_to_realids(self, encoder, ids):
        return encoder.inverse_transform(ids).tolist()

    def map_ids(self, userIds, movieIds, reverse=False):
        try:
            if reverse:
                userIds = self.ids_to_realids(self.user_encoder, userIds)
                movieIds = self.ids_to_realids(
                    self.movie_encoder, movieIds)
            else:
                userIds = self.realids_to_ids(self.user_encoder, userIds)
                movieIds = self.realids_to_ids(
                    self.movie_encoder, movieIds)
        except:
            raise BackendError("Incorrect ids")
        return userIds, movieIds


def get_cols(ratings: pd.DataFrame):
    cat_cols = ratings.iloc[:1].select_dtypes("int").columns.to_list()
    num_cols = ratings.iloc[:1].select_dtypes("float").columns.to_list()
    if "rating" in num_cols:
        num_cols.remove("rating")
    ratings[cat_cols] = ratings[cat_cols].astype("int32")
    ratings[num_cols] = ratings[num_cols].astype("float32")
    return cat_cols, num_cols


class BackendError(RuntimeError):
    def __init__(self, msg) -> None:
        super().__init__()
        self.msg = msg
