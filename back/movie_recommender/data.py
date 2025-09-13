import pandas as pd


def read_db(db_url):
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


class MovieLens():

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame, max_rating):
        self.cat_cols, self.num_cols = get_cols(ratings)
        self.max_rating = max_rating
        self.ratings = ratings
        self.movies = movies

    @classmethod
    def split_genres(cls, movies: pd.DataFrame):
        import itertools
        genres = set(itertools.chain.from_iterable(movies.movie_genres))

        def handle_genre(genre):
            return [bool(g in genre) for g in genres]

        movies[["movie_genre_" + g for g in genres]
               ] = movies.movie_genres.apply(handle_genre).tolist()
        movies.drop(columns=["movie_genres"], inplace=True)
        return movies

    @classmethod
    def split(cls, ratings: pd.DataFrame):
        movie_cols = cls.movie_cols(ratings)
        user_cols = cls.user_cols(ratings)
        movies = ratings[movie_cols].drop_duplicates(
            "movie_id").sort_values("movie_id").set_index("movie_id", drop=False)
        users = ratings[user_cols].drop_duplicates(
            "user_id").sort_values("user_id").set_index("user_id", drop=False)
        return movies, users

    @classmethod
    def preprocess(cls, ratings: pd.DataFrame, movies: pd.DataFrame, max_rating, hot_encode_genres=False):
        if hot_encode_genres:
            movies = cls.split_genres(movies)

        return MovieLens(
            ratings,
            movies,
            max_rating
        )

    @classmethod
    def add_rating_feature(cls, *dfs: pd.DataFrame):
        train = dfs[0]
        movie_rating_features = train.groupby("movie_id").agg(
            movie_mean_rating=("rating", "mean"),
            movie_total_rating=("rating", "count"),
        )
        user_rating_features = train.groupby("user_id").agg(
            user_mean_rating=("rating", "mean"),
            user_total_rating=("rating", "count"),
        )

        def helper(df: pd.DataFrame):
            if not df.size:
                return df
            df = df.merge(movie_rating_features, on="movie_id", how="left")
            df = df.merge(user_rating_features, on="user_id", how="left")
            return df
        return [helper(df) for df in dfs]

    @classmethod
    def movie_cols(cls, df: pd.DataFrame): return list(
        filter(lambda c: c.startswith("movie"), df.columns))

    @classmethod
    def user_cols(cls, df: pd.DataFrame): return list(
        filter(lambda c: c.startswith("user"), df.columns))

    def prepare_for_training(
        self,
        train_size=.8,
        columns2scale=[],
        add_rating_features=False,
        encode_year=False,
    ):
        ratings = self.ratings
        movies = self.movies
        if encode_year:
            movies.movie_year = movies.movie_year.astype("category").cat.codes
        # return movies, ratings
        ratings = ratings.merge(movies, on="movie_id")

        assert ratings.isna().mean().mean() == 0

        # ratings.drop(columns=["time", "imdbId", "title"], inplace=True)
        train_size = int(len(ratings) * train_size)

        if add_rating_features:
            # splitting before add_rating_feature inroduces nan values
            # train, test = self.add_rating_feature(train, test)
            ratings, = self.add_rating_feature(ratings)
        train, test = ratings[:train_size], ratings[train_size:]

        if columns2scale:
            train, test = MovieLens.scale(train, test, columns2scale)
        return train, test

    @classmethod
    def scale(cls, train_X: pd.DataFrame, test_X: pd.DataFrame, columns):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler()
        train_X[columns] = scaler.fit_transform(train_X[columns])
        if test_X.size:
            test_X[columns] = scaler.transform(test_X[columns])
        return train_X, test_X


def get_cols(ratings: pd.DataFrame):
    cat_cols = ratings.iloc[:1].select_dtypes(
        ["int", "bool", "category"]).columns.to_list()
    num_cols = ratings.iloc[:1].select_dtypes("float").columns.to_list()
    if "rating" in num_cols:
        num_cols.remove("rating")
    ratings[cat_cols] = ratings[cat_cols]
    ratings[num_cols] = ratings[num_cols].astype("float32")
    return cat_cols, num_cols


# def smote(X: pd.DataFrame, max_rating):
#     from imblearn.over_sampling import SMOTE

#     y = (X.pop("rating") * max_rating).astype(int)
#     sm = SMOTE(random_state=42)
#     X, y = sm.fit_resample(X, y)
#     y = y / max_rating
#     return X, y
