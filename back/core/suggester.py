import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsTransformer


class Suggester:
    """
    Useful for suggesting movies to be ran through a DNN, to reduce the input to the DNN
    from millions to thousands or so. 
    """

    def __init__(self, n_neighbors_user=5, n_neighbors_movie=5) -> None:
        self.movie_knn = KNeighborsTransformer(n_neighbors=n_neighbors_movie)
        self.user_knn = KNeighborsTransformer(n_neighbors=n_neighbors_user)
        self.is_fit = False

    def fit(self, movies: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame):
        self.movieIds = movies.movie
        self.userIds = users.user
        self.movies = movies = movies.drop(columns="movie").astype(float)
        self.users = users = users.drop(columns="user").astype(float)
        self.movie_knn.fit(movies.values)
        self.user_knn.fit(users.values)

        self.is_fit = True
        self.ratings = ratings

    def suggest(
        self,
        user_id,
        nmovies,
        n_neighbor_users=10,
        n_neighbor_movies=100,
        n=10,
        add_closest_users=True,
        deterministic=False,
    ):
        """
        Get most relevant movies to user, and users close to the user.
        """
        assert self.is_fit
        # top movies that the user has already rated
        top_movies_for_user = self.get_users_top_movies(
            [user_id],
            deterministic=deterministic, n=n
        )
        if len(top_movies_for_user) == 0:
            # user hasn't rated any movies
            return self.movieIds.unique()

        # get similar users
        closest_users = self.get_closest_users([user_id], n_neighbor_users)
        if add_closest_users:
            # top movies for colsest users
            top_movies_for_similar_users = self.get_users_top_movies(
                self.userIds[closest_users].tolist(),
                deterministic=deterministic, n=n,
            )

        # get closest movies to user movies
        closest_movies_for_user = self.get_closest_movies(
            top_movies_for_user,
            n_neighbor_movies,
        )

        # closest_movies = movies.iloc[closest_movies]
        if add_closest_users:
            # get only the intersection between top movies of similar users and the user himself
            top_movies = np.intersect1d(
                top_movies_for_similar_users,
                closest_movies_for_user,
            )
        else:
            top_movies = closest_movies_for_user

        return top_movies[:nmovies]

    def get_users_top_movies(self, userIds: list[int], n=3, deterministic=False):
        ratings = self.ratings
        movie_ids = []
        for uid in userIds:
            user_movie_ids = ratings.loc[ratings.user == uid].sort_values(
                "rating", ascending=False).iloc[:n*5].movie.values
            if len(user_movie_ids):
                if not deterministic:
                    movie_ids.extend(np.random.choice(
                        user_movie_ids, n).tolist())
                else:
                    movie_ids.extend(user_movie_ids[:n])
        return movie_ids

    def get_closest_users(self, user_ids, k):
        return self.user_knn.kneighbors(
            self.users.loc[self.userIds.isin(user_ids)].values,
            n_neighbors=k,
            return_distance=False,
        )[0]

    def get_closest_movies(self, movie_ids, k):
        return self.movie_knn.kneighbors(
            self.movies.loc[self.movieIds.isin(movie_ids)].values,
            n_neighbors=k, return_distance=False,
        )[0]


def load_suggester():
    from .data import load_ds, preprocess
    from .utils import exclude
    movies, ratings = load_ds()
    movies, users, *_ = preprocess(
        movies, ratings,
    )
    movie_cols = exclude(
        movies.columns.tolist(),
        "__movieId", "main_genre",
    )
    user_cols = exclude(
        users.columns.tolist(),
        "__userId",
    )
    suggester = Suggester()
    suggester.fit(
        movies[movie_cols],
        users[user_cols],
        ratings
    )
    return suggester
