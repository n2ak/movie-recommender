import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsTransformer
from core.data import MovieLens

from core.utils import display


class Suggester:
    """
    Useful for suggesting movies to be ran through a DNN, to reduce the input to the DNN
    from millions to thousands or so. 
    """

    def __init__(self, n_neighbors_user=5, n_neighbors_movie=5) -> None:
        self.movie_knn = KNeighborsTransformer(n_neighbors=n_neighbors_movie)
        self.user_knn = KNeighborsTransformer(n_neighbors=n_neighbors_user)
        self.is_fit = False

    def fit(self, movielens: MovieLens):
        movies = movielens.movies.copy()
        users = movielens.users.copy()
        ratings = movielens._ratings.copy()

        self.movies = movies
        self.users = users
        self.movie_knn.fit(movies.drop("movieId", axis=1).values)
        self.user_knn.fit(users.drop("userId", axis=1).values)

        self.is_fit = True
        self.ratings = ratings

    def movies_with_genres(self, genres, relation, inids=None):
        movies = self.movies
        if inids is not None:
            movies = movies.loc[movies.movieId.isin(inids)]
        relation = {"or": np.any, "and": np.all}[relation]
        cond = relation(movies[[f"movie_{g}" for g in genres]], axis=1)
        return movies.loc[cond].movieId.tolist()

    def suggest(
        self,
        user_id,
        nmovies,
        n_neighbor_users=100,
        n_neighbor_movies=100,
        genres: list[str] = [],
        add_closest_users=True,
        relation=None,
    ):
        """
        Get most relevant movies to user, and users close to the user.
        """
        assert self.is_fit
        # top movies that the user has already rated
        aleady_watched = self.get_users_top_movies(
            [user_id], genres=genres,
            relation=relation,
        )
        # print("aleady_watched", len(aleady_watched))
        if len(aleady_watched):
            # get closest movies to user movies
            closest_movies_for_user = self.get_closest_movies(
                aleady_watched,
                n_neighbor_movies,
                genres=genres,
                relation=relation,
            )
            # get similar users
            # print("closest_movies_for_user", len(closest_movies_for_user))
            if add_closest_users:
                closest_users = self.get_closest_users(
                    [user_id], n_neighbor_users)
                # print("closest_users", closest_users)
                if len(closest_users):
                    # top movies for colsest users
                    top_movies_for_similar_users = self.get_users_top_movies(
                        closest_users,
                        genres=genres,
                        relation=relation,
                    )
                    intersection = np.concatenate(
                        [top_movies_for_similar_users,
                         closest_movies_for_user,]
                    )
                    return np.unique(intersection)[:nmovies]
            else:
                # Don't add silmilar users' movies
                # so just return closes movies to user
                pass
            return np.unique(closest_movies_for_user)[:nmovies]
        else:
            # user hasn't rated any movies
            print("No ratings cant suggest")
            movie_ids = None
            if len(genres):
                movie_ids = self.movies_with_genres(genres, relation)
            # sort movies based on avg_rating
            movie_ids = self.sort_movies(movie_ids=movie_ids)
            return np.unique(movie_ids)[:nmovies]

    def suggest_similar_movies(self, userId: int, movieIds: list[int], count: int):
        """
        Suggest movies similar to the given movieIds for the user.
        """
        assert self.is_fit
        # Get closest movies to the provided movieIds
        similar_movie_ids = self.get_closest_movies(
            movieIds, k=self.movies.shape[0])
        # Remove movies already rated by the user
        user_rated = set(self.ratings[self.ratings.userId == userId].movieId)
        filtered = [mid for mid in np.unique(
            similar_movie_ids) if mid not in user_rated and mid not in movieIds]
        assert np.intersect1d(filtered, movieIds).size == 0
        return np.array(filtered[:count])

    def sort_movies(self, movie_ids=None):
        movies = self.movies
        if movie_ids is not None:
            movies = movies.loc[movies.movieId.isin(movie_ids)]
        movies = movies.sort_values(["movie_avg_rating"], ascending=False)
        return movies.movieId.tolist()

    def get_users_top_movies(self, userIds: list[int], genres=[], relation=None):
        ratings = self.ratings
        condition = ratings.userId.isin(userIds)
        if len(genres) > 0:
            moviesIds = self.movies_with_genres(genres, relation)
            condition &= ratings.movieId.isin(moviesIds)
        ratings = ratings.loc[condition]
        if len(ratings) == 0:
            return []
        ratings = ratings.sort_values("rating", ascending=False)
        return np.unique(ratings.movieId)

    def get_closest_users(self, user_ids, k) -> np.ndarray:
        users_data = self.users.loc[self.users.userId.isin(
            user_ids)].drop("userId", axis=1).values
        if users_data.shape[0] == 0:
            return []
        return self.user_knn.kneighbors(
            users_data,
            n_neighbors=k,
            return_distance=False,
        )[0]

    def get_closest_movies(self, movie_ids, k, genres=[], relation=None):
        movies_data = self.movies.loc[self.movies.movieId.isin(
            movie_ids)].drop("movieId", axis=1).values
        if movies_data.shape[0] == 0:
            return []
        movie_ids = self.movie_knn.kneighbors(
            movies_data,
            n_neighbors=k, return_distance=False,
        )[0]
        if len(genres):
            movie_ids = self.movies_with_genres(
                genres, relation, inids=movie_ids)
        # display(self.movies.loc[self.movieIds.isin(movie_ids)])
        return movie_ids


def load_suggester():
    from .data import MovieLens
    movie_lens = MovieLens.get_instance()
    suggester = Suggester()
    suggester.fit(movie_lens)
    return suggester
