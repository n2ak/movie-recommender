import mlflow
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsTransformer
from movie_recommender.common.feature_store import FeatureStore
from movie_recommender.common.logging import Logger
from movie_recommender.common.workflow import (
    register_last_model, promote_model_to_champion, model_uri, get_registered_model_run_id,
    make_run_name
)
import functools
from movie_recommender.common.env import SS_REGISTERED_NAME


class SimilaritySearch(mlflow.pyfunc.PythonModel):  # type: ignore
    run_id: str

    def __init__(self, n_neighbors_user=5, n_neighbors_movie=5) -> None:
        n_neighbors_movie = int(n_neighbors_movie)
        n_neighbors_user = int(n_neighbors_user)

        self.movie_knn = KNeighborsTransformer(n_neighbors=n_neighbors_movie)
        self.user_knn = KNeighborsTransformer(n_neighbors=n_neighbors_user)
        self.is_fit = False
        self._params = dict(
            n_neighbors_movie=n_neighbors_movie,
            n_neighbors_user=n_neighbors_user,
        )

    def fit(self, ratings: pd.DataFrame):

        self.ratings = ratings
        self.users = FeatureStore.users
        self.movies = FeatureStore.movies
        self.all_genres = [c.removeprefix("movie_genre_")
                           for c in self.movies.columns if c.startswith("movie_genre_")]

        self.movie_knn.fit(self.__movies_data(self.movies))
        self.user_knn.fit(self.__users_data(self.users))
        self.is_fit = True

        return self

    @classmethod
    def load_from_disk(cls, champion=True) -> "SimilaritySearch":
        model = mlflow.pyfunc.load_model(
            model_uri(SS_REGISTERED_NAME, champion)
        ).unwrap_python_model()
        model.run_id = get_registered_model_run_id(
            SS_REGISTERED_NAME, champion=champion)
        return model  # type: ignore

    def __movies_data(self, movies: pd.DataFrame):
        return movies.drop(columns=["movie_id"])

    def __users_data(self, users: pd.DataFrame):
        return users.drop(columns=["user_id"])

    def map_genres(self, genres: list[str]):
        diff = self.setdiff(genres, self.all_genres, keep_order=False)
        if len(diff) != 0:
            raise Exception(
                f"Invalid genres: {diff}, possible genres: {self.all_genres}")
        genres = ["movie_genre_" + g for g in genres]
        return genres

    def movies_with_genres(self, genres: list[str], ids=None) -> pd.DataFrame:
        movies = self.movies
        genres = self.map_genres(genres)
        if ids is not None:
            movies = self._get_movies(ids)
        movies = movies.loc[np.logical_or.reduce(movies[genres], axis=1)]
        return movies

    def ratings_with_genres(self, genres):
        ratings = self.ratings
        genres = self.map_genres(genres)
        ratings = ratings.loc[np.logical_or.reduce(ratings[genres], axis=1)]
        return ratings

    @functools.lru_cache(maxsize=256)
    def suggest(
        self,
        user_id: int,
        genres: list[str],
        n_neighbor_users=100,
        n_neighbor_movies=100,
        add_closest_users=True,
        filter_watched=True
    ):
        """
        Get most relevant movies to user, and users close to the user.
        """
        assert isinstance(user_id, int)
        assert isinstance(genres, (list, tuple))
        assert self.is_fit
        genres = list(genres)  # because of lru_cache

        # top movies that the user has already rated
        top_movie_ids = self.get_users_top_movies_ids(
            [user_id], genres=genres,
        )
        if len(top_movie_ids):
            # get closest movies to user movies
            closest_movies_for_user = self.get_closest_movies(
                top_movie_ids,
                n_neighbor_movies,
                genres=genres,
            )[:10]
            # get similar users
            ret = closest_movies_for_user
            if add_closest_users:

                closest_users = self.get_closest_users(
                    [user_id], n_neighbor_users)[:10]
                if len(closest_users):
                    # top movies for colsest users
                    top_movies_for_similar_users = self.get_users_top_movies_ids(
                        closest_users,
                        genres=genres,
                    )
                    ret = np.concatenate(
                        [closest_movies_for_user, top_movies_for_similar_users]
                    )

            # remove already watched moves
            already_watched = self.ratings[self.ratings.user_id.isin(
                [user_id])].movie_id.unique()
            if filter_watched:
                ret = self.setdiff(ret, already_watched)
        else:
            movies = self.movies
            if genres:
                movies = self.movies[self.movies.genres.isin(genres)]
            ret = movies.movie_id.values
        return ret

    @functools.lru_cache(maxsize=256)
    def suggest_similar_movies(
        self,
        user_id: int,
        movie_ids: list[int],
        n_neighbor_movies=100,
        filter_watched=True
    ):
        """
        Get most relevant movies to user, and users close to the user.
        """
        assert isinstance(user_id, int)
        assert isinstance(movie_ids, (list, tuple))
        assert self.is_fit
        movie_ids = list(movie_ids)  # because of lru_cache

        ret = self.get_closest_movies(
            movie_ids,
            n_neighbor_movies,
        )[:300]

        if filter_watched:
            already_watched = self.ratings[self.ratings.user_id.isin(
                [user_id])].movie_id.unique()
            diff = self.setdiff(ret, already_watched)
            if len(diff) > 0:
                ret = diff
        return ret

    def get_users_top_movies_ids(self, user_ids, genres=[]) -> np.ndarray:
        # pd.DataFrame(dict(user_ids=user_ids))
        ratings = self.ratings
        ratings = ratings[ratings.user_id.isin(user_ids)]
        if genres:
            ratings = self.ratings_with_genres(genres)
        ratings = ratings.sort_values("rating", ascending=False)
        return ratings.movie_id.drop_duplicates().values  # type: ignore

    def get_closest_users(self, user_ids, k, ret_df=False):
        users_data = self._get_users(user_ids)
        if users_data.shape[0] == 0:
            return []
        new_user_ids = self.user_knn.kneighbors(
            self.__users_data(users_data),
            n_neighbors=k,
            return_distance=False,
        ).T.flatten()  # type: ignore
        new_user_ids = unique_keep_order(new_user_ids)
        new_user_ids = self.users.iloc[new_user_ids].user_id
        new_user_ids = self.setdiff(new_user_ids, user_ids)
        # if ret_df:
        #     return self._get_users(ids)
        return new_user_ids

    def get_closest_movies(self, movie_ids, k, genres=[], ret_df=False):
        movies = self._get_movies(movie_ids)
        if movies.shape[0] == 0:
            return []
        new_movie_ids = self.movie_knn.kneighbors(
            self.__movies_data(movies),
            n_neighbors=k, return_distance=False,
            # transpose to get the closest movies of each movie
        ).T.flatten()  # type: ignore
        new_movie_ids = unique_keep_order(new_movie_ids)
        new_movie_ids = self.movies.iloc[new_movie_ids].movie_id
        new_movie_ids = self.setdiff(new_movie_ids, movie_ids)
        if len(genres):
            return self.movies_with_genres(
                genres,
                ids=new_movie_ids
            ).movie_id
            # if ret_df:
        return new_movie_ids

    def setdiff(self, arr1, arr2, keep_order=True):
        if not keep_order:
            return np.setdiff1d(arr1, arr2)
        res = ([int(i) for i in arr1 if i not in arr2])
        return res

    def _get_movies(self, ids) -> pd.DataFrame:
        return self.movies.loc[ids]

    def _get_users(self, ids) -> pd.DataFrame:
        return self.users.loc[ids]

    def save(self, exp_name):
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(
            run_name=make_run_name("simsearch"),
            tags={"model_type": "SimilaritySearch"}
        ) as run:
            mlflow.log_params(self._params)
            run_id: str = run.info.run_id
            Logger.info("Run id: %s", run_id)
            info = mlflow.pyfunc.log_model(python_model=self)

        version = register_last_model(
            registered_name=SS_REGISTERED_NAME).version
        promote_model_to_champion(SS_REGISTERED_NAME, version)
        return run_id


class B:
    def __init__(self):
        self.a = 1


def unique_keep_order(arr):
    unique_elements, first_occurrence_indices = np.unique(
        arr, return_index=True)
    sorted_indices = np.sort(first_occurrence_indices)
    result = arr[sorted_indices]
    return result
