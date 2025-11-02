import os
from movie_recommender.common.workflow import StorageClient
from movie_recommender.common.env import ARTIFACT_ROOT


class _FeatureStore:
    def __init__(self):
        users, movies = StorageClient.get_instance().download_parquet_from_bucket(
            ARTIFACT_ROOT, "users_features", "movies_features"
        )
        self.users = users
        self.movies = movies

    def get_movies_features(self, movie_ids: list[int]):
        return self.movies.iloc[movie_ids]

    def get_user_features(self, user_id: int):
        return self.users.iloc[[user_id]]


FeatureStore = _FeatureStore()
