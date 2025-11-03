import os
from movie_recommender.common.workflow import StorageClient
from movie_recommender.common.env import ARTIFACT_ROOT, BUCKET


class _FeatureStore:
    def __init__(self):
        users, movies = StorageClient.get_instance().read_parquets_from_bucket(
            BUCKET,
            f"{ARTIFACT_ROOT}/users_features",
            f"{ARTIFACT_ROOT}/movies_features"
        )
        self.users = users
        self.movies = movies

    def get_movies_features(self, movie_ids: list[int]):
        return self.movies.iloc[movie_ids]

    def get_user_features(self, user_id: int):
        return self.users.iloc[[user_id]]


FeatureStore = _FeatureStore()
