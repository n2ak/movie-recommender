from typing import Optional, Union
import pandas as pd
import numpy as np


class _FeatureStore:
    def __init__(self):
        self.users = pd.DataFrame()
        self.movies = pd.DataFrame()

    def get_movies_features(self, movie_ids: list[int]):
        return self.movies.iloc[movie_ids]

    def get_user_features(self, user_id: int):
        return self.users.iloc[[user_id]]


FeatureStore = _FeatureStore()
