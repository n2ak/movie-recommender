import torch
import numpy as np
from litserve import LitAPI
from numpy.typing import NDArray

from movie_recommender.simsearch.sim_search import SimilaritySearch
from movie_recommender.common.feature_store import FeatureStore


class API(LitAPI):
    def setup(self, device):
        self.simsearch = SimilaritySearch.load_from_disk()
        self.feature_store = FeatureStore

    def _suggest(self, request):
        userId = request["userId"]
        type = request["type"]
        if type == "recommend":
            movieIds = self.simsearch.suggest(
                request.userId,
                n_neighbor_users=10,
                n_neighbor_movies=10,
                genres=tuple(request.genres),
            )
        elif type == "similar":
            movieIds = self.simsearch.suggest_similar_movies(
                request.userId,
                movie_ids=tuple(request.movieIds),
                n_neighbor_movies=10,
            )
        else:
            raise Exception(f"Invalid request {type=}")
        return userId, list(movieIds)

    def decode_request(self, request):
        user_id, movie_ids = self._suggest(request)
        movies = self.feature_store.get_movies_features(
            movie_ids)  # type: ignore
        users = self.feature_store.get_user_features(user_id)
        input = self._prepare(users, movies)
        return input

    def encode_response(self, output):
        return {
            "output": output
        }

    def _prepare(self, users, movies):
        raise NotImplementedError()

    def apply_temp(
        self,
        ratings: list[float] | NDArray[np.float32],
        userIds: list[int] | NDArray[np.int32],
        movieIds: list[int] | NDArray[np.int32],
        temp: float,
        count=10,
    ):
        assert len(ratings) == len(userIds) == len(
            movieIds), (len(ratings), len(userIds), len(movieIds))
        ratings = np.asarray(ratings, dtype=np.float32)
        userIds = np.asarray(userIds, dtype=np.int32)
        movieIds = np.asarray(movieIds, dtype=np.int32)

        if temp != 0:
            assert 0 < temp <= 1

            def multinomial(*arrs: NDArray, count) -> list[NDArray]:
                # TODO use numpy
                count = min(count, len(arrs[0]))
                array = torch.softmax(torch.from_numpy(arrs[0]), -1)
                array /= temp
                indices = torch.multinomial(array, count).tolist()
                return [arr[indices] for arr in arrs]

            ratings, userIds, movieIds = multinomial(
                ratings, userIds, movieIds,
                count=count,  # TODO use counts
            )

        ratings, userIds, movieIds = self.sort(
            ratings, userIds, movieIds)
        return ratings, userIds, movieIds
