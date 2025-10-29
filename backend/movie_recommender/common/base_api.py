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
        # TODO: use count
        user_id = request["userId"]
        type = request["type"]
        if type == "recommend":
            movie_ids = self.simsearch.suggest(
                user_id,
                n_neighbor_users=10,
                n_neighbor_movies=10,
                genres=tuple(),  # TODO
            )
        elif type == "similar":
            movie_ids = tuple(request["userId"])
            movie_ids = self.simsearch.suggest_similar_movies(
                user_id,
                movie_ids=movie_ids,
                n_neighbor_movies=10,
            )
        else:
            raise Exception(f"Invalid request {type=}")
        return user_id, list(movie_ids)

    def decode_request(self, request, context):
        user_id, movie_ids = self._suggest(request)
        context["meta"] = user_id, movie_ids, request.get("temp", 0)

        movies = self.feature_store.get_movies_features(
            movie_ids)  # type: ignore
        users = self.feature_store.get_user_features(user_id)
        input = self._prepare(users, movies)
        return input

    def encode_response(self, output, context: dict):
        user_id, movie_ids, temp = context["meta"]
        ratings, _, movie_ids = self.apply_temp(
            ratings=output,
            user_ids=[user_id]*len(movie_ids),
            movie_ids=movie_ids,
            temp=temp
        )
        return [
            {
                'predictedRating': r,
                'movieId': m,
                'userId': user_id
            }
            for r, m in zip(ratings.tolist(), movie_ids.tolist())
        ]

    def _prepare(self, users, movies):
        raise NotImplementedError()

    def apply_temp(
        self,
        ratings: list[float] | NDArray[np.float32],
        user_ids: list[int] | NDArray[np.int32],
        movie_ids: list[int] | NDArray[np.int32],
        temp: float,
        count=10,
    ):
        assert len(ratings) == len(user_ids) == len(
            movie_ids), (len(ratings), len(user_ids), len(movie_ids))
        ratings = np.asarray(ratings, dtype=np.float32)
        user_ids = np.asarray(user_ids, dtype=np.int32)
        movie_ids = np.asarray(movie_ids, dtype=np.int32)

        if temp != 0:
            assert 0 < temp <= 1

            def multinomial(*arrs: NDArray, count) -> list[NDArray]:
                # TODO use numpy
                count = min(count, len(arrs[0]))
                array = torch.softmax(torch.from_numpy(arrs[0]), -1)
                array /= temp
                indices = torch.multinomial(array, count).tolist()
                return [arr[indices] for arr in arrs]

            ratings, user_ids, movie_ids = multinomial(
                ratings, user_ids, movie_ids,
                count=count,  # TODO use counts
            )

        ratings, user_ids, movie_ids = self.sort(ratings, user_ids, movie_ids)
        return ratings, user_ids, movie_ids

    def sort(self, *arrs: NDArray):
        sorted_indices = np.argsort(arrs[0])[::-1]
        return [arr[sorted_indices] for arr in arrs]
