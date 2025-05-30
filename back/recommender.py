from dataclasses import asdict, dataclass
from typing import Optional
from functools import lru_cache
from pydantic import BaseModel as Schema
if True:
    import sys
    import os
    p = os.path.join(os.path.abspath(__file__), "..")
    sys.path.append(p)
    from core.utils import Instance
    from core.modeling.dlrm import DLRM
    from core.suggester import load_suggester, MovieLens


class Recommender(Instance):
    def init(self):
        self.suggester = load_suggester()
        self.model: DLRM = DLRM.load(
            "dlrm.pt",
            model_only=True,
        )

    @staticmethod
    def recommend_movies(data: "Request"):
        self = Recommender.get_instance()
        userId = data.userId
        genres = data.genres
        start = data.start
        count = data.count
        relation = data.relation
        [userId], movieIds = MovieLens.get_instance().map_ids([userId], [])
        if len(movieIds) == 0:
            movieIds = self.suggester.suggest(
                userId,
                count,
                n_neighbor_users=10,
                n_neighbor_movies=10,
                genres=genres,
                relation=relation,
            ).tolist()
            assert len(movieIds) > 0

        return self.recom(userId, movieIds)

    @staticmethod
    def recommend_similar_movies(data: "SimilarMoviesRequest"):
        self = Recommender.get_instance()
        userId = data.userId
        start = data.start
        count = data.count
        movieIds = data.movieIds
        [userId], movieIds = MovieLens.get_instance().map_ids([userId], movieIds)
        movieIds = self.suggester.suggest_similar_movies(
            userId,
            movieIds,
            count
        ).tolist()
        assert len(movieIds) > 0
        return self.recom(userId, movieIds)

    def recom(self, userId, movieIds):
        return self.model.recommend_for_user(
            userId,
            movieIds=movieIds,
            max_rating=10,
            wrap=True
        )


@dataclass
class Response():
    userId: int
    result: list[dict]  # Recommendation
    time: float
    status_code: int
    error: Optional[str] = None

    asdict = asdict


class Request(Schema):
    userId: int
    genres: list[str]
    start: int
    count: Optional[int]
    relation: str


class SimilarMoviesRequest(Schema):
    userId: int
    start: int
    count: Optional[int]
    movieIds: list[int]
