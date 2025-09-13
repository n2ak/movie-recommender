from typing import Callable, TypeVar, Any
from fastapi.testclient import TestClient
from api import app, RecomResponse, RecomRequest
from movie_recommender.recommender import ModelType


client = TestClient(app)

MAX_RATING = 5

T = TypeVar("T")


def recomend_movies(data: RecomRequest):
    json = data.model_dump()
    return client.post('/movies-recom', json=json)


def check_response(func: Callable[[T], Any], data: T):
    resp_json = func(data)
    assert resp_json.status_code == 200, resp_json.status_code
    result = RecomResponse(**resp_json.json())
    assert result.status_code == 200, result.error
    assert result.userId == data.userId  # type: ignore

    for res in result.result:
        assert 0 <= res.predicted_rating <= MAX_RATING
    return result


MODELS: list[ModelType] = ["xgb_cpu", "xgb_cuda", "dlrm_cpu", "dlrm_cuda"]


def test_api():
    for model in MODELS:
        data = RecomRequest(
            userId=1,
            genres=[],
            start=0,
            count=None,
            model=model,
            temp=0,
        )
        check_response(recomend_movies, data)


def test_genres():
    for model in MODELS:
        data = RecomRequest(
            userId=1,
            genres=["Action"],
            start=0,
            count=None,
            model=model,
            temp=0,
        )
        check_response(recomend_movies, data)
