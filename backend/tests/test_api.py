import pytest_asyncio
import pytest
from typing import TypeVar
from httpx import ASGITransport, AsyncClient
from api import app, RecomResponse, RecomRequest
from movie_recommender.recommender import ModelType
from asgi_lifespan import LifespanManager  # pip install 'asgi-lifespan==2.*'


MAX_RATING = 5

T = TypeVar("T")


async def recomend_movies(aclient: AsyncClient, data: RecomRequest):
    json = data.model_dump()
    return await aclient.post('/movies-recom', json=json)


async def check_response(resp_json, data):
    assert resp_json.status_code == 200, resp_json.status_code
    result = RecomResponse(**resp_json.json())
    assert result.status_code == 200, result.error
    assert result.userId == data.userId  # type: ignore

    for res in result.result:
        assert 0 <= res.predicted_rating <= MAX_RATING
    return result


# + {"xgb_cuda","dlrm_cuda"}
MODELS: set[ModelType] = {"xgb_cuda", "dlrm_cuda"}


@pytest_asyncio.fixture
async def manager():
    async with LifespanManager(app) as manager:
        yield manager.app


@pytest_asyncio.fixture
async def aclient(manager):
    async with AsyncClient(transport=ASGITransport(manager), base_url="http://test") as client:
        print("Client is ready")
        yield client


# @pytest.mark.asyncio
# async def test_api(aclient: AsyncClient):
#     for model in MODELS:
#         print(model)
#         data = RecomRequest(
#             userId=1,
#             genres=[],
#             start=0,
#             count=None,
#             model=model,
#             temp=0,
#         )
#         await check_response(await recomend_movies(aclient, data), data)


@pytest.mark.asyncio
async def test_genres(aclient: AsyncClient):
    for model in MODELS:
        data = RecomRequest(
            userId=1,
            genres=["Action"],
            start=0,
            count=None,
            model=model,
            temp=0.5,
        )
        await check_response(await recomend_movies(aclient, data),  data)
