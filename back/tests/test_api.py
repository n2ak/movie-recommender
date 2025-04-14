import json
import pytest


@pytest.fixture
def app():
    from api import app
    app.config.update({
        "TESTING": True,
    })
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


MODELS = [
    "DLRM",
    # "NCF",
]


def test_api(client):
    for model_name in MODELS:
        for relation in ["or", "and"]:
            data = {
                "userId": 1,
                "movieIds": [1, 2],
                "model": model_name,
                "genres": [],
                "start": 0,
                "count": None,
                "relation": relation
            }
            _test(data, client)


def test_genres(client):
    for model_name in MODELS:
        for relation in ["or", "and"]:
            data = {
                "userId": 1,
                "movieIds": [],
                "model": model_name,
                "genres": ["Fantasy"],
                "start": 0,
                "count": None,
                "relation": relation
            }
            _test(data, client)


def test_api_no_movieIds(client):
    for model_name in MODELS:
        data = {
            "userId": 1,
            "movieIds": [],
            "model": model_name,
            "genres": [],
            "start": 0,
            "count": None,
            "relation": "or"
        }
        _test(data, client)


# def test_user_norating(client):
#     for model_name in MODELS:
#         data = {
#             "userId": 0,
#             "movieIds": [1, 2],
#             "model": model_name,
#         }
#         _test(data, client)


def _test(data, client):
    response = client.post('/movies-recom', json=data)
    result = json.loads(response.get_data().decode())
    assert response.status_code == 200, result["errors"]
    for r in result["result"]:
        assert r["userId"] == data["userId"]
        if "movieIds" in data and len(data["movieIds"]):
            assert r["movieId"] in data["movieIds"]
        assert 0 <= r["predicted_rating"] <= 5
