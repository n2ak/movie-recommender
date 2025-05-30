from fastapi.testclient import TestClient
from api import app, Response, SimilarMoviesRequest
client = TestClient(app)
MAX_RATING = 10


def recomend_movies(data):
    return client.post('/movies-recom', json=data)


def recomend_similar_movies(data):
    return client.post('/recom-similar-movies', json=data)


def check_response(func, data):
    resp_json = func(data)
    assert resp_json.status_code == 200, resp_json.status_code
    result = Response(**resp_json.json())
    assert result.status_code == 200, result.error
    assert result.userId == data["userId"]

    for res in result.result:
        assert 0 <= res["predicted_rating"] <= MAX_RATING


def test_api():
    for relation in ["or", "and"]:
        data = {
            "userId": 1,
            "genres": [],
            "start": 0,
            "count": None,
            "relation": relation
        }
        check_response(recomend_movies, data)


def test_genres():
    for relation in ["or", "and"]:
        data = {
            "userId": 1,
            "genres": ["Action"],
            "start": 0,
            "count": None,
            "relation": relation
        }
        check_response(recomend_movies, data)


def test_api_no_movieIds():
    data = {
        "userId": 1,
        "genres": ["Action", "Drama"],
        "start": 0,
        "count": None,
        "relation": "or"
    }
    check_response(recomend_movies, data)


def test_recomm_similar_movies():
    data = {
        "userId": 1,
        "start": 0,
        "count": None,
        "movieIds": [1],
    }
    check_response(recomend_similar_movies, data)
