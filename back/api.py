from typing import TypeVar, Callable, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI
if True:
    import sys
    sys.path.append(".")
    from core.data import BackendError
    from recommender import Request, Response, Recommender, SimilarMoviesRequest


app = FastAPI()
T = TypeVar("T")


@app.post("/movies-recom")
async def movies_recom(data: Request):
    return handle(data, Recommender.recommend_movies)


@app.post("/recom-similar-movies")
async def recom_similar_movies(data: SimilarMoviesRequest):
    return handle(data, Recommender.recommend_similar_movies)


def handle(request: T, func: Callable[[T], Any]):
    response = Response(
        userId=request.userId,
        result=[],
        time=0,
        status_code=200,
    )
    try:
        import time
        s = time.monotonic()
        recommendations = func(request)
        e = time.monotonic()
        response.time = e-s
        response.result = [r.asdict() for r in recommendations]
    except Exception as e:
        print("*****Error******")
        if isinstance(e, BackendError):
            response.error = e.msg
            print("BackendError:", e.msg)
        else:
            print("Exception:", e)
        response.status_code = 500
        print("****************")
    return response.asdict()
