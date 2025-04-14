from core.suggester import load_suggester, Suggester
from pydantic import BaseModel as Schema, ValidationError
from functools import lru_cache
from core.modeling.base import BaseModel
from core.utils import Instance
from enum import Enum
from typing import Optional
import flask as F
from dataclasses import asdict, dataclass
app = F.Flask(__name__)


class ModelEnum(str, Enum):
    DLRM = 'DLRM'
    NCF = "NCF"


@dataclass
class Response():
    result: list
    time: float
    status_code: int
    errors: str = ""
    asdict = asdict


class Request(Schema):
    userId: int
    movieIds: list[int]
    genres: list[str]
    model: ModelEnum
    start: int
    count: Optional[int]
    relation: str


class Recommender(Instance):
    filenames = {
        ModelEnum.DLRM: "dlrm.pt",
        # ModelEnum.NCF: "ncf.pt",
    }
    MODELS = {}

    def init(self):
        self.suggester = load_suggester()

    def get_model(self, name) -> BaseModel:
        models = self.MODELS
        filenames = self.filenames
        if name not in models:
            print("Loading", name)
            models[name] = BaseModel.load(
                filenames[name],
                model_only=True,
            )
            print(f"Model '{name}' is loaded.")
        return models[name]

    @staticmethod
    def recommend(data: Request):
        return Recommender.get_instance()._recommend(
            data.model,
            data.userId,
            tuple(data.movieIds),
            tuple(data.genres),
            data.start,
            data.count,
            data.relation,
        )

    @lru_cache()
    def _recommend(self, model_name, userId, movieIds, genres, start, count, relation):
        print(f"{userId=} {movieIds=} {genres=}")
        if len(movieIds) == 0:
            movieIds = self.suggester.suggest(
                userId,
                10,
                n_neighbor_users=60, n_neighbor_movies=300,
                genres=genres,
                relation=relation,
                deterministic=True,
            )
            assert len(movieIds) > 0
        print("*"*10, "Cache miss")
        model = self.get_model(model_name)
        return model.recommend_for_users(
            **model.prepare(
                userId,
                movieIds=movieIds,
            ),
            start=start,
            count=count,
        )


@app.post("/movies-recom")
def movies_recom():
    response = Response(
        result=None,
        time=0,
        status_code=200,
        errors=None,
    )
    try:
        import time
        # time.sleep(.5)
        data = Request(**F.request.json)
        print("*"*20)
        s = time.monotonic()
        recommendations = Recommender.recommend(
            data
        )
        e = time.monotonic()
        response.time = e-s
        response.result = recommendations
    except ValidationError as e:
        # If the JSON data doesn't match the Pydantic model, return a 400 Bad Request response
        response.errors = {
            str(err["loc"]): err["msg"] for err in e.errors()
        }
        response.status_code = 400
    except Exception as e:
        response.errors = {
            "error": type(e)
        }
        response.status_code = 400
    return F.jsonify(response.asdict()), response.status_code


def warm_up():
    print("Warming up")
    Recommender.get_instance().recommend(Request(
        userId=1,
        movieIds=[1],
        genres=[],
        model=ModelEnum.DLRM,
        start=0,
        count=10,
        relation="and",
    ))


if __name__ == "__main__":
    host = "localhost"
    port = 3333
    warm_up()
    app.run(host, port, debug=True)
