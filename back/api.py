from core.suggester import load_suggester, Suggester
from pydantic import BaseModel as Schema, ValidationError
from functools import lru_cache
from core.modeling.base import BaseModel
from enum import Enum
from typing import Optional
import flask as F

app = F.Flask(__name__)


class ModelEnum(str, Enum):
    DLRM = 'DLRM'
    NCF = "NCF"


class Request(Schema):
    userId: int
    movieIds: list[int] = []
    model: ModelEnum = ModelEnum.DLRM
    start: int = 0
    count: Optional[int] = None


class Recommender:
    filenames = {
        ModelEnum.DLRM: "dlrm.pt",
        ModelEnum.NCF: "ncf.pt",
    }
    MODELS = {}
    suggester: Suggester
    instance: "Recommender" = None

    @staticmethod
    def get_instance():
        if Recommender.instance is None:
            recommender = Recommender()
            recommender.suggester = load_suggester()
            Recommender.instance = recommender
        return Recommender.instance

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
    def recommend(model_name, userId, movieIds, start, count):
        return Recommender.get_instance()._recommend(
            model_name, userId, movieIds, start, count
        )

    @lru_cache()
    def _recommend(self, model_name, userId, movieIds, start, count):
        if len(movieIds) == 0:
            movieIds = self.suggester.suggest(
                userId,
                10,
                n_neighbor_users=60, n_neighbor_movies=300,
            )
            assert len(movieIds) > 0
        print("*"*10, "Cache hit")
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
def get_movies_recom():
    try:
        import time
        # time.sleep(.5)
        data = Request(**F.request.json)
        print("*"*20)
        s = time.monotonic()
        recommendations = Recommender.recommend(
            data.model, data.userId,
            tuple(data.movieIds),
            data.start, data.count,
        )
        e = time.monotonic()
        return F.jsonify({
            "time": e-s,
            "result": recommendations,
        })
    except ValidationError as e:
        # If the JSON data doesn't match the Pydantic model, return a 400 Bad Request response
        return F.jsonify({'errors': {
            str(err["loc"]): err["msg"] for err in e.errors()
        }}), 400


if __name__ == "__main__":
    host = "localhost"
    port = 3333
    app.run(host, port, debug=True)
