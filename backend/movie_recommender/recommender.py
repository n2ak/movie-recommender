from movie_recommender.utils import Timer
import os
from typing import Optional, Literal
from pydantic import BaseModel as Schema
if True:
    import sys
    import os
    p = os.path.join(os.path.abspath(__file__), "..")
    sys.path.append(p)
    from movie_recommender.modeling.base import Recommendation, MovieRecommender
    from movie_recommender.utils import Singleton
    from movie_recommender.sim_search import SimilaritySearch

ModelType = Literal["xgb_cpu", "xgb_cuda", "dlrm_cpu", "dlrm_cuda"]


class Recommender(Singleton):
    max_rating = 5
    champion = True
    current_champions = {}
    registered_names = {
        "dlrm": os.environ["DLRM_REGISTERED_NAME"],
        "xgb": os.environ["XGB_REGISTERED_NAME"],
        "simsearch": os.environ["SS_REGISTERED_NAME"],
    }

    def init(self):
        self.simsearch = SimilaritySearch.load_from_disk()
        self.models: dict[ModelType, Optional[MovieRecommender]] = {
            "dlrm_cpu": None,
            "dlrm_cuda": None,
            "xgb_cpu": None,
            "xgb_cuda": None,
        }

    @classmethod
    def load_all_models(cls, exclude: list[ModelType] = []):
        instance = cls.get_instance()
        for model in ["xgb_cpu", "xgb_cuda", "dlrm_cpu", "dlrm_cuda"]:
            if model in exclude:
                continue
            print(f"Preloading {model=}")
            instance.get_model(model)  # type: ignore

    def get_model(self, modelname: ModelType, reload=False) -> MovieRecommender:
        model = self.models[modelname]
        if model is None or reload:
            match modelname:
                case "dlrm_cpu":
                    from movie_recommender.modeling.dlrm import DLRM
                    model = DLRM.load(champion=self.champion, device="cpu")
                case "dlrm_cuda":
                    from movie_recommender.modeling.dlrm import DLRM
                    model = DLRM.load(champion=self.champion, device="cuda")
                case "xgb_cpu":
                    from movie_recommender.modeling.xgbmr import XGBMR
                    model = XGBMR().load_model(champion=self.champion, device="cpu")
                case "xgb_cuda":
                    from movie_recommender.modeling.xgbmr import XGBMR
                    model = XGBMR().load_model(champion=self.champion, device="cuda")
                case _:
                    raise Exception(f"{modelname}?")
        self.models[modelname] = model
        return model

    @staticmethod
    def single(requests: "Request"):
        return Recommender.batched([requests])

    @staticmethod
    def batched(requests: list["Request"]):
        # print(f"Processing a batch of {len(requests)} requests")
        self = Recommender.get_instance()
        resp = []
        import time
        s = time.monotonic()
        with Timer(f"simsearch"):
            for i, request in enumerate(requests):
                movieIds = self.simsearch.suggest(
                    request.userId,
                    n_neighbor_users=10,
                    n_neighbor_movies=10,
                    genres=tuple(request.genres),
                )
                assert len(movieIds) > 0
                resp.append(movieIds)
        e = time.monotonic()
        return self.recom(
            userIds=[r.userId for r in requests],
            movieIds=resp,
            model=requests[0].model,
            temps=[r.temp for r in requests],
            counts=[(r.count or 999999999) for r in requests],
            starts=[r.start for r in requests],
        )

    def recom(
        self, userIds: list[int], movieIds: list[list[int]],
        model: ModelType, temps: list[float], counts: list[int],
        starts: list[int],
    ):
        results = self.get_model(model).recommend_for_users_batched(
            userIds,
            movieIds=movieIds,
            max_rating=self.max_rating,
            temps=temps,
            wrap=True,
        )
        ret = [result[start:start + count]
               for result, start, count in zip(results, starts, counts)]
        return ret

    @classmethod
    def check_for_new_champions(cls):
        from movie_recommender.workflow import get_champion_run_id
        recommender = Recommender.get_instance()

        def check(full_name, name):
            run_id = get_champion_run_id(cls.registered_names[name])
            old_run_id = cls.current_champions.get(full_name, run_id)
            new = old_run_id != run_id
            cls.current_champions[full_name] = run_id
            if new:
                print(f"New champion for model={full_name}")
            return new

        print("Checking for new champions")
        for model_name in recommender.models:
            name = model_name.removesuffix("_cpu")
            name = name.removesuffix("_cuda")
            if check(model_name, name):
                # we have to reload
                recommender.get_model(model_name, reload=True)

        if check("simsearch", "simsearch"):
            recommender.simsearch = SimilaritySearch.load_from_disk()


class Response(Schema):
    userId: int
    result: list[Recommendation]  # Recommendation
    time: float
    status_code: int
    error: Optional[str] = None


class Request(Schema):
    userId: int
    genres: list[str]
    start: int
    count: Optional[int]
    model: ModelType
    temp: float


class SimilarMoviesRequest(Schema):
    userId: int
    start: int
    count: Optional[int]
    movieIds: list[int]
    model: ModelType
    temp: float
