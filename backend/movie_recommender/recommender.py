from movie_recommender.utils import Timer
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
    from movie_recommender.logging import Logger
    from movie_recommender.env import SS_REGISTERED_NAME, XGB_REGISTERED_NAME, DLRM_REGISTERED_NAME
ModelType = Literal["xgb_cpu", "xgb_cuda", "dlrm_cpu", "dlrm_cuda"]


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
    model: ModelType | Literal["best"]
    temp: float


class SimilarMoviesRequest(Schema):
    userId: int
    start: int
    count: Optional[int]
    movieIds: list[int]
    model: ModelType | Literal["best"]
    temp: float


class Recommender(Singleton):
    max_rating = 5
    champion = True
    current_champions = {}
    registered_names = {
        "dlrm": DLRM_REGISTERED_NAME,
        "xgb": XGB_REGISTERED_NAME,
        "simsearch": SS_REGISTERED_NAME,
    }

    def init(self):
        self.simsearch = SimilaritySearch.load_from_disk()
        self.models: dict[ModelType, Optional[MovieRecommender]] = {
            "dlrm_cpu": None,
            "dlrm_cuda": None,
            "xgb_cpu": None,
            "xgb_cuda": None,
        }

    @staticmethod
    def single(requests: "Request"):
        return Recommender.batched([requests])

    @staticmethod
    def batched(requests: list[Request | SimilarMoviesRequest]):
        self = Recommender.get_instance()
        resp = []

        with Timer(f"simsearch"):
            for i, request in enumerate(requests):
                movieIds = self._preprcoss_request(request)
                assert len(movieIds) > 0
                resp.append(movieIds)

        return self._recom(
            userIds=[r.userId for r in requests],
            movieIds=resp,
            model=requests[0].model,
            temps=[r.temp for r in requests],
            counts=[(r.count or 999999999) for r in requests],
            starts=[r.start for r in requests],
        )

    @classmethod
    def load_all_models(cls, exclude: list[ModelType] = []):
        instance = cls.get_instance()
        models: list[ModelType] = ["xgb_cpu",
                                   "xgb_cuda", "dlrm_cpu", "dlrm_cuda"]
        for model in models:
            if model in exclude or (instance.models[model] is not None):
                continue
            Logger.info(f"Preloading {model=}")
            instance.get_model(model)  # type: ignore

    def get_best_model(self, metric="val-mae") -> MovieRecommender:
        models: list[MovieRecommender] = list(
            self.models.values())  # type: ignore
        assert all([model is not None for model in models])
        models = sorted(
            models, key=lambda model: model.get_stats[metric], reverse=True)
        return models[0]

    def get_model(self, modelname: ModelType | Literal["best"], reload=False) -> MovieRecommender:
        if modelname == "best":
            self.load_all_models()
            return self.get_best_model()
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
                    model = XGBMR.load_model(
                        champion=self.champion, device="cpu")
                case "xgb_cuda":
                    from movie_recommender.modeling.xgbmr import XGBMR
                    model = XGBMR.load_model(
                        champion=self.champion, device="cuda")
                case _:
                    raise Exception(f"{modelname}?")
        self.models[modelname] = model
        return model

    def _preprcoss_request(self, request: Request | SimilarMoviesRequest):
        if isinstance(request, Request):
            return self.simsearch.suggest(
                request.userId,
                n_neighbor_users=10,
                n_neighbor_movies=10,
                genres=tuple(request.genres),
            )
        elif isinstance(request, SimilarMoviesRequest):
            return self.simsearch.suggest_similar_movies(
                request.userId,
                movie_ids=tuple(request.movieIds),
                n_neighbor_movies=10,
            )
        else:
            raise Exception(f"Invalid request type '{type(request)}'")

    def _recom(
        self, userIds: list[int], movieIds: list[list[int]],
        model: ModelType | Literal["best"], temps: list[float], counts: list[int],
        starts: list[int],
    ):
        results = self.get_model(model).recommend_for_users_batched(
            userIds,
            movieIds=movieIds,
            max_rating=self.max_rating,
            temps=temps
        )
        ret = [result[start:start + count]
               for result, start, count in zip(results, starts, counts)]
        return ret

    @classmethod
    def check_for_new_champions(cls):
        from movie_recommender.workflow import get_registered_model_run_id
        recommender = Recommender.get_instance()

        def check(full_name, name):
            run_id = get_registered_model_run_id(
                cls.registered_names[name], champion=True)
            old_run_id = cls.current_champions.get(full_name, run_id)
            new = old_run_id != run_id
            cls.current_champions[full_name] = run_id
            if new:
                Logger.info(f"New champion for model={full_name}")
            return new

        Logger.info("Checking for new champions")
        for model_name in recommender.models:
            name = model_name.removesuffix("_cpu")
            name = name.removesuffix("_cuda")
            if check(model_name, name):
                # we have to reload
                recommender.get_model(model_name, reload=True)

        if check("simsearch", "simsearch"):
            recommender.simsearch = SimilaritySearch.load_from_disk()
