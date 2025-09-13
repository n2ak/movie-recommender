from movie_recommender.logging import logger
import numpy as np
import pandas as pd
import typing
T = typing.TypeVar("T")


def get_env(name: str, default: T) -> T:
    import os
    val = str(os.environ.get(name, default))
    if isinstance(default, bool):
        return val.lower() in ["true", "1"]  # type: ignore
    return default.__class__(val)  # type: ignore


class Env_():
    @property
    def exp_name(self): return get_env("EXP_NAME", "movie_recom")
    @property
    def MAX_RATING(self): return get_env("MAX_RATING", 5)
    @property
    def NUM_BOOST_ROUND(self): return get_env("NUM_BOOST_ROUND", 500)
    @property
    def VERBOSE_EVAL(self): return get_env("VERBOSE_EVAL", 1000)
    @property
    def OPTIMIZE(self): return get_env("OPTIMIZE", False)
    @property
    def TRAIN_SIZE(self): return get_env("TRAIN_SIZE", .9)
    @property
    def EPOCH(self): return get_env("EPOCH", 2)

    @property
    def DB_URL(self): return get_env(
        "DB_URL", "postgresql+psycopg2://admin:password@localhost:5432/mydb")

    @property
    def MLFLOW_TRACKING_URI(self):
        import mlflow
        uri = get_env(
            "MLFLOW_TRACKING_URI",
            "http://localhost:8081"
        )
        mlflow.set_tracking_uri(uri)
        logger.info("mlflow tracking uri is set to: %s", uri)
        return uri

    def dump(self):
        logger.info("*********************ENV***********************")
        logger.info("exp_name: %s", self.exp_name)
        logger.info("MAX_RATING: %s", self.MAX_RATING)
        logger.info("NUM_BOOST_ROUND: %s", self.NUM_BOOST_ROUND)
        logger.info("VERBOSE_EVAL: %s", self.VERBOSE_EVAL)
        logger.info("OPTIMIZE: %s", self.OPTIMIZE)
        logger.info("TRAIN_SIZE: %s", self.TRAIN_SIZE)
        logger.info("EPOCH: %s", self.EPOCH)
        logger.info("DB_URL: %s", self.DB_URL)
        logger.info("MLFLOW_TRACKING_URI: %s", self.MLFLOW_TRACKING_URI)
        logger.info("***********************************************")


Env = Env_()
Env.MLFLOW_TRACKING_URI


def read_parquet(*paths: str):
    return [pd.read_parquet(p) for p in paths]


def mae(logits, y) -> tuple[str, float]:
    max_rating = Env.MAX_RATING
    y = y * max_rating
    logits = logits * max_rating
    return "mae", np.abs(logits - y).mean().item()


def rmse(logits, y) -> tuple[str, float]:
    return "rmse", np.sqrt(np.power(logits - y, 2).mean()).item()
