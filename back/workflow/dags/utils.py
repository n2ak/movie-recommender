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
        uri = get_env(
            "MLFLOW_TRACKING_URI",
            "http://localhost:8081"
        )
        print("mlflow tracking uri is set to: %s", uri)
        return uri

    def dump(self):
        print("*********************ENV***********************")
        print("exp_name: %s", self.exp_name)
        print("MAX_RATING: %s", self.MAX_RATING)
        print("NUM_BOOST_ROUND: %s", self.NUM_BOOST_ROUND)
        print("VERBOSE_EVAL: %s", self.VERBOSE_EVAL)
        print("OPTIMIZE: %s", self.OPTIMIZE)
        print("TRAIN_SIZE: %s", self.TRAIN_SIZE)
        print("EPOCH: %s", self.EPOCH)
        print("DB_URL: %s", self.DB_URL)
        print("MLFLOW_TRACKING_URI: %s", self.MLFLOW_TRACKING_URI)
        print("***********************************************")


Env = Env_()
Env.MLFLOW_TRACKING_URI


def read_parquet(*paths: str):
    import pandas as pd
    return [pd.read_parquet(p) for p in paths]
