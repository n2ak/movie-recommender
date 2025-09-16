import os

import mlflow.xgboost
from movie_recommender.logging import logger
import numpy as np
import pandas as pd
from .base import MovieRecommender
import xgboost as xgb
import mlflow
from numpy.typing import NDArray
from movie_recommender.workflow import (
    download_artifacts, register_last_model_and_try_promote, log_temp_artifacts, get_champion_run_id, model_uri
)


registered_name = os.environ["XGB_REGISTERED_NAME"]


class XGBMR(MovieRecommender[NDArray]):
    MAX_BATCH_SIZE = 256*8*4

    def __init__(
        self,
        **kwargs,
    ):
        params = dict(
            objective='reg:squarederror',
            device="cuda",
        ) | kwargs
        self.params = params

    def predict(self, batch, max_rating):
        # print("XGB batch length:", X.shape)
        pred = self.model.predict(xgb.DMatrix(batch))
        return pred*max_rating

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, champion=True, device="cpu"):
        import pathlib
        best_model: xgb.Booster = mlflow.xgboost.load_model(  # type: ignore
            model_uri(registered_name, champion)
        )
        best_model.set_param({"device": device})
        print("XGB loaded on device:", device)

        artifact_path = download_artifacts(
            run_id=get_champion_run_id(registered_name),
            artifact_path="resources",
        )

        def read_file(name) -> pd.DataFrame:
            return pd.read_parquet(pathlib.Path(artifact_path) / f"{name}.parquet")

        self.movies = read_file("movies")
        self.users = read_file("users")
        self.model: xgb.Booster = best_model
        logger.info(
            f"Loaded champion model, {registered_name=}"
        )
        return self

    def _prepare_batch(
        self,
        user_ids: list[int],
        movie_ids_list: list[list[int]],
    ) -> list[NDArray]:
        datas = []
        for user_id, movie_ids in zip(user_ids, movie_ids_list):
            user_data = self.users.loc[[user_id]]
            movie_data = self.movies.loc[movie_ids]
            data = pd.merge(user_data, movie_data, how="cross")
            datas.append(data)
        batch = pd.concat(datas, axis=0)
        # batch.drop(["user_id", "movie_id"], inplace=True)

        def chunk_split(arr, n):
            return [arr[i:i + n] for i in range(0, len(arr), n)]
        batch_size = self.MAX_BATCH_SIZE
        return chunk_split(batch, batch_size)  # type: ignore

    def set_data(self, users: pd.DataFrame, movies: pd.DataFrame):

        user_cols = users.columns.tolist()
        movie_cols = movies.columns.tolist()
        self.cols = user_cols + movie_cols

        assert users.index.name == "user_id"
        assert users.index.nunique() == users.index.max()+1

        assert movies.index.name == "movie_id"
        assert movies.index.nunique() == movies.index.max()+1

        self.movies = movies
        self.users = users

    def fit(
        self,
        X, y,
        eval_set,
        experiment_name: str,
        **training_config
    ) -> tuple[dict, str]:
        mlflow.set_experiment(experiment_name)
        mlflow.xgboost.autolog()  # type: ignore
        train_matrix = xgb.DMatrix(X, y)
        test_matrix = xgb.DMatrix(*eval_set)

        with mlflow.start_run(tags={"model_type": "XGBMR"}) as run:
            run_id = run.info.run_id
            logger.info("Run id: %s", run_id)
            mlflow.log_params(self.params)
            mlflow.log_params(training_config)

            eval_results = {}
            self.model = xgb.train(
                params=self.params,
                dtrain=train_matrix,
                evals=[(test_matrix, "val"),
                       (train_matrix, "train")],
                evals_result=eval_results,
                **training_config
            )

            self.log_artifacts()
            logger.info("Done training.")
        register_last_model_and_try_promote(
            registered_name=registered_name,
            metric_name="val_loss"
        )
        return eval_results, run_id

    def log_artifacts(self):
        logger.info("Logging artifacts.")

        def save(dir):
            self.movies.to_parquet(f"{dir}/movies.parquet")
            self.users.to_parquet(f"{dir}/users.parquet")
        log_temp_artifacts(save, artifact_path="resources")
