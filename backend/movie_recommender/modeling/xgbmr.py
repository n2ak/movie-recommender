import json
import os

import mlflow.xgboost
from movie_recommender.logging import Logger
import pandas as pd
from .base import MovieRecommender
import xgboost as xgb
import mlflow
from numpy.typing import NDArray
from typing import Optional
from movie_recommender.workflow import (
    download_artifacts, register_last_model_and_try_promote, log_temp_artifacts, get_registered_model_run_id, model_uri, get_run_metrics
)


registered_name = os.environ["XGB_REGISTERED_NAME"]


class XGBMR(MovieRecommender[NDArray]):
    MAX_BATCH_SIZE = 256*8*4

    def __init__(self, users: pd.DataFrame, movies: pd.DataFrame, cols: Optional[list[str]] = None):
        self._set_data(users=users, movies=movies, cols=cols)

    def predict(self, batch, max_rating):
        pred = self.model.predict(xgb.DMatrix(batch))
        return pred*max_rating

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, champion=True, device="cpu"):
        import pathlib
        best_model: xgb.Booster = mlflow.xgboost.load_model(  # type: ignore
            model_uri(registered_name, champion)
        )
        best_model.set_param({"device": device})
        Logger.info("XGB loaded on device: %s", device)

        run_id = get_registered_model_run_id(
            registered_name, champion=champion
        )
        assert run_id is not None

        custom_params = json.loads(mlflow.get_run(
            run_id).data.params["custom_params"])
        cols = custom_params["cols"]
        assert isinstance(cols, list), type(cols)
        artifact_path = download_artifacts(
            run_id=run_id,
            artifact_path="resources",
        )

        def read_file(name) -> pd.DataFrame:
            return pd.read_parquet(pathlib.Path(artifact_path) / f"{name}.parquet")

        model = XGBMR(
            read_file("users"),
            read_file("movies"),
            cols=cols
        )

        model.model = best_model
        Logger.info(
            f"Loaded champion model, {registered_name=}"
        )
        model.run_id = run_id
        return model

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
        batch = batch[self.cols]

        def chunk_split(arr, n):
            return [arr[i:i + n] for i in range(0, len(arr), n)]
        batch_size = self.MAX_BATCH_SIZE
        return chunk_split(batch, batch_size)  # type: ignore

    def _prepare_simple(  # type: ignore
        self,
        user_ids: list[int],
        movie_ids: list[int],
    ):
        user_data = self.users.loc[user_ids]
        movie_data = self.movies.loc[movie_ids]

        user_data.reset_index(drop=True, inplace=True)
        movie_data.reset_index(drop=True, inplace=True)

        batch = pd.concat([user_data, movie_data], axis=1)
        batch = batch[self.cols]
        return batch

    def _set_data(self, users: pd.DataFrame, movies: pd.DataFrame, cols=None):

        user_cols = users.columns.tolist()
        movie_cols = movies.columns.tolist()
        if cols is None:
            cols = user_cols + movie_cols

        assert users.index.name == "user_id"
        assert users.index.nunique() == users.index.max()+1

        assert movies.index.name == "movie_id"
        assert movies.index.nunique() == movies.index.max()+1

        self.movies = movies
        self.users = users
        self.cols = cols

    def fit(
        self,
        params: dict,
        X, y,
        experiment_name: str,
        eval_set=None,
        **training_config
    ) -> tuple[dict, str]:
        mlflow.set_experiment(experiment_name)
        mlflow.xgboost.autolog()  # type: ignore

        train_matrix = xgb.DMatrix(X, y)
        params = params | dict(
            objective='reg:squarederror',
            device="cuda",
        )

        evals = [(train_matrix, "train")]
        if eval_set:
            evals.append((xgb.DMatrix(*eval_set), "val"))
        with mlflow.start_run(tags={"model_type": "XGBMR"}) as run:
            run_id = run.info.run_id
            Logger.info("Run id: %s", run_id)

            eval_results = {}
            self.model = xgb.train(
                params=params,
                dtrain=train_matrix,
                evals=evals,
                evals_result=eval_results,
                **training_config
            )
            if "custom_metric" in training_config:
                del training_config["custom_metric"]
            mlflow.log_param("custom_params", json.dumps(dict(
                training_config=training_config,
                cols=self.cols
            )))
            self.log_artifacts()
            Logger.info("Done training.")

        register_last_model_and_try_promote(
            registered_name=registered_name,
            metric_name="val-mae"
        )
        self.run_id = run_id
        return eval_results, run_id

    def log_artifacts(self):
        Logger.info("Logging artifacts.")

        def save(dir):
            self.movies.to_parquet(f"{dir}/movies.parquet")
            self.users.to_parquet(f"{dir}/users.parquet")
        log_temp_artifacts(save, artifact_path="resources")
