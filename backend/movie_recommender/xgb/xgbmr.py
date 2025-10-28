import json
import mlflow.xgboost
from movie_recommender.common.logging import Logger
import pandas as pd
import xgboost as xgb
import mlflow
from numpy.typing import NDArray
from typing import Optional
from movie_recommender.common.workflow import (
    download_artifacts, register_last_model_and_try_promote, log_temp_artifacts,
    get_registered_model_run_id, model_uri, make_run_name
)

from ..common.env import XGB_REGISTERED_NAME


class XGBMR():
    MAX_BATCH_SIZE = 256*8*4

    def __init__(self, cols: Optional[list[str]] = None):
        self.cols = cols

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
            model_uri(XGB_REGISTERED_NAME, champion)
        )
        best_model.set_param({"device": device})
        Logger.info("XGB loaded on device: %s", device)

        run_id = get_registered_model_run_id(
            XGB_REGISTERED_NAME, champion=champion
        )
        assert run_id is not None

        custom_params = json.loads(mlflow.get_run(
            run_id).data.params["custom_params"])
        cols = custom_params["cols"]
        model = XGBMR(
            cols=cols
        )
        model.model = best_model
        Logger.info(
            f"Loaded champion model, {XGB_REGISTERED_NAME=}"
        )
        model.run_id = run_id
        return model

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
        with mlflow.start_run(
            run_name=make_run_name("xgb"),
            tags={"model_type": "XGBMR"}
        ) as run:
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
            Logger.info("Done training.")

        register_last_model_and_try_promote(
            registered_name=XGB_REGISTERED_NAME,
            metric_name="val-mae"
        )
        self.run_id = run_id
        return eval_results, run_id
