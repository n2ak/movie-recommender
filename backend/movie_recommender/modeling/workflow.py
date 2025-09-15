
import functools
from movie_recommender.logging import logger
import numpy as np
from .base import MovieRecommender
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mlflow
import pandas as pd


def load_runs_frame(exp_name, model_type=None, order_by=None):
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        raise Exception(f"No experiment with name: '{exp_name}'.")
    runs: pd.DataFrame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id])  # type: ignore
    if len(runs) == 0:
        raise Exception(f"No runs in experiment with name: '{exp_name}'.")
    runs.rename(columns={
        "tags.mlflow.runName": "run_name",
        "tags.model_type": "model_type",
        "params.model_params": "model_params",
    }, inplace=True)
    if "model_params" not in runs.columns:
        runs["model_params"] = None
    cols = ["run_id", "experiment_id", "status",
            "model_params", "model_type", "run_name"]
    cols += [c for c in runs.columns if c.startswith("metrics.")]
    runs = runs[cols]
    if model_type is not None:
        runs = runs[runs["model_type"] == model_type]

    if order_by:
        runs = runs.sort_values(order_by)
    return runs, experiment.experiment_id


def load_models_frame(exp_name, model_type=None, order_by=None):
    runs, experiment_id = load_runs_frame(
        exp_name, model_type=model_type, order_by=order_by)
    models = mlflow.search_logged_models(experiment_ids=[experiment_id])
    if len(runs) == 0:
        raise Exception(f"No models in experiment with name: '{exp_name}'.")
    models = models.rename(columns={"source_run_id": "run_id"})
    models = models[["run_id", "model_id"]]
    models = runs.merge(models, on="run_id")
    return models


def load_xgboost(model_uri: str, run_name: str):
    if not len(mlflow.artifacts.list_artifacts(artifact_uri=model_uri)):
        raise Exception(f"No artifact from {model_uri=}, {run_name=}")
    import xgboost as xgb
    best_model: xgb.Booster = mlflow.xgboost.load_model(  # type: ignore
        model_uri)
    return best_model


def load_pytorch(model_uri: str, run_name: str):
    if not len(mlflow.artifacts.list_artifacts(artifact_uri=model_uri)):
        raise Exception(f"No artifact from {model_uri=}, {run_name=}")
    return mlflow.pytorch.load_model(model_uri)


def load_best_model(experiment_name, model_type, sortby, run_id=None):
    models = load_models_frame(
        experiment_name, model_type=model_type, order_by=sortby)
    if run_id is not None:
        logger.info("Selecting models with run_id: %s", run_id)
        models = models[models.run_id == run_id]
    if len(models) == 0:
        raise Exception(
            f"No models, {experiment_name=}, {model_type=}, {run_id=}"
        )

    model_row = models.iloc[0]
    model_id = model_row["model_id"]
    run_name = model_row["run_name"]
    run_id = model_row["run_id"]

    model_uri = f"models:/{model_id}"
    return model_uri, run_id, run_name


def log_txt(val: str, name: str):
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(prefix=f"{name}_", suffix=".txt", mode="r+t") as file:
        file.write(val)
        file.flush()
        mlflow.log_artifact(file.name)


def save_figures(figures: dict[str, Figure], run_id, artifact_path=None):
    def savefn(d):
        for name, fig in figures.items():
            fig.savefig(f"{d}/{name}")
    log_temp_artifacts(savefn, run_id=run_id, artifact_path=artifact_path)


def log_temp_artifacts(save_fn, artifact_path=None, run_id=None):
    import tempfile
    import mlflow
    with tempfile.TemporaryDirectory() as f:
        save_fn(f)
        mlflow.log_artifacts(f, artifact_path, run_id=run_id)


@functools.lru_cache(maxsize=16)
def download_artifacts(run_id: str, artifact_path: str):
    print(f"Getting artifacts, {run_id=}, {artifact_path=}")
    files = mlflow.artifacts.list_artifacts(
        run_id=run_id)
    # print("files", files)
    if not len(files):
        raise Exception(f"No files! {run_id=}, {artifact_path=}")

    return mlflow.artifacts.download_artifacts(  # type: ignore
        run_id=run_id,
        artifact_path=artifact_path,
    )


def save_plots(
    model: "MovieRecommender",
    prepare: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_ds,
    test_ds,
    max_rating: int,
    run_id: str
):
    figures = {}
    from movie_recommender.modeling.workflow import save_figures
    from ..utils import report
    user_ids, movie_ids, train_y = prepare(train_ds)
    report(
        model,
        user_ids,
        movie_ids,
        train_y * max_rating,
        max_rating,
        title="Training"
    )
    figures["train.png"] = plt.gcf()
    plt.close()

    user_ids, movie_ids, test_y = prepare(test_ds)
    report(
        model,
        user_ids,
        movie_ids,
        test_y * max_rating,
        max_rating,
        title="Testing"
    )
    figures["test.png"] = plt.gcf()
    plt.close()
    save_figures(figures, run_id=run_id)
