
import os
import mlflow
import functools
import numpy as np
import pandas as pd
import mlflow.artifacts
from minio import Minio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Callable, Optional


_mlflowClient: Optional[mlflow.MlflowClient] = None
minioClient: Optional[Minio] = None


def connect_minio():
    global minioClient
    import os
    if minioClient is not None:
        return
    endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    secure = endpoint.startswith("https://")
    endpoint_stripped = endpoint.replace(
        "http://", "").replace("https://", "")
    print("Secure", secure)
    minioClient = Minio(
        endpoint=endpoint_stripped,
        access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region=os.environ["AWS_DEFAULT_REGION"],
        secure=secure,
    )
    print()
    print("Available Buckets", minioClient.list_buckets())
    print("Connected to minio on:", endpoint)


def connect_mlflow():
    global _mlflowClient
    if _mlflowClient is not None:
        return
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    _mlflowClient = mlflow.MlflowClient(uri)
    print("Connected to mlflow on:", uri)


def get_mlflow_client():
    assert _mlflowClient is not None
    return _mlflowClient


def model_uri(registered_name: str, champion=True):
    if champion:
        return f"models:/{registered_name}@champion"
    return f"models:/{registered_name}/latest"


def try_promote_model(model_name: str, metric: str):
    import mlflow.exceptions
    assert_mlflow_connection()
    assert _mlflowClient is not None
    latest_registered_version = _mlflowClient.get_registered_model(
        model_name).latest_versions[-1]  # type: ignore
    try:
        champion = _mlflowClient.get_model_version_by_alias(
            model_name, "champion")
        assert champion.run_id is not None
        champion_loss = _mlflowClient.get_run(
            champion.run_id).data.metrics[metric]
        latest_loss = _mlflowClient.get_run(
            latest_registered_version.run_id).data.metrics[metric]
        print(
            f"Champion '{metric}': {champion_loss:.4f},",
            f"Latest '{metric}': {latest_loss:.4f}"
        )
        if latest_loss < champion_loss:
            _mlflowClient.set_registered_model_alias(
                model_name, "champion", latest_registered_version.version)
            print(f"{model_name} has new champion")
        else:
            print(f"No new champion for{model_name}")
    except mlflow.exceptions.RestException:
        _mlflowClient.set_registered_model_alias(
            model_name, "champion", latest_registered_version.version)
        print(f"{model_name}'s first model")


def get_champion_run_id(registered_name: str):
    assert_mlflow_connection()
    assert _mlflowClient is not None
    return _mlflowClient.get_model_version_by_alias(name=registered_name, alias="champion").run_id


def assert_mlflow_connection():
    assert _mlflowClient is not None
    assert mlflow.is_tracking_uri_set()


def register_last_model(registered_name: str):
    info = mlflow.last_logged_model()
    if info is None:
        raise Exception("No last logged model")
    return mlflow.register_model(f"models:/{info.model_id}", name=registered_name)


def register_last_model_and_try_promote(registered_name: str, metric_name: str):
    assert_mlflow_connection()
    register_last_model(registered_name)
    try_promote_model(registered_name, metric_name)


def log_txt(val: str, name: str):
    assert_mlflow_connection()
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
    assert_mlflow_connection()
    import tempfile
    import mlflow
    with tempfile.TemporaryDirectory() as f:
        save_fn(f)
        print("Logging folder", f)
        mlflow.log_artifacts(f, artifact_path, run_id=run_id)


@functools.lru_cache(maxsize=16)
def download_artifacts(run_id: str, artifact_path: str):
    assert_mlflow_connection()
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
    model,  # : MovieRecommender,
    prepare: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_ds,
    test_ds,
    max_rating: int,
    run_id: str
):
    figures = {}
    from movie_recommender.workflow import save_figures
    from .utils import report
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


def download_file_from_s3(bucket, object_name, path):
    assert minioClient is not None
    minioClient.fget_object(bucket, object_name, path)
    return path


def download_parquet_from_s3(bucket: str, *filenames: str):
    import tempfile
    connect_minio()
    with tempfile.TemporaryDirectory() as dir:
        dfs = [pd.read_parquet(download_file_from_s3(
            bucket, f'{filename}.parquet', f"{dir}/{filename}.parquet")
        ) for filename in filenames]
    return dfs
