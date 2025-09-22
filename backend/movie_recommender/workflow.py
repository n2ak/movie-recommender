
import os
import functools
from typing import Optional
from datetime import datetime

import mlflow
import pandas as pd
import mlflow.artifacts
from minio import Minio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
from .logging import Logger

_mlflowClient: Optional[mlflow.MlflowClient] = None
minioClient: Optional[Minio] = None


def connect_minio():
    global minioClient
    import os
    if minioClient is None:
        endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
        secure = endpoint.startswith("https://")
        endpoint_stripped = endpoint.replace(
            "http://", "").replace("https://", "")
        Logger.debug("Secure %s", secure)
        Logger.debug("endpoint %s", endpoint)
        minioClient = Minio(
            endpoint=endpoint_stripped,
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region=os.environ["AWS_DEFAULT_REGION"],
            secure=secure,
        )
        Logger.info("Connected to minio on: %s", endpoint)
    Logger.debug("Available Buckets %s", minioClient.list_buckets())


def connect_mlflow():
    global _mlflowClient
    if _mlflowClient is not None:
        return
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    _mlflowClient = mlflow.MlflowClient(uri)
    # as a health check
    mlflow.search_experiments()
    Logger.info("Connected to mlflow on: %s", uri)


def get_mlflow_client():
    assert _mlflowClient is not None
    return _mlflowClient


def model_uri(registered_name: str, champion=True):
    if champion:
        return f"models:/{registered_name}@champion"
    return f"models:/{registered_name}/latest"


def get_run_metrics(run_id: str):
    assert _mlflowClient is not None
    return _mlflowClient.get_run(run_id).data.metrics


def try_promote_model(model_name: str, metric: str, minimum=True):
    import mlflow.exceptions
    assert_mlflow_connection()
    assert _mlflowClient is not None

    latest_registered = _mlflowClient.get_registered_model(
        model_name).latest_versions[-1]  # type: ignore

    last_version = latest_registered.version

    try:
        champion = _mlflowClient.get_model_version_by_alias(
            model_name, "champion")
        assert champion.run_id is not None

        champion_loss = get_run_metrics(champion.run_id)[metric]
        latest_loss = get_run_metrics(latest_registered.run_id)[metric]

        Logger.debug(
            f"Champion '{metric}': {champion_loss:.4f}," +
            f" Latest '{metric}': {latest_loss:.4f}"
        )

        condition = latest_loss < champion_loss if minimum else latest_loss > champion_loss

        if condition:
            promote_model_to_champion(model_name, last_version)
            Logger.debug(f"'{model_name}' has new champion")
        else:
            Logger.debug(f"No new champion for '{model_name}'")

    except mlflow.exceptions.RestException:
        promote_model_to_champion(model_name, last_version)
        Logger.debug(f"{model_name}'s first model")


def promote_model_to_champion(registered_name, version: str):
    assert _mlflowClient is not None
    _mlflowClient.set_registered_model_alias(
        registered_name, "champion", version)
    Logger.debug(f"Promoted {version=} of {registered_name=} to champion!")


def get_registered_model_run_id(registered_name: str, champion: bool):
    assert_mlflow_connection()
    assert _mlflowClient is not None
    alias = "champion" if champion else "latest"
    return _mlflowClient.get_model_version_by_alias(name=registered_name, alias=alias).run_id


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
        Logger.debug("Logging folder %s", f)
        mlflow.log_artifacts(f, artifact_path, run_id=run_id)


@functools.lru_cache(maxsize=16)
def download_artifacts(run_id: str, artifact_path: str):
    assert_mlflow_connection()
    Logger.debug(f"Getting artifacts, {run_id=}, {artifact_path=}")
    files = mlflow.artifacts.list_artifacts(
        run_id=run_id)
    if not len(files):
        raise Exception(f"No files! {run_id=}, {artifact_path=}")

    return mlflow.artifacts.download_artifacts(  # type: ignore
        run_id=run_id,
        artifact_path=artifact_path,
    )


def save_plots(
    model,
    train_data:  tuple[NDArray, NDArray, NDArray],
    test_data:  tuple[NDArray, NDArray, NDArray],
    max_rating: int,
    run_id: str
):
    figures = {}
    from movie_recommender.workflow import save_figures
    from .utils import report
    user_ids, movie_ids, train_y = train_data
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

    Logger.info("Test")
    user_ids, movie_ids, test_y = test_data
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


def read_parquet_from_s3(bucket: str, filepath: str):
    assert minioClient and minioClient._provider, minioClient
    key = minioClient._provider.retrieve()._access_key
    secret = minioClient._provider.retrieve()._secret_key
    endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]

    return pd.read_parquet(
        f"s3://{bucket}/{filepath}",
        storage_options=dict(
            key=key,
            secret=secret,
            client_kwargs=dict(
                endpoint_url=endpoint
            )
        )
    )


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


def upload_folder_to_s3(dir: str, bucket):
    import pathlib
    assert minioClient is not None
    Logger.debug(f"Uploading {dir=} to s3 {bucket=}")
    for file in pathlib.Path(dir).glob("*"):
        filepath = str(file)
        object_name = filepath.split("/")[-1]

        Logger.debug(f"Uploading {object_name=} to s3 {bucket=} {filepath=}")

        minioClient.fput_object(
            bucket, object_name=object_name, file_path=filepath
        )


def upload_file_to_s3(bucket: str, filepath: str):
    assert minioClient is not None
    object_name = filepath.split("/")[-1]

    Logger.debug(f"Uploading {object_name=} to s3 {bucket=} {filepath=}")
    minioClient.fput_object(
        bucket, object_name=object_name, file_path=filepath
    )


def upload_parquet_to_s3(bucket, **dfs: pd.DataFrame):
    import tempfile
    with tempfile.TemporaryDirectory() as path:
        for name, df in dfs.items():
            df.to_parquet(f"{path}/{name}.parquet")
        upload_folder_to_s3(path, bucket)


def make_run_name(prefix: str):
    now = datetime.now().strftime('%Y_%m_%d_%Hh_%Mm_%Ss')
    return f"{prefix.lower()}_{now}"
