
import functools
from typing import Optional, Literal
from datetime import datetime

import mlflow
import pandas as pd
import mlflow.artifacts
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray
from .logging import Logger
from .env import MLFLOW_TRACKING_URI, STORAGE_PROVIDER, config
_mlflow_client: Optional[mlflow.MlflowClient] = None
_storage_client: Optional["StorageClient"] = None


class StorageClient:
    def __init__(
        self,
        provider: Literal["gcp", "aws", "minio"],
        config: dict,
    ):
        self.provider = provider

        if provider == "aws":
            import boto3
            self.client = boto3.client(
                "s3",
                aws_access_key_id=config["access_key"],
                aws_secret_access_key=config["secret_key"],
                region_name=config.get("region", "us-east-1"),
            )

        elif provider == "gcp":
            from google.cloud import storage
            self.client = storage.Client()

        elif provider == "minio":
            import boto3
            self.client = boto3.client(
                "s3",
                # e.g. http://localhost:9000
                endpoint_url=config["endpoint"],
                aws_access_key_id=config["access_key"],
                aws_secret_access_key=config["secret_key"],
                region_name=config.get("region", "us-east-1"),
            )

        else:
            raise ValueError(
                "Unknown provider: must be 'aws', 'gcp', or 'minio'")

    def upload_file(self, bucket_name, file_path, dest_path):
        if self.provider in ("aws", "minio"):
            self.client.upload_file(  # type: ignore
                file_path, bucket_name, dest_path)
        elif self.provider == "gcp":
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(dest_path)
            blob.upload_from_filename(file_path)

    def download_file(self, bucket_name, src_path, dest_path):
        if self.provider in ("aws", "minio"):
            self.client.download_file(  # type: ignore
                bucket_name, src_path, dest_path)
        elif self.provider == "gcp":
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(src_path)
            blob.download_to_filename(dest_path)

    def list_files(self, bucket_name, prefix=""):
        if self.provider in ("aws", "minio"):
            resp = self.client.list_objects_v2(  # type: ignore
                Bucket=bucket_name, Prefix=prefix)
            return [item["Key"] for item in resp.get("Contents", [])]
        elif self.provider == "gcp":
            bucket = self.client.bucket(bucket_name)
            return [blob.name for blob in bucket.list_blobs(prefix=prefix)]

    def delete_file(self, bucket_name, path):
        if self.provider in ("aws", "minio"):
            self.client.delete_object(  # type: ignore
                Bucket=bucket_name, Key=path)
        elif self.provider == "gcp":
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(path)
            blob.delete()

    def get_uri(self, bucket_name, object_path):
        """
          Return the canonical/public URI for a given object path.
        """
        if self.provider in ["aws", "minio"]:
            # S3 public URL
            return f"s3://{bucket_name}/{object_path}"
        elif self.provider == "gcp":
            # GCS public URL
            return f"gs://{bucket_name}/{object_path}"
        else:
            raise ValueError("Unknown provider")

    def list_buckets(self):
        if self.provider in ("aws", "minio"):
            resp = self.client.list_buckets()
            return [bucket["Name"] for bucket in
                    resp.get("Buckets", [])]  # type: ignore
        elif self.provider == "gcp":
            return [bucket.name for bucket in self.client.list_buckets()]
        else:
            raise ValueError("Unknown provider")


def connect_storage_client():
    global _storage_client
    if _storage_client is None:
        _storage_client = StorageClient(
            provider=STORAGE_PROVIDER,
            config=config,
        )
    Logger.debug("Available Buckets %s", _storage_client.list_buckets())


def connect_mlflow():
    global _mlflow_client
    if _mlflow_client is not None:
        return
    uri = MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    _mlflow_client = mlflow.MlflowClient(uri)
    # as a health check
    mlflow.search_experiments()
    Logger.info("Connected to mlflow on: %s", uri)


def get_mlflow_client():
    assert _mlflow_client is not None
    return _mlflow_client


def model_uri(registered_name: str, champion=True):
    if champion:
        return f"models:/{registered_name}@champion"
    return f"models:/{registered_name}/latest"


def get_run_metrics(run_id: str):
    assert _mlflow_client is not None
    return _mlflow_client.get_run(run_id).data.metrics


def try_promote_model(model_name: str, metric: str, minimum=True):
    import mlflow.exceptions
    assert_mlflow_connection()
    assert _mlflow_client is not None

    latest_registered = _mlflow_client.get_registered_model(
        model_name).latest_versions[-1]  # type: ignore

    last_version = latest_registered.version

    try:
        champion = _mlflow_client.get_model_version_by_alias(
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
    assert _mlflow_client is not None
    _mlflow_client.set_registered_model_alias(
        registered_name, "champion", version)
    Logger.debug(f"Promoted {version=} of {registered_name=} to champion!")


def get_registered_model_run_id(registered_name: str, champion: bool):
    assert_mlflow_connection()
    assert _mlflow_client is not None
    alias = "champion" if champion else "latest"
    return _mlflow_client.get_model_version_by_alias(name=registered_name, alias=alias).run_id


def assert_mlflow_connection():
    assert _mlflow_client is not None
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
    from backend.movie_recommender.common.workflow import save_figures
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
    df, = download_parquet_from_s3(bucket, filepath)
    return df


def download_file_from_s3(bucket, object_name, path):
    assert _storage_client is not None
    _storage_client.download_file(bucket, object_name, path)
    return path


def download_parquet_from_s3(bucket: str, *filenames: str):
    import tempfile
    connect_storage_client()
    with tempfile.TemporaryDirectory() as dir:
        dfs = [pd.read_parquet(download_file_from_s3(
            bucket, f'{filename}.parquet', f"{dir}/{filename}.parquet")
        ) for filename in filenames]
    return dfs


def upload_folder_to_s3(dir: str, bucket):
    import pathlib
    assert _storage_client is not None
    Logger.debug(f"Uploading {dir=} to s3 {bucket=}")
    for file in pathlib.Path(dir).glob("*"):
        filepath = str(file)
        object_name = filepath.split("/")[-1]

        Logger.debug(f"Uploading {object_name=} to s3 {bucket=} {filepath=}")

        _storage_client.upload_file(
            bucket, filepath, object_name
        )


def upload_file_to_s3(bucket: str, filepath: str):
    assert _storage_client is not None
    object_name = filepath.split("/")[-1]

    Logger.debug(f"Uploading {object_name=} to s3 {bucket=} {filepath=}")
    _storage_client.upload_file(
        bucket, filepath, object_name
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
