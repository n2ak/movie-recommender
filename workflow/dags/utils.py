import pathlib
from minio import Minio
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
        return uri

    def dump(self):
        print("*********************ENV***********************")
        print("exp_name:", self.exp_name)
        print("MAX_RATING:", self.MAX_RATING)
        print("NUM_BOOST_ROUND:", self.NUM_BOOST_ROUND)
        print("VERBOSE_EVAL:", self.VERBOSE_EVAL)
        print("OPTIMIZE:", self.OPTIMIZE)
        print("TRAIN_SIZE:", self.TRAIN_SIZE)
        print("EPOCH:", self.EPOCH)
        print("DB_URL:", self.DB_URL)
        print("MLFLOW_TRACKING_URI:", self.MLFLOW_TRACKING_URI)
        print("***********************************************")


Env = Env_()


def read_parquet(*paths: str):
    import pandas as pd
    return [pd.read_parquet(p) for p in paths]


client: typing.Optional[Minio] = None


def connect_minio():
    global client
    import os
    if client is None:
        endpoint = os.environ["MLFLOW_S3_ENDPOINT_URL"]
        secure = endpoint.startswith("https://")
        endpoint = endpoint.replace("http://", "").replace("https://", "")
        print("Secure", secure)
        client = Minio(
            endpoint=endpoint,
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region=os.environ["AWS_DEFAULT_REGION"],
            secure=secure,
        )
        print("Buckets", client.list_buckets())


def upload_files(save_fn: typing.Callable[[pathlib.Path], typing.Any], bucket):
    import tempfile
    assert client is not None
    with tempfile.TemporaryDirectory() as dir:
        save_fn(pathlib.Path(dir))
        print(f"Uploading {dir=} to s3 {bucket=}")
        for file in pathlib.Path(dir).glob("*"):
            filepath = str(file)
            object_name = filepath.split("/")[-1]
            print(f"Uploading {object_name=} to s3 {bucket=} {filepath=}")
            client.fput_object(
                bucket, object_name=object_name, file_path=filepath)


def download_file(bucket, object_name, path):
    assert client is not None
    client.fget_object(bucket, object_name, path)
    return path


def download_parquet(bucket: str, *filenames: str):
    import tempfile
    import pandas as pd
    connect_minio()
    with tempfile.TemporaryDirectory() as dir:
        dfs = [pd.read_parquet(download_file(
            bucket, f'{filename}.parquet', f"{dir}/{filename}.parquet")
        ) for filename in filenames]
    return dfs
