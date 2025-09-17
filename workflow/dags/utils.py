from minio import Minio
import typing


client: typing.Optional[Minio] = None


def connect_minio(minio_creds: dict[str, str]):
    global client
    if client is None:
        endpoint = minio_creds["endpoint"].replace(
            "http://", "").replace("https://", "")
        client = Minio(
            endpoint=endpoint,
            access_key=minio_creds["minio_user"],
            secret_key=minio_creds["minio_passwd"],
            region=minio_creds.get("region", "us-east-1"),
            secure=False,
        )
    return client


def read_parquet(*paths: str):
    import pandas as pd
    return [pd.read_parquet(p) for p in paths]


def upload_folder_to_s3(bucket: str, dir: str, creds):
    import pathlib
    client = connect_minio(creds)
    print(f"Uploading {dir=} to s3 {bucket=}")
    for file in pathlib.Path(dir).glob("*"):
        filepath = str(file)
        object_name = filepath.split("/")[-1]
        print(f"Uploading {object_name=} to s3 {bucket=} {filepath=}")
        client.fput_object(
            bucket, object_name=object_name, file_path=filepath
        )
