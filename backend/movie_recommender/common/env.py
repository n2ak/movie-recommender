import os
import typing


DATABASE_URL = os.environ["DATABASE_URL"]
BUCKET = os.environ["BUCKET"]
if BUCKET.endswith("/"):
    BUCKET = BUCKET.removesuffix("/")

ARTIFACT_ROOT = f"{BUCKET}/artifacts"

SS_REGISTERED_NAME = os.environ["SS_REGISTERED_NAME"]
XGB_REGISTERED_NAME = os.environ["XGB_REGISTERED_NAME"]
DLRM_REGISTERED_NAME = os.environ["DLRM_REGISTERED_NAME"]

MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")

STORAGE_PROVIDER: typing.Literal["aws", "gcp", "minio"] = os.environ[
    "STORAGE_PROVIDER"]  # type: ignore

match STORAGE_PROVIDER:
    case "gcp":
        config = {}
    case "aws":
        config = {
            "access_key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "region": os.environ["AWS_REGION"],
        }
    case "minio":
        config = {
            "endpoint": os.environ["S3_ENDPOINT"],
            "access_key": os.environ["AWS_ACCESS_KEY_ID"],
            "secret_key": os.environ["AWS_SECRET_ACCESS_KEY"],
            "region": os.environ["AWS_REGION"],
        }
    case _:
        raise Exception(f"Invalid provider: {STORAGE_PROVIDER}")
