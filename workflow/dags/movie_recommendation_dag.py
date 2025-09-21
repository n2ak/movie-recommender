from airflow import DAG  # type: ignore
from pendulum import datetime
from docker.types import DeviceRequest
from airflow.sdk import Variable
from airflow.decorators import task  # type: ignore
from airflow.providers.docker.operators.docker import DockerOperator  # type: ignore

default_args = {
    'owner': 'movie_recommender',
    'retries': 1,
}


def get_docker_operator(task_id, image, command, environment, gpu=True, **kwargs):
    return DockerOperator(
        container_name=task_id,
        task_id=task_id,
        image=image,
        auto_remove='success',
        docker_url='unix://var/run/docker.sock',
        command=command,
        device_requests=[
            DeviceRequest(count=-1, capabilities=[['gpu']])
        ]if gpu else [],
        environment=environment,
        **kwargs,
    )


with DAG(
    dag_id='movie_recommender',
    default_args=default_args,
    # schedule='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'etl', 'mlflow', 'pytorch', 'xgb'],
) as dag:

    config: dict[str, dict[str, str]] = Variable.get(
        "config", deserialize_json=True)

    xgb: dict[str, str] = config["xgb"]
    dlrm: dict[str, str] = config["dlrm"]
    globalvar: dict[str, str] = config["global"]

    DLRM_EPOCHS = dlrm["epochs"]
    DLRM_MAX_RATING = dlrm["max_rating"]
    DLRM_TRAIN_SIZE = dlrm["train_size"]

    XGB_MAX_RATING = xgb["max_rating"]
    XGB_TRAIN_SIZE = xgb["train_size"]
    XGB_NUM_BOOST_ROUND = xgb["num_boosting_round"]

    DB_URL = globalvar["db_url"]
    EXP_NAME = globalvar["exp_name"]
    DB_MINIO_BUCKET = globalvar["db_minio_bucket"]
    MLFLOW_TRACKING_URI = globalvar["mlflow_tracking_uri"]
    MINIO_USER = globalvar["minio_user"]
    MINIO_PASSWORD = globalvar["minio_password"]
    MINIO_S3_ENDPOINT = globalvar["minio_endpoint"]

    @task
    def extract_data_task(db_url: str):
        try:
            from .extract import extract_data
        except ImportError:
            from extract import extract_data  # type: ignore
        return extract_data(db_url)

    @task
    def process_data_task_for_dlrm(dir):
        try:
            from .transform import process_data
        except ImportError:
            from transform import process_data  # type: ignore
        return process_data(dir, "dlrm", int(DLRM_MAX_RATING), float(DLRM_TRAIN_SIZE))

    @task
    def process_data_task_for_simsearch(dir):
        try:
            from .transform import process_data
        except ImportError:
            from transform import process_data  # type: ignore
        return process_data(dir, "simsearch", 5, 1)

    @task
    def upload_result_to_s3(s3_bucket: str, directory: str):
        try:
            from .utils import upload_folder_to_s3 as upload
        except ImportError:
            from utils import upload_folder_to_s3 as upload  # type: ignore
        upload(s3_bucket, directory, creds=dict(
            minio_user=MINIO_USER,
            minio_passwd=MINIO_PASSWORD,
            endpoint=MINIO_S3_ENDPOINT,
        ))

    train_dlrm_task = get_docker_operator(
        task_id='train_dlrm_task',
        image='pytorch_train',
        command=["python", "train_dlrm.py", "train"],
        environment=dict(
            EPOCHS=DLRM_EPOCHS,
            EXP_NAME=EXP_NAME,
            MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI,
            DB_URL=DB_URL,
            DLRM_REGISTERED_NAME="dlrm",
            DB_MINIO_BUCKET=DB_MINIO_BUCKET,
            MLFLOW_S3_ENDPOINT_URL=MINIO_S3_ENDPOINT,
            AWS_ACCESS_KEY_ID=MINIO_USER,
            AWS_SECRET_ACCESS_KEY=MINIO_PASSWORD,
            AWS_DEFAULT_REGION="us-east-1",
        ),
        pool="one_task_pool",  # must be set in Admin -> pools with 1 slot
    )
    train_xgb_task = get_docker_operator(
        task_id='train_xgb_task',
        image='xgb_train',
        command=["python", "train_xgb.py", "train"],
        environment=dict(
            NUM_BOOST_ROUND=XGB_NUM_BOOST_ROUND,
            EXP_NAME=EXP_NAME,
            MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI,
            DB_URL=DB_URL,
            XGB_REGISTERED_NAME="xgb",
            DB_MINIO_BUCKET=DB_MINIO_BUCKET,
            MLFLOW_S3_ENDPOINT_URL=MINIO_S3_ENDPOINT,
            AWS_ACCESS_KEY_ID=MINIO_USER,
            AWS_SECRET_ACCESS_KEY=MINIO_PASSWORD,
            AWS_DEFAULT_REGION="us-east-1",
        ),
        pool="one_task_pool",
    )
    test_dlrm_task = get_docker_operator(
        task_id='test_dlrm_task',
        image='pytorch_train',
        command=["python", "train_dlrm.py", "test"],
        environment=dict(
            EXP_NAME=EXP_NAME,
            MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI,
            DLRM_REGISTERED_NAME="dlrm",
            DB_MINIO_BUCKET=DB_MINIO_BUCKET,
            MLFLOW_S3_ENDPOINT_URL=MINIO_S3_ENDPOINT,
            AWS_ACCESS_KEY_ID=MINIO_USER,
            AWS_SECRET_ACCESS_KEY=MINIO_PASSWORD,
            AWS_DEFAULT_REGION="us-east-1",
        ),
        pool="one_task_pool",
    )
    test_xgb_task = get_docker_operator(
        task_id='test_xgb_task',
        image='xgb_train',
        command=["python", "train_xgb.py", "test"],
        environment=dict(
            EXP_NAME=EXP_NAME,
            MLFLOW_TRACKING_URI=MLFLOW_TRACKING_URI,
            XGB_REGISTERED_NAME="xgb",
            DB_MINIO_BUCKET=DB_MINIO_BUCKET,
            MLFLOW_S3_ENDPOINT_URL=MINIO_S3_ENDPOINT,
            AWS_ACCESS_KEY_ID=MINIO_USER,
            AWS_SECRET_ACCESS_KEY=MINIO_PASSWORD,
            AWS_DEFAULT_REGION="us-east-1",
        ),
        pool="one_task_pool",
    )

    data_dir = extract_data_task(DB_URL)
    dlrm_dir = process_data_task_for_dlrm(data_dir)
    simsearch_dir = process_data_task_for_simsearch(data_dir)

    (
        upload_result_to_s3(DB_MINIO_BUCKET, dlrm_dir)
        >> train_dlrm_task
        >> test_dlrm_task
    )  # type: ignore
    (
        upload_result_to_s3(DB_MINIO_BUCKET, data_dir)
        >> train_xgb_task
        >> test_xgb_task
    )  # type: ignore
    (
        upload_result_to_s3(DB_MINIO_BUCKET, simsearch_dir)
        # >> train_simsearch
    )  # type: ignore
