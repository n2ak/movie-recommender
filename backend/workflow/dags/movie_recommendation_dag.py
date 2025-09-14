from airflow import DAG
from airflow.decorators import task
from airflow.providers.docker.operators.docker import DockerOperator  # type: ignore
from docker.types import DeviceRequest
from pendulum import datetime
from utils import Env

default_args = {
    'owner': 'movie_recommender',
    'retries': 1,
}


def get_docker_operator(task_id, image, command, environment, gpu=True, **kwargs):
    return DockerOperator(
        container_name=task_id,
        task_id=task_id,
        image=image,
        auto_remove='force',
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
    schedule='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'etl', 'mlflow', 'pytorch', 'xgb'],
) as dag:

    @task
    def extract_data_task():
        try:
            from .extract import extract_data
        except ImportError:
            from extract import extract_data  # type: ignore
        return extract_data()

    @task
    def process_data_task(model_type):
        try:
            from .transform import process_data
        except ImportError:
            from transform import process_data  # type: ignore
        return process_data(model_type)

    train_dlrm_task = get_docker_operator(
        task_id='train_dlrm_task',
        image='pytorch_train',
        command=["python", "train_dlrm.py", "train"],
        environment=dict(
            EPOCHS=2,
            EXP_NAME=Env.exp_name,
            MLFLOW_TRACKING_URI=Env.MLFLOW_TRACKING_URI,
            DB_URL=Env.DB_URL,
        ),
        pool="apool", # must be set in Admin -> pools with 1 slot
    )
    train_xgb_task = get_docker_operator(
        task_id='train_xgb_task',
        image='xgb_train',
        command=["python", "train_xgb.py", "train"],
        environment=dict(
            NUM_BOOST_ROUND=1000,
            EXP_NAME=Env.exp_name,
            MLFLOW_TRACKING_URI=Env.MLFLOW_TRACKING_URI,
            DB_URL=Env.DB_URL,
        ),
        pool="apool",
    )
    test_dlrm_task = get_docker_operator(
        task_id='test_dlrm_task',
        image='pytorch_train',
        command=["python", "train_dlrm.py", "test"],
        environment=dict(
            EXP_NAME=Env.exp_name,
            MLFLOW_TRACKING_URI=Env.MLFLOW_TRACKING_URI,
        ),
        pool="apool",
    )
    test_xgb_task = get_docker_operator(
        task_id='test_xgb_task',
        image='xgb_train',
        command=["python", "train_xgb.py", "test"],
        environment=dict(
            EXP_NAME=Env.exp_name,
            MLFLOW_TRACKING_URI=Env.MLFLOW_TRACKING_URI,
        ),
        pool="apool",
    )
    extract = extract_data_task()

    (
        extract
        >> process_data_task("dlrm")
        >> train_dlrm_task
        >> test_dlrm_task
    )
    (
        extract
        >> process_data_task("xgb")
        >> train_xgb_task
        >> test_xgb_task
    )
