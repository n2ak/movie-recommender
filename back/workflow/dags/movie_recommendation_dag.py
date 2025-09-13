from airflow import DAG
from airflow.decorators import task
from airflow.providers.docker.operators.docker import DockerOperator  # type: ignore
from docker.types import DeviceRequest
from pendulum import datetime

default_args = {
    'owner': 'movie_recommender',
    'retries': 1,
}


with DAG(
    dag_id='movie_recommender',
    default_args=default_args,
    schedule='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'etl', 'mlflow', 'registry'],
) as dag:

    @task(multiple_outputs=True)
    def extract_data_task():
        try:
            from .extract import extract_data
        except ImportError:
            from extract import extract_data  # type: ignore
        return extract_data()

    @task(multiple_outputs=True)
    def process_data_task(p1, p2, model_type):
        try:
            from .transform import process_data
        except ImportError:
            from transform import process_data  # type: ignore
        return process_data(p1, p2, model_type)

    train_dlrm_task = DockerOperator(
        dag=dag,
        task_id='train_dlrm_task',
        image='pytorch_train',
        auto_remove='force',
        docker_url='unix://var/run/docker.sock',
        command=["python", "train_dlrm.py"],  # TODO,
        device_requests=[
            # Request all available GPUs
            DeviceRequest(count=-1, capabilities=[['gpu']])
        ],
    )

    paths: dict = extract_data_task()
    ratings_path, movies_path = paths["ratings_path"], paths["movies_path"]

    # xgb_paths: dict = process_data_task(
    # ratings_path, movies_path, "xgb")  # type: ignore
    process_data_task(
        ratings_path, movies_path, "dlrm") >> train_dlrm_task  # type: ignore
