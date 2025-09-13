from airflow import DAG
from airflow.decorators import task
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
    tags=['ml', 'etl', 'mlflow', 'registry']
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

    @task
    def train_xgb_task(p1, p2):
        try:
            from .train_xgb import train_xgb
        except ImportError:
            from train_xgb import train_xgb  # type: ignore
        return train_xgb(p1, p2)

    @task
    def train_dlrm_task(p1, p2):
        try:
            from .train_dlrm import train_dlrm
        except ImportError:
            from train_dlrm import train_dlrm  # type: ignore
        return train_dlrm(p1, p2)

    @task
    def train_similarity_search_task(p1, p2):
        try:
            from .test import train_simsearch
        except ImportError:
            from test import train_simsearch  # type: ignore
        return train_simsearch(p1, p2)

    @task
    def test_xgb_task():
        try:
            from .test import test_xgb_model
        except ImportError:
            from test import test_xgb_model  # type: ignore
        return test_xgb_model()

    @task
    def test_dlrm_task():
        try:
            from .test import test_dlrm_model
        except ImportError:
            from test import test_dlrm_model  # type: ignore
        return test_dlrm_model()

    @task
    def test_simsearch_task(data):
        try:
            from .test import test_simsearch
        except ImportError:
            from test import test_simsearch  # type: ignore
        return test_simsearch(data)

    paths: dict = extract_data_task()  # type: ignore
    ratings_path, movies_path = paths["ratings_path"], paths["movies_path"]

    xgb_paths: dict = process_data_task(
        ratings_path, movies_path, "xgb")  # type: ignore
    dlrm_paths: dict = process_data_task(
        ratings_path, movies_path, "dlrm")  # type: ignore

    xgb_train_path, xgb_movies_path = xgb_paths["train_path"], xgb_paths["test_path"]
    dlrm_train_path, dlrm_movies_path = dlrm_paths["train_path"], dlrm_paths["test_path"]

    run_ddata = train_similarity_search_task(ratings_path, movies_path)

    (
        train_xgb_task(xgb_train_path, xgb_movies_path) >>
        test_xgb_task()
    )  # type: ignore
    (
        train_dlrm_task(dlrm_train_path, dlrm_movies_path) >>
        test_dlrm_task()
    )  # type: ignore
    test_simsearch_task(run_ddata)
