import mlflow
import pandas as pd
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from movie_recommender.data import MovieLens
from movie_recommender.utils import report
from movie_recommender.modeling.xgbmr import XGBMR
from movie_recommender.train_utils import mae as _mae, get_env
from movie_recommender.workflow import download_parquet
import logging

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def training(
    experiment_name: str,
    train_ds: pd.DataFrame, test_ds: pd.DataFrame,
    users, movies,
    booster,
    max_depth,
    lr,
    max_rating,
    num_boost_round=10_000,
    verbose_eval=1000,
    early_stopping_rounds=500,
):
    assert "user_id" not in train_ds.columns
    assert "movie_id" not in train_ds.columns

    model = XGBMR(
        eval_metric=["rmse"],
        booster=booster,
        max_depth=max_depth,
        learning_rate=lr,
    )
    model.set_data(users, movies)
    # train_X, train_y = train_ds[model.cols], train_ds.rating
    # test_X, test_y = test_ds[model.cols], test_ds.rating
    result, run_id = model.fit(
        train_ds[model.cols],
        train_ds.rating,
        eval_set=(test_ds[model.cols], test_ds.rating),
        experiment_name=experiment_name,
        num_boost_round=num_boost_round,
        verbose_eval=verbose_eval,
        early_stopping_rounds=early_stopping_rounds,
        custom_metric=mae,
        maximize=False,
    )
    from movie_recommender.workflow import save_plots
    # save_plots(model, prepare, train_ds, test_ds, max_rating, run_id=run_id)
    return result


def train_xgb(train: pd.DataFrame, test: pd.DataFrame, max_rating: int, exp_name: str, optimize: bool,
              num_boost_round: int, tracking_uri: str):
    train.set_index(["user_id", "movie_id"], inplace=True)
    test.set_index(["user_id", "movie_id"], inplace=True)

    users, movies = split_(pd.concat([train, test], axis=0))
    logger.info("****************Starting xgb training...**************")
    logger.info("Train ds shape: %s", train.shape)
    logger.info("Test ds shape: %s", test.shape)
    logger.info("num_boost_round: %s", num_boost_round)
    logger.info("optimize: %s", optimize)
    logger.info(f"Mlflow {exp_name=}, {tracking_uri=}")

    mlflow.set_tracking_uri(tracking_uri)

    if optimize:
        optimized(exp_name, train, test, users,
                  movies, max_rating, num_boost_round)
    else:
        training(
            exp_name,
            train, test,
            users, movies,
            booster="gbtree",
            lr=0.02,
            max_rating=max_rating,
            max_depth=6,
            num_boost_round=num_boost_round
        )
    logger.info("****************Training is done*******************")


def optimized(exp_name: str, train, test, users, movies, max_rating, num_boost_round):
    import optuna

    def objective_func(trial: optuna.Trial):
        booster = trial.suggest_categorical("booster", ["gbtree"])
        learning_rate = trial.suggest_float(
            "learning_rate", low=0.001, high=0.3)
        max_depth = trial.suggest_int("max_depth", low=3, high=6)
        eval_results = training(
            exp_name,
            train, test,
            users, movies,
            max_rating=max_rating,
            booster=booster, max_depth=max_depth, lr=learning_rate,
            num_boost_round=num_boost_round,
            verbose_eval=1000,
        )
        return eval_results["val"]["mae"][-1]
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_func, n_trials=20, show_progress_bar=True)


def split_(df: pd.DataFrame):
    movie_cols = MovieLens.movie_cols(df)
    user_cols = MovieLens.user_cols(df)
    movies = df.droplevel("user_id")[movie_cols]
    movies = movies.loc[~movies.index.duplicated(keep='first'), :]

    users = df.droplevel("movie_id")[user_cols]
    users = users.loc[~users.index.duplicated(keep='first'), :]
    return users, movies


def prepare(X: pd.DataFrame):
    X = X.reset_index()
    user_ids, movie_ids, y = X.user_id, X.movie_id, X.rating
    return np.array(user_ids), np.array(movie_ids), np.array(y)


def mae(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    y = dtrain.get_label()
    return _mae(predt, y)


def test_xgb_model():
    from movie_recommender.recommender import Recommender, Request
    Recommender.instance = None
    Recommender.single(Request(
        userId=0,
        genres=[],
        model="xgb_cpu",
        temp=0,
        start=0,
        count=10,
    ))
    logger.info("XGB model test passed successfully")


if __name__ == "__main__":
    import sys
    import os
    arg = sys.argv[1]
    uri = get_env("MLFLOW_TRACKING_URI", "http://localhost:8081")
    mlflow.set_tracking_uri(uri)
    bucket = os.environ["DB_MINIO_BUCKET"]

    if arg == "train":
        db_url = get_env(
            "DB_URL", 'postgresql+psycopg2://admin:password@localhost:5432/mydb')
        logger.info("DB URL: %s", db_url)
        train, test = download_parquet(bucket, "xgb_train", "xgb_test")

        train_xgb(
            train,
            test,
            max_rating=5,
            exp_name=get_env("EXP_NAME", "movie_recom"),
            optimize=False,
            num_boost_round=get_env("NUM_BOOST_ROUND", 500),
            tracking_uri=uri,
        )
    elif arg == "test":
        test_xgb_model()
    else:
        logger.error(f'Invalid arg {arg}')
        sys.exit(1)
