import optuna
import pandas as pd
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin

from movie_recommender.logging import Logger
from movie_recommender.modeling.xgbmr import XGBMR
from movie_recommender.data import movie_cols, user_cols
from movie_recommender.train_utils import fix_split, mae, simple_split
from movie_recommender.workflow import read_parquet_from_s3, save_plots, connect_minio, connect_mlflow


def split_genres(df):
    df.movie_genres = df.movie_genres.apply(lambda x: x.split(","))
    return df


def get_genre_stats(df: pd.DataFrame):
    from collections import defaultdict
    genres_counts = defaultdict(int)
    genres_avg_ratings = defaultdict(float)
    for _, row in df.iterrows():
        for g in row.movie_genres:
            genres_counts[g] += 1
            genres_avg_ratings[g] += row.rating
    for k in genres_avg_ratings:
        genres_avg_ratings[k] = genres_avg_ratings[k] / genres_counts[k]
    return genres_counts, genres_avg_ratings


def get_movies_data(train: pd.DataFrame, year_bins):
    train.movie_year = pd.cut(
        train["movie_year"], bins=year_bins, labels=False, include_lowest=True)
    df = train[["movie_id", "movie_year"]].merge(
        train.groupby("movie_id").agg(
            movie_avg_rating=("rating", "mean"),
            movie_total_rating=("rating", "count"),
        ),
        on="movie_id",
        how="right"
    ).drop_duplicates("movie_id")
    df = pd.get_dummies(df, columns=[
        "movie_year",
    ])
    return df.set_index("movie_id")


def merge(df, movies_data, user_data):
    df = df.merge(
        movies_data,
        on="movie_id",
        how="left"
    ).merge(
        user_data,
        on="user_id",
        how="left"
    )
    return df


def get_user_data(train):
    # genres_counts, genres_avg_ratings = get_genre_stats(train)
    genres = get_possible_genres(train)

    genres_dummies = pd.DataFrame(
        train.movie_genres.apply(lambda movie_genres: [
                                 item in movie_genres for item in genres]).tolist(),
        columns=genres
    )
    genres_dummies["user_id"] = train.user_id.tolist()
    user_genres_mean = genres_dummies.copy()
    user_genres_mean[genres] *= train.rating.values[:, None]
    user_genres_mean = user_genres_mean.groupby("user_id").agg("mean")
    user_genres_counts = genres_dummies.groupby("user_id").agg("sum")

    user_data = pd.concat([
        user_genres_mean.rename(columns={g: f"mean_{g}" for g in genres}),
        user_genres_mean.idxmax(axis=1).rename("highly rated genre"),
        user_genres_counts.rename(columns={g: f"total_{g}" for g in genres}),
        user_genres_counts.idxmax(axis=1).rename("most rated genre")
    ],
        axis=1,
    ).merge(
        train.groupby("user_id").agg(
            user_avg_rating=("rating", "mean"),
            user_total_rating=("rating", "count"),
        ),
        on="user_id",
        how="left"
    )
    user_data = pd.get_dummies(user_data, columns=[
        "highly rated genre",
        "most rated genre",
    ])
    user_data = user_data.loc[~user_data.index.duplicated(keep='first'), :]
    return user_genres_counts, user_genres_mean, user_data


def get_possible_genres(df):
    from itertools import chain
    return list(set(chain.from_iterable(df.movie_genres)))


class MovieLensPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, n_year_bins=3):
        self.n_year_bins = n_year_bins

    def fit(self, train: pd.DataFrame, y=None, scale=True):
        year_bins = np.histogram_bin_edges(
            train["movie_year"], bins=self.n_year_bins)
        train = train.copy()
        train = split_genres(train)
        movies = get_movies_data(train.copy(), year_bins)
        _, _, users = get_user_data(train)

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        movies[movies.columns] = StandardScaler().fit_transform(movies)
        users[users.columns] = StandardScaler().fit_transform(users)

        self.movie_data = movies
        self.user_data = users

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X[['rating', 'movie_id', 'user_id']]
        X = X.merge(
            self.movie_data,
            on="movie_id",
            how="left"
        ).merge(
            self.user_data,
            on="user_id",
            how="left"
        )
        X = X.set_index(["user_id", "movie_id"])
        y = X.pop("rating") / 5
        return X, y

    def align(self, train, test):
        train, test = train.align(test, join="outer", axis=1, fill_value=0)
        return train, test

    def scale(self, train, test):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        scaler = StandardScaler().fit(train)
        cols = train.columns
        train[cols] = scaler.transform(train[cols])
        test[cols] = scaler.transform(test[cols])
        return train, test

    def do_transform(self, train, test):
        X_train, y_train = self.transform(train)
        X_test, y_test = self.transform(test)
        X_train, X_test = self.align(X_train, X_test)
        # X_train, X_test = self.scale(X_train, X_test)

        assert X_train.columns.tolist() == X_test.columns.tolist()

        assert X_train.isna().mean().mean() == 0
        assert X_test.isna().mean().mean() == 0

        return X_train, X_test, y_train, y_test


def cv(df: pd.DataFrame, params, n_splits=3, seed=0, **training_params):
    from sklearn.model_selection import KFold
    scores = []
    for train_idx, val_idx in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df):
        X_train = df.iloc[train_idx]
        X_val = df.iloc[val_idx]
        X_train, X_val = fix_split(df, X_train, X_val)

        # Logger.info("",X_train.shape,X_val.shape,
        #     "Val new user ids :",np.setdiff1d(
        #         X_val.user_id.unique(),
        #         X_train.user_id.unique(),
        #     ).shape,
        #     "Val new movie ids:", np.setdiff1d(
        #         X_val.movie_id.unique(),
        #         X_train.movie_id.unique(),
        #     ).shape,
        # )

        processor = MovieLensPreprocessor().fit(X_train)
        X_train, X_val, y_train, y_val = processor.do_transform(X_train, X_val)

        booster = train_xgb(
            params,
            X_train, y_train, X_val, y_val,
            **training_params,
            maximize=False,
        )
        pred = booster.predict(xgb.DMatrix(X_val, y_val))
        Logger.debug(f"Pred mean: {pred.mean():.2f} , std: {pred.std():.2f}")
        scores.append(mae(pred, y_val))

    scores = np.array(scores)
    return scores


def train_xgb(
    params,
    X_train, y_train, X_test, y_test,
    **kwargs,
):
    assert np.all((0 <= y_train) & (y_train <= 1))
    assert np.all((0 <= y_test) & (y_test <= 1))
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_test, y_test)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dtrain, "train"), (dval, "val")],
        **kwargs,
    )
    return booster


def split_(df: pd.DataFrame):
    movies = df.droplevel("user_id")[movie_cols(df)]
    movies = movies.loc[~movies.index.duplicated(keep='first'), :]

    users = df.droplevel("movie_id")[user_cols(df)]
    users = users.loc[~users.index.duplicated(keep='first'), :]
    return users, movies


def prepare(X: pd.DataFrame, y):
    X = X.reset_index()
    user_ids, movie_ids = X.user_id, X.movie_id
    return np.array(user_ids), np.array(movie_ids), np.array(y)


def test_xgb_model():
    XGBMR.load_model(champion=False, device="cuda").recommend_for_users_batched(
        [0, 1],
        movieIds=[[0, 1], [10, 30]],
        max_rating=5,
        clamp=True,
        temps=[0.1, 0.3]
    )
    Logger.info("XGB model test passed successfully")


def main(
    ratings: pd.DataFrame,
    num_boost_round: int = 10_000,
    early_stopping_rounds=500,
    verbose_eval: int | bool = False,
    n_splits=3,
    pct_thresh=0.10,
    n_trials=10,
):
    def objective_func(trial: optuna.Trial):
        params = {
            'objective': trial.suggest_categorical('objective', ['reg:squarederror']),
            'device': trial.suggest_categorical('device', ["cuda"]),
            'random_state': trial.suggest_categorical('random_state', [0]),
            'eval_metric': trial.suggest_categorical('eval_metric', ["rmse"]),
            'booster': trial.suggest_categorical('booster', ['gbtree']),

            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'lambda': trial.suggest_float('lambda', 1e-3, 10, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10, log=True)
        }
        scores = cv(
            ratings, n_splits=n_splits, seed=0,
            params=params,
            num_boost_round=num_boost_round,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
        )
        return np.mean(scores)

    # Optimize
    Logger.info("Running CV...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_func, n_trials=n_trials,  # type: ignore
                   show_progress_bar=True)
    Logger.info("Found best params: %s", study.best_params)

    # Retrain
    train, test = simple_split(ratings, .8)
    processor = MovieLensPreprocessor().fit(train)
    X_train, X_test, y_train, y_test = processor.do_transform(train, test)
    booster = train_xgb(
        study.best_params,
        X_train, y_train,
        X_test, y_test,
        num_boost_round=num_boost_round,
        verbose_eval=1000,
        early_stopping_rounds=early_stopping_rounds,
        maximize=False,
    )

    importances = pd.DataFrame(booster.get_score(importance_type="gain"), index=[
                               "importance"]).T.reset_index(names="feature")
    cols = importances[importances.importance > (
        importances.importance.max() * pct_thresh)].feature.tolist()

    Logger.info("Found most impactful features: %s", cols)

    Logger.info("\n")

    Logger.info("Training XGBMR...")
    from movie_recommender.modeling.xgbmr import XGBMR

    def mae_(predt: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        return mae(predt, y)

    model = XGBMR(processor.user_data, processor.movie_data, cols=cols)
    _, run_id = model.fit(
        study.best_trial.params,
        X_train[model.cols], y_train,
        eval_set=(X_test[model.cols], y_test),
        experiment_name="movie_recom",
        verbose_eval=500,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        custom_metric=mae_,
    )
    Logger.info("\n")
    Logger.info("Saving plots...")
    save_plots(
        model,
        prepare(X_train[model.cols], y_train),
        prepare(X_test[model.cols], y_test),
        max_rating=5,
        run_id=run_id,
    )


if __name__ == "__main__":
    import sys
    import os
    arg = sys.argv[1]
    bucket = os.environ["DB_MINIO_BUCKET"]
    connect_minio()
    connect_mlflow()
    if arg == "train":
        ratings = read_parquet_from_s3(bucket, "ratings.parquet")
        movies = read_parquet_from_s3(bucket, "movies.parquet")
        ratings = ratings.merge(movies, on="movie_id")
        ratings.rating *= 5

        main(
            ratings,
            verbose_eval=1000
        )
    elif arg == "test":
        test_xgb_model()
    else:
        Logger.error(f'Invalid arg {arg}')
        sys.exit(1)
