

import pandas as pd
from train_utils import get_env, read_ds
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from movie_recommender.logging import logger
from movie_recommender.sim_search import SimilaritySearch
from movie_recommender.modeling.workflow import save_figures

simsearch_exp_name = "SimilaritySearch"


def train_simsearch(ratings: pd.DataFrame, movies: pd.DataFrame, max_rating: int, tracking_uri: str):
    # ratings, movies = read_parquet(rating_path, movies_path)
    if isinstance(movies.movie_genres[0], str):
        movies.movie_genres = movies.movie_genres.apply(lambda x: x.split(","))

    # needed for somereason
    ratings = ratings.rename(str, axis="columns")
    movies = movies.rename(str, axis="columns")

    simsearch = SimilaritySearch().fit(ratings.copy(), movies.copy(), max_rating)
    run_id = simsearch.save(simsearch_exp_name, tracking_uri)
    return run_id


def test_simsearch():

    figures: dict[str, Figure] = {}

    simsearch = SimilaritySearch.load_from_disk(simsearch_exp_name)
    test_movies(simsearch, figures)
    test_users(simsearch, figures)

    genres = ["Romance", "Mystery", "Animation"]
    movies = simsearch.movies_with_genres(
        genres,
    )
    assert all([any([(g in c) for c in movies.columns]) for g in genres])
    simsearch.get_users_top_movies_ids(
        [1, 3],
        genres=["Romance"]
    )
    uid = 1
    movie_ids = simsearch.suggest(
        uid,
        add_closest_users=True,
        filter_watched=True,
        genres=("Romance",)
    )
    assert simsearch._get_movies(
        movie_ids)[["movie_genre_Romance"]].all().item()

    # save_figures(figures, run_id=run_id)
    logger.info("Similarity Search test passed successfully")


def test_users(simsearch: SimilaritySearch, figures: dict[str, Figure]):

    user_ids = [100, 200]
    closest_users = simsearch.get_closest_users(
        user_ids=user_ids,
        k=100,
        ret_df=True
    )
    X = simsearch.users.drop(columns=["user_id"])
    pca = PCA(2).fit(X)

    positions1 = pca.transform(simsearch._get_users(
        user_ids).drop(columns=["user_id"]))
    positions2 = pca.transform(simsearch._get_users(
        closest_users).drop(columns=["user_id"]))
    positions3 = pca.transform(X)

    plt.scatter(positions3[:, 0], positions3[:, 1], alpha=.1)
    plt.scatter(positions2[:, 0], positions2[:, 1], c="blue", label="Closest")
    plt.scatter(positions1[:, 0], positions1[:, 1],
                c="red", label="Target users")
    figures["users.png"] = plt.gcf()
    plt.close()


def test_movies(simsearch: SimilaritySearch, figures: dict[str, Figure]):
    movie_ids = [1000, 2000]

    closest_movies = simsearch.get_closest_movies(
        movie_ids=movie_ids,
        k=100,
    )
    X = simsearch.movies.drop(columns=["movie_id"])
    pca = PCA(2)

    positions1 = pca.fit_transform(X)
    positions2 = pca.transform(simsearch._get_movies(
        closest_movies).drop(columns=["movie_id"]))
    positions3 = pca.transform(simsearch._get_movies(
        movie_ids).drop(columns=["movie_id"]))

    plt.scatter(positions1[:, 0], positions1[:, 1], alpha=.1)
    plt.scatter(positions2[:, 0], positions2[:, 1], c="blue", label="Closest")
    plt.scatter(positions3[:, 0], positions3[:, 1],
                c="red", label="Target movies")

    figures["movies.png"] = plt.gcf()
    plt.legend()


if __name__ == "__main__":
    import sys
    import mlflow
    uri = get_env("MLFLOW_TRACKING_URI", "http://localhost:8081")
    mlflow.set_tracking_uri(uri)
    arg = sys.argv[1]
    if arg == "train":
        db_url = get_env(
            "DB_URL", 'postgresql+psycopg2://admin:password@localhost:5432/mydb')
        ratings = read_ds("ratings", db_url)
        movies = read_ds("movies", db_url)
        train_simsearch(ratings, movies, 5, uri)
    elif arg == "test":
        test_simsearch()
    else:
        logger.error(f'Invalid arg {arg}')
        sys.exit(1)
