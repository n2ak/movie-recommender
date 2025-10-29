import sys
import os
sys.path.append(os.path.abspath("."))
if True:
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from sklearn.decomposition import PCA

    from movie_recommender.common.logging import Logger
    from movie_recommender.simsearch.sim_search import SimilaritySearch
    from training.data import process_data_for_simsearch
    from movie_recommender.common.workflow import MlflowClient, StorageClient

simsearch_exp_name = "SimilaritySearch".lower()


def train_simsearch(train: pd.DataFrame, exp_name=simsearch_exp_name):
    simsearch = SimilaritySearch().fit(train)
    run_id = simsearch.save(exp_name)
    return run_id


def test_simsearch():
    figures: dict[str, Figure] = {}

    simsearch = SimilaritySearch.load_from_disk(champion=False)
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

    MlflowClient.get_instance().save_figures(figures, run_id=simsearch.run_id)
    Logger.info("Similarity Search test passed successfully")


def test_users(simsearch: SimilaritySearch, figures: dict[str, Figure]):

    user_ids = [10, 4]
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
    movie_ids = [3, 10]

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
    bucket = os.environ["DB_MINIO_BUCKET"]

    ratings, movies = StorageClient.get_instance().download_parquet_from_bucket(
        bucket, "ratings", "movies"
    )
    train = process_data_for_simsearch(
        ratings,
        movies,
        1,
    )
    train_simsearch(train)
    test_simsearch()

    Logger.info("Done.")
