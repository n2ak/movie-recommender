import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsTransformer


class Suggester:
    """
    Useful for suggesting movies to be ran through a DNN, to reduce the input to the DNN
    from millions to thousands or so. 
    """

    def __init__(self, n_neighbors_user=5, n_neighbors_movie=5) -> None:
        self.movie_knn = KNeighborsTransformer(n_neighbors=n_neighbors_movie)
        self.user_knn = KNeighborsTransformer(n_neighbors=n_neighbors_user)
        self.is_fit = False

    def fit(self, movies: pd.DataFrame, users: pd.DataFrame, ratings: pd.DataFrame):
        self.movieIds = movies.movie
        self.userIds = users.user
        self.movies = movies = movies.drop(columns="movie").astype(float)
        self.users = users = users.drop(columns="user").astype(float)
        self.movie_knn.fit(movies.values)
        self.user_knn.fit(users.values)

        self.is_fit = True
        self.ratings = ratings

    def contains_genres(self, genres, relation, df=None):
        if df is None:
            df = self.ratings
        assert isinstance(genres, (list, tuple))
        a = df.movie_genres.str.contains(genres[0])
        for g in genres[1:]:
            if relation == "and":
                a &= df.genres.str.contains(g)
            elif relation == "or":
                a |= df.genres.str.contains(g)
            else:
                raise Exception(f"Invalid relation '{relation}'")
        return a

    def suggest(
        self,
        user_id,
        nmovies,
        n_neighbor_users=10,
        n_neighbor_movies=100,
        n=10,
        genres: list[str] = [],
        add_closest_users=True,
        deterministic=False,
        relation=None,
    ):
        """
        Get most relevant movies to user, and users close to the user.
        """
        assert self.is_fit
        # top movies that the user has already rated
        top_movies = top_movies_for_user = self.get_users_top_movies(
            [user_id],
            deterministic=deterministic, n=n, genres=genres,
            relation=relation,
        )
        if len(top_movies_for_user):
            # get closest movies to user movies
            closest_movies_for_user = self.get_closest_movies(
                top_movies_for_user,
                n_neighbor_movies,
                genres=genres,
                relation=relation,
            )
            # get similar users
            if len(top_movies) and add_closest_users:
                closest_users = self.get_closest_users(
                    [user_id], n_neighbor_users)
                if len(closest_users):
                    # top movies for colsest users
                    top_movies_for_similar_users = self.get_users_top_movies(
                        self.userIds.iloc[closest_users].tolist(),
                        deterministic=deterministic, n=n,
                        genres=genres,
                        relation=relation,
                    )
                    # get only the intersection between top movies of similar users and the user himself
                    intersection = np.intersect1d(
                        top_movies_for_similar_users,
                        closest_movies_for_user,
                    )
                    if intersection.size > 0:
                        top_movies = intersection
            else:
                top_movies = top_movies_for_user
        else:
            # user hasn't rated any movies
            print("No ratings cant suggest")
            if len(genres):
                # TODO self.movies doesnt contain "genres"
                movies = self.ratings.drop_duplicates("movie")
                top_movies = movies.loc[self.contains_genres(
                    genres, relation, df=movies)
                ].movie
            else:
                top_movies = self.movieIds.unique()
        return top_movies[:nmovies]

    def get_users_top_movies(self, userIds: list[int], n=3, deterministic=False, genres=[], relation=None):
        ratings = self.ratings
        movie_ids = set()
        genres_condition = True
        if len(genres) > 0:
            assert relation in ["or", "and"]
            genres_condition = self.contains_genres(genres, relation)
        for uid in userIds:
            # TODO maybe no need for the loop
            condition = genres_condition & (ratings.user == uid)
            user_movie_ids = ratings.loc[condition].sort_values(
                "rating", ascending=False).iloc[:n*5].movie.values
            if len(user_movie_ids):
                if not deterministic:
                    movie_ids.update(np.random.choice(
                        user_movie_ids, n).tolist())
                else:
                    movie_ids.update(user_movie_ids[:n])
        return list(movie_ids)

    def get_closest_users(self, user_ids, k):
        users_data = self.users.loc[self.userIds.isin(user_ids)].values
        if users_data.shape[0] == 0:
            return []
        return self.user_knn.kneighbors(
            users_data,
            n_neighbors=k,
            return_distance=False,
        )[0]

    def get_closest_movies(self, movie_ids, k, genres, relation):
        movies_data = self.movies.loc[self.movieIds.isin(movie_ids)].values
        if movies_data.shape[0] == 0:
            return []
        movie_ids = self.movie_knn.kneighbors(
            movies_data,
            n_neighbors=k, return_distance=False,
        )[0]
        if len(genres):
            condition = self.contains_genres(genres, relation)
            movie_ids = self.ratings.loc[
                self.ratings.movie.isin(movie_ids) & condition].movie
        return movie_ids


def load_suggester():
    from .data import MovieLens

    movie_lens = MovieLens.get_instance()

    movie_cols = movie_lens.movie_cat_cols + movie_lens.movie_num_cols
    user_cols = movie_lens.user_cat_cols + movie_lens.user_num_cols
    movies = movie_lens.movies[movie_cols]
    users = movie_lens.users[user_cols]
    ratings = movie_lens.ratings

    suggester = Suggester()
    suggester.fit(
        movies,
        users,
        ratings
    )
    return suggester
