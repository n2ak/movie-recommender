import numpy as np
import pandas as pd


class SimilaritySearch:
    def __init__(self, df: pd.DataFrame, encoded_key):
        import faiss

        assert df.shape[1] == 1

        # get direction
        embds = np.stack(df[encoded_key].values)
        embds = embds / np.linalg.norm(embds, axis=1, keepdims=True)

        self.index = faiss.IndexFlatL2(embds.shape[1])
        self.index.add(embds)  # type: ignore
        self.df = df
        self.encoded_key = encoded_key

    def get_movies_embedding(self, movie_ids: list[int]):
        return np.stack(self.df.loc[movie_ids][self.encoded_key].values)

    def suggest_similar_movies(self, movie_ids: list[int], genres: list[str] = [], k=10, ):
        # TODO: use genres
        assert isinstance(movie_ids, list)
        assert isinstance(movie_ids[0], int)
        embedding = self.get_movies_embedding(movie_ids)
        d, indices = self.index.search(embedding, k)  # type: ignore
        indices = indices.flatten()
        return self.df.iloc[indices].index.values.reshape(len(movie_ids), -1)

    @classmethod
    def load(cls, db_url: str, movies_table: str):
        from sqlalchemy import create_engine
        engine = create_engine(db_url)
        movies = pd.read_sql_table(movies_table, engine, columns=[
                                   "tmdbId", "overview_encoded"])
        movies.rename(columns={
            "tmdbId": "id",
            "overview_encoded": "embeddings"
        }, inplace=True)
        import json
        movies["embeddings"] = movies.embeddings.apply(json.loads)

        return cls(
            movies.set_index("id"), "embeddings"
        )
