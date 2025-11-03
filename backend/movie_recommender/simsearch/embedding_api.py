from litserve import LitAPI
import numpy as np
import pandas as pd


class EmbeddingAPI(LitAPI):
    def setup(self, device):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def predict(self, input):
        embeddings = self.model.encode(input)
        return embeddings.tolist()
