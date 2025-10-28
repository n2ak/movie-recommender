import torch

from movie_recommender.dlrm.dlrm import DLRM
from movie_recommender.common.base_api import API
from movie_recommender.common.logging import Logger


class DLRMLitAPI(API):

    def setup(self, device):
        super().setup(device)
        Logger.info(f"Setting up DLRM LitAPI on device {device}")
        self.model = DLRM.load(champion=True, device=device)
        Logger.info("Setup complete.")

        self.num_cols = self.model.params.num_cols
        self.cat_cols = self.model.params.cat_cols

    def _prepare(self, users, movies):
        all = users.merge(movies, how="cross")
        num = torch.from_numpy(all[self.num_cols].values).float()
        cat = torch.from_numpy(all[self.cat_cols].values)
        return {
            "num": num,
            "cat": cat,
        }

    def predict(self, input):
        return self.model.predict(input)
