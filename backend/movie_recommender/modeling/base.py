from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import torch
import numpy as np
from torch import nn
from typing import Type, TypeVar, Generic
from numpy.typing import NDArray
from movie_recommender.utils import Timer
T = TypeVar('T')


class MovieRecommender(ABC, Generic[T]):
    def recommend_for_users_batched(
        self,
        userIds: list[int],
        movieIds: list[list[int]],
        max_rating: int,
        temps: list[float],
        wrap=False,
        clamp=True,
    ) -> list[list["Recommendation"]] | list[list[float]]:
        from ..utils import is_array
        assert is_array(userIds)
        assert is_array(movieIds)
        assert is_array(movieIds[0])
        assert isinstance(userIds[0], int)
        assert isinstance(movieIds[0][0], int)
        assert len(userIds) == len(movieIds)

        with Timer("Batch prepare"):
            batch = self._prepare_batch(userIds, movieIds)
        with Timer("Prediction"):
            ratings = np.concat([self.predict(b, max_rating) for b in batch])
        if clamp:
            ratings = np.clip(ratings, 0, max_rating)
        # if temp != 0: TODO
        #     ratings, userIds, movieIds = self.apply_temp(ratings, userIds, movieIds,temp)
        batch = self.unbatch(ratings, [len(m) for m in movieIds])
        if not wrap:
            return []
        return [self.to_pred([u] * len(m), m, r,) for u, m, r in zip(userIds, movieIds, batch)]

    def unbatch(self, batch: NDArray[np.float32], lengths: list[int]):
        arr: list[NDArray[np.float32]] = []
        s = 0
        for l in lengths:
            arr.append(batch[s:s+l])
            s += l
        assert [len(m) for m in arr] == lengths
        return arr

    def apply_temp(
        self,
        ratings: list[float] | np.ndarray,
        userIds: list[int] | np.ndarray,
        movieIds: list[int] | np.ndarray,
        temp: float,
    ):
        assert 0 < temp <= 1

        ratings = np.array(ratings, dtype=np.float32)

        def multinomial(array, count, nexp=1):
            count = min(count, len(array))
            array = torch.from_numpy(array)
            array = torch.softmax(array, -1)
            array /= temp
            return torch.multinomial(array, count).tolist()
        indices = multinomial(ratings, 10)
        ratings = ratings[indices]
        userIds = np.array(userIds, dtype=np.float32)[indices]
        movieIds = np.array(movieIds, dtype=np.float32)[indices]
        return ratings, userIds, movieIds

    @abstractmethod
    def predict(
        self,
        batch: T,
        max_rating: int,
    ) -> NDArray[np.float32]:
        raise NotImplementedError()

    @abstractmethod
    def _prepare_batch(
        self,
        userIds: list[int],
        movieIds: list[list[int]],
    ) -> list[T]:
        raise NotImplementedError()

    def to_pred(self, userIds, movieIds, preds):
        from ..utils import is_array
        assert is_array(userIds), type(userIds)
        assert is_array(movieIds), type(movieIds)
        assert is_array(preds), type(preds)
        assert len(userIds) == len(movieIds) == len(
            preds), (len(userIds), len(movieIds), len(preds))
        return [
            Recommendation(int(u), int(m), float(p)) for u, m, p in zip(userIds, movieIds, preds)]


@dataclass(eq=True)
class Recommendation:
    userId: int
    movieId: int
    predicted_rating: float
    asdict = asdict


class MLP(nn.Module):
    def __init__(self, layer_neurons: list[int], act: Type[nn.Module] = nn.ReLU, p=.0, final_act=None, norm=False):
        super().__init__()
        layers = []
        for i, (inc, outc) in enumerate(zip(layer_neurons[:-1], layer_neurons[1:])):
            last = i == len(layer_neurons)-2
            layers.append(nn.Linear(inc, outc, bias=not last))
            if not last:
                if norm:
                    layers.append(nn.BatchNorm1d(outc))
                layers.append(act())
                if p:
                    layers.append(nn.Dropout(p))
        if final_act:
            layers.append(final_act())
        self.seq = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.seq(x)
