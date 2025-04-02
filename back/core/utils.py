import torch
import pandas as pd
import numpy as np


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def exclude(arr: list, *s: str):
    for i in s:
        assert isinstance(i, str)
        if i in arr:
            arr.remove(i)
    return arr


def read_csv(p, **kwargs):
    p = get_dataset_path() / p
    return pd.read_csv(p, **kwargs)


def to_csv(df: pd.DataFrame, p: str, **kwargs):
    df.to_csv(get_dataset_path()/p, **kwargs)


def _get_base_path():
    import pathlib
    path = pathlib.Path(__file__).absolute().parent.parent
    return path


def get_dataset_path():
    return _get_base_path() / "dataset"


def get_models_path():
    return _get_base_path() / "models"


# async def pre_process(pivot=False) -> None:
#     from prisma import Prisma
#     prisma = Prisma()
#     await prisma.connect()
#     ratings = await get_ratings(prisma)
#     await prisma.disconnect()
#     if not pivot:
#         return None, ratings
#     rating_matrix = create_target_matrix(ratings)
#     real_matrix = torch.from_numpy(rating_matrix.values)
#     return real_matrix, ratings


# def create_target_matrix(ratings):
#     return ratings.pivot(
#         index='userId', columns='movieId', values='rating').fillna(0)


# async def get_ratings(client, limit=None):
#     """Get ratings"""
#     data = await client.usermovierating.find_many()
#     users = list(map(lambda u: u.userModelId, data))
#     movies = list(map(lambda u: u.movieModelId, data))
#     ratings = list(map(lambda u: u.rating, data))
#     df = pd.DataFrame({
#         "userId": users,
#         "movieId": movies,
#         "rating": ratings,
#     })
#     print("ratings", df.shape)
#     if limit is not None:
#         df = df.iloc[:limit]
#     df = df.sort_values(["userId", "movieId"])
#     return df
