import torch
import pandas as pd
import numpy as np


class Instance:
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
            cls.instance.init()
            print(cls.__name__, "has been initialized")
        return cls.instance


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def exclude(arr: list, *s: str):
    arr = arr[:]
    for i in s:
        assert isinstance(i, str)
        if i in arr:
            arr.remove(i)
    return arr


def read_csv(p, **kwargs) -> pd.DataFrame:
    p = get_dataset_path() / p
    return pd.read_csv(p, **kwargs)


def to_csv(df: pd.DataFrame, p: str, **kwargs):
    df.to_csv(get_dataset_path()/p, index=False, **kwargs)


def _get_base_path():
    import pathlib
    path = pathlib.Path(__file__).absolute().parent.parent
    return path


def get_dataset_path():
    return _get_base_path() / "dataset"


def get_models_path():
    return _get_base_path() / "models"
