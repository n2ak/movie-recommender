import typing
import numpy as np

MAX_RATING = 5

T = typing.TypeVar("T")


def mae(logits, y) -> tuple[str, float]:
    max_rating = MAX_RATING
    y = y * max_rating
    logits = logits * max_rating
    return "mae", np.abs(logits - y).mean().item()


def rmse(logits, y) -> tuple[str, float]:
    return "rmse", np.sqrt(np.power(logits - y, 2).mean()).item()


def get_env(name: str, default: T) -> T:
    import os
    val = str(os.environ.get(name, default))
    if isinstance(default, bool):
        return val.lower() in ["true", "1"]  # type: ignore
    return default.__class__(val)  # type: ignore


def read_ds(table_name: str, db_url):
    import pandas as pd
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    table = pd.read_sql_table(table_name, engine)
    return table
