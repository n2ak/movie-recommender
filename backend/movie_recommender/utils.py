from collections import defaultdict
import time
from contextlib import contextmanager
from sklearn.metrics import mean_absolute_error
from typing import Self
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from movie_recommender.logging import logger
sns.set_theme(style="darkgrid")
plt.style.use('dark_background')


class Singleton:
    instance = None

    @classmethod
    def get_instance(cls, **kwargs) -> Self:
        if cls.instance is None:
            instance = cls()
            instance.init(**kwargs)
            logger.info("%s has been initialized", cls.__name__)
            cls.instance = instance
        return cls.instance

    def init(self):
        raise NotImplementedError()


def display(a):
    from IPython.display import display
    display(a)


def is_array(arr):
    return isinstance(arr, (list, np.ndarray))


def report(
    model,
    userIds: list | np.ndarray,
    movieIds: list | np.ndarray,
    y: list | np.ndarray,
    max_rating: int,
    figsize=(16, 8),
    title="",
):
    recommendations = model.recommend_for_users(
        userIds,
        movieIds,
        max_rating,
        clamp=True,
        wrap=True,
        temp=0,
    )
    df = pd.DataFrame(dict(
        userIds=[r.userId for r in recommendations],
        movieIds=[r.movieId for r in recommendations],
        rating=y,
        pred=[r.predicted_rating for r in recommendations],
    ))
    assert 0 <= df.rating.max() <= max_rating, (df.rating.max(), max_rating)
    assert 0 <= df.pred.max() <= max_rating

    df["error"] = df.rating - df.pred

    logger.info(
        f"True  : mean {df.rating.mean():.3f}, std {df.rating.std():.3f}")
    logger.info(f"Pred  : mean {df.pred.mean():.3f}, std {df.pred.std():.3f}")
    logger.info(f"MAE   : {mean_absolute_error(df.rating, df.pred)}")
    logger.info(
        f"Error : mean {df.error.mean():.3f}, std {df.error.std():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()  # type: ignore
    sns.histplot(df.rating, fill=True, label="true",  # type: ignore
                 ax=axes[0], bins=10, kde=False)
    sns.histplot(df.pred, fill=True, label="pred", ax=axes[0],  # type: ignore
                 bins=10, kde=False)
    sns.histplot(df.error, fill=True, label="error",  # type: ignore
                 ax=axes[1], kde=True, bins=20)
    sns.barplot(df.astype(int).groupby(
        "rating").mean().error.abs(), ax=axes[2])  # type: ignore
    sns.barplot(
        data=(
            pd.DataFrame(
                dict(
                    Pred=bins(df.pred.round().astype(int), max_rating+1),
                    TargetRating=bins(
                        df.rating.round().astype(int), max_rating+1),
                    rating=np.arange(max_rating+1),
                ))
            .melt(id_vars=["rating"], var_name="cat", value_name="count")
        ),
        y="count", x="rating", hue="cat",
        ax=axes[3],
    )

    axes[1].set_title("Prediction error")
    axes[2].set_title("Mean error of each rating")
    axes[3].set_title("Number of ratings")

    axes[1].set_xlim([-max_rating, max_rating])
    axes[2].set_ylim([0, max_rating])

    # for a in axes:
    #     a.legend()
    plt.legend()
    plt.suptitle(title)
    plt.tight_layout()

    return df


def bins(arr, nbins):
    unique, counts = np.unique(arr, axis=0, return_counts=True)
    arr = np.zeros((nbins,))
    arr[unique] = counts
    return arr


indent = 0

timers = defaultdict(float)


@contextmanager
def Timer(name, mute=True):
    global indent
    indent += 1
    s = time.monotonic()
    yield timers
    e = time.monotonic()
    delta = (e-s)*1000
    indent -= 1
    if indent == 0:
        if not mute:
            for k, v in timers.items():
                print(f'{k}: {v:.0f}ms')
            print('\t'*indent,
                  f"******* {name} ended in {delta:.0f}ms *********")
        timers.clear()
    else:
        timers[name] += delta
