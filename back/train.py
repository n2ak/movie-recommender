from model import MF, train2, loss_fn, Params, train1
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
import asyncio
from prisma import Prisma
import pandas as pd


async def get_ratings(client: Prisma, limit=None):
    data = await client.usermovierating.find_many()
    users = list(map(lambda u: u.userModelId, data))
    movies = list(map(lambda u: u.movieModelId, data))
    ratings = list(map(lambda u: u.rating, data))
    df = pd.DataFrame({
        "userId": users,
        "movieId": movies,
        "rating": ratings,
    })
    print("ratings", df.shape)
    if limit is not None:
        df = df.iloc[:limit]
    df = df.sort_values(["userId", "movieId"])
    return df


async def pre_process() -> None:
    prisma = Prisma()
    await prisma.connect()
    ratings = await get_ratings(prisma)
    rating_matrix = ratings.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)
    await prisma.disconnect()
    print("ratings", ratings.shape)
    print("rating_matrix", rating_matrix.shape)
    real_matrix = torch.from_numpy(rating_matrix.values)
    return real_matrix, ratings


def main(epochs, params: Params, weight_decay=0, real_matrix=None, filename=None):
    if real_matrix is None:
        real_matrix, ratings = asyncio.run(pre_process())
    print("*"*10)
    print(ratings.head())
    nu, nm = real_matrix.shape
    params.nu = nu+1
    params.nm = nm+1
    print("Params", params)
    if filename is not None:
        model, optimizer = MF.load(filename, opt=True)
    else:
        model = MF(params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params.lr,
                               weight_decay=weight_decay)

    print("training starts")
    return model, optimizer, train1(model, optimizer, loss_fn, real_matrix, epochs, params)


if __name__ == "__main__":
    print("*************")
    ubs = 0x100
    mbs = 0x1000
    epochs = 100
    params = Params(-1, -1, ubs=ubs, mbs=mbs,
                    lr=1e-2, embd=32, use_gelu=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, optimizer, (losses, stds, f1) = main(
        epochs,
        params,
        filename="models/model.pt"
    )
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    ax1, ax2, ax3 = axes
    for ax, title, arr in zip(axes, ["Loss", "F1", "Std"], [losses, f1, stds]):
        ax.plot(arr, label=title)
        ax.set_ylim([0, None])
        ax.legend()
    ax2.set_ylim([0, 1])
    fig.savefig("result.png")
    model.save(optimizer=optimizer)
    print("*************")
