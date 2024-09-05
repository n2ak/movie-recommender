from model import MF, MFParams
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
import asyncio
import pandas as pd
import tqdm
import numpy as np
try:
    from prisma import Prisma
except:
    print("No prisma")
    pass

import torch.utils.data


class DLI():
    def __init__(self, ds: "DL", l, bs) -> None:
        self.ds = ds
        self.len = l
        self.current = -1
        self.bs = bs

    def __next__(self):
        self.current += 1
        if self.current < self.len:
            return self.ds.get(self.current * self.bs)
        raise StopIteration

    def __len__(self): return self.len


class DL():
    def __init__(self, u, m, r, batch_size, shuffle=True) -> None:
        super().__init__()
        if shuffle:
            rp = torch.randperm(len(u))
            u = u[rp]
            m = m[rp]
            r = r[rp]
        self.userIds = u
        self.movieIds = m
        self.ratings = r

        self.nu = len(self.userIds.unique())
        self.nm = len(self.movieIds.unique())
        self.bs = batch_size

    def __iter__(self):
        import numpy as np
        return DLI(
            self,
            int(np.ceil(len(self.userIds)/self.bs)),
            self.bs,
        )

    def get(self, i):
        s, e = i, i+self.bs
        return self.userIds[s:e], self.movieIds[s:e], self.ratings[s:e]


def set_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


async def get_ratings(client: "Prisma", limit=None):
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


def create_target_matrix(ratings):
    return ratings.pivot(
        index='userId', columns='movieId', values='rating').fillna(0)


async def pre_process(pivot=False) -> None:
    prisma = Prisma()
    await prisma.connect()
    ratings = await get_ratings(prisma)
    await prisma.disconnect()
    if not pivot:
        return None, ratings
    rating_matrix = create_target_matrix(ratings)
    real_matrix = torch.from_numpy(rating_matrix.values)
    return real_matrix, ratings


def get_first_of_every_user(df):
    indices = []
    rows = []
    for i in range(df.userId.nunique()):
        if i == 0:
            continue
        row = df.loc[df.userId == i]
        indices.append(row.index[0])
        rows.append(row.iloc[0])
    df = df.drop(indices, axis=0)
    return df, pd.DataFrame(rows)


def split(ratings, split=None):
    ratings = ratings[["userId", "movieId", "rating"]]  # .sample(frac=1)
    nu, nm = ratings.userId.nunique(), ratings.movieId.nunique()
    if split is not None:
        dfs = get_first_of_every_user(ratings)
        # s = int(ratings.shape[0]*0.8)
        # dfs = ratings.iloc[:s], ratings.iloc[s:]
    else:
        dfs = [ratings]
    dfs = [
        (torch.from_numpy(df.userId.values).long(),
         torch.from_numpy(df.movieId.values).long(),
         torch.from_numpy(df.rating.values).float())
        for df in dfs
    ]
    del ratings
    return nu, nm, dfs


def main(epochs, ratings: pd.DataFrame = None, filename=None, load_model=False):
    if ratings is None:
        _, ratings = asyncio.run(pre_process())
        ratings.to_csv("train.csv", index=False)
    print("*"*10)
    # nu, nm = real_matrix.shape
    nu, nm, (df1, df2) = split(ratings, .8)
    nu = nu+1
    nm = nm+1

    train_dl = DL(*df1, 0x1000*2, shuffle=True)
    val_dl = DL(*df2, 0x100, shuffle=False)
    from ncf import default_ncf, NCF, NCFParams
    scheduler = None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(1337)
    if not load_model:
        ncf, ncf_opt, last_state = default_ncf(nu, nm, device)
    else:
        ncf, ncf_opt, last_state = NCF.load(
            filename, optCls=torch.optim.AdamW, device=device)

    def loss_fn(pred, real, embedings, _):
        return F.mse_loss(pred, real)
    print("training starts")
    losses, f1, val_losses, val_f1 = train2(
        ncf, ncf_opt, train_dl, loss_fn, lambda a, b, *
        _: compute_f1(a, b, None),
        epochs, device, scheduler=scheduler, val_ld=val_dl,
        last_state=last_state)
    ncf.save(filename, optimizer=ncf_opt,
             losses=losses, f1=f1, val_losses=val_losses, val_f1=val_f1)
    return losses, f1, val_losses, val_f1


def train2(
    model: MF,
    optimizer,
    dl,
    loss_fn,
    metric,
    epochs,
    device="cpu",
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    val_ld=None,
    last_state={}
):
    if val_ld is None:
        val_loss = np.nan
    losses = last_state.get("losses", [])
    ms = last_state.get("ms", [])
    val_losses = last_state.get("val_losses", [])
    val_ms = last_state.get("val_ms", [])
    lr = [group['lr'] for group in optimizer.param_groups]
    try:
        lowest_l = 9999999999
        lowest_e = 9999999999
        for epoch in (bar := tqdm.trange(len(losses), len(losses)+epochs)):
            l, f = [], []
            for batch in dl:
                loss, m = model.train_tick(loss_fn, metric, batch)
                l.append(loss.item())
                f.append(m)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if val_ld is not None:
                val_loss, val_m = eval(model, val_ld, loss_fn, metric)
                val_losses.append(val_loss)
                val_ms.append(val_m)

            losses.append(np.mean(l))
            ms.append(np.mean(f))
            if losses[-1] < lowest_l:
                lowest_l = losses[-1]
                lowest_e = epoch

            # scheduler.step(val_loss if val_ld is not None else losses[-1])
            if scheduler is not None:
                scheduler.step(epoch)
                lr = scheduler.get_last_lr()

            bar.set_description(
                f"Epoch:{epoch+1},LR:{lr[0]:.6f}/{epoch-lowest_e},Loss : {losses[-1]:.4f}/{lowest_l:.4f}, "
                f"Val loss: {(val_loss):.4f}")

    except KeyboardInterrupt:
        pass
    return losses, ms, val_losses, val_ms


def compute_f1(real: torch.Tensor, pred: torch.Tensor, params):
    with torch.no_grad():
        from sklearn.metrics import f1_score, precision_score, mean_squared_error
        real, pred = real.detach().cpu(), pred.detach().cpu()
        real = (real.flatten()*2).round().numpy()
        pred = (pred.flatten()*2).round().numpy()
        return f1_score(real, pred, average="weighted")


@torch.no_grad()
def eval(model: MF, dl, loss_fn, metric):
    model.eval()
    losses = []
    ms = []
    for batch in dl:
        loss, m = model.train_tick(loss_fn, metric, batch)
        ms.append(m)
        losses.append(loss.item())
    return np.mean(losses), np.mean(ms)


def plot(losses, f1, vl, vf, show_min_loss=True):
    import matplotlib.pyplot as plt

    def show(ax, t, v, label, min, ylim=[0, None]):
        c1 = ax.plot(t, label=label)[0]
        c2 = ax.plot(v, label=f"val_{label}")[0]
        ax.set_ylim(ylim)
        for arr, c in zip([t, v], [c1, c2]):
            c = c.get_color()
            if len(arr) and show_min_loss:
                if min:
                    m = np.min(arr)
                    a = np.argmin(arr)
                else:
                    m = np.max(arr)
                    a = np.argmax(arr)
                ax.axvline(x=a, color=c)
                ax.text(a-int(len(arr)/3), m, f"Min:{m:.4f}")
        ax.legend()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    show(axes[0], losses, vl, "Loss", True)
    show(axes[1], f1, vf, "F1", False, [0, 1])
    return fig, axes


if __name__ == "__main__":
    print("*************")
    epochs = 200
    filename = "models/ncf1.pt"
    losses, f1, val_losses, val_f1 = main(
        epochs,
        filename=filename,
    )
    fig, _ = plot(losses, f1, val_losses, val_f1)
    fig.savefig("result.png")
    print("*************")
