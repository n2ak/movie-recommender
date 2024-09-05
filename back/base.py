import torch
from torch import nn, optim


def get_and_del(obj: dict, k: str):
    o = obj[k]
    del obj[k]
    return o


class BaseModel(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

    @classmethod
    def param_cls(self): raise NotImplementedError()

    def save(self, filename="models/model.pt", optimizer=None, **kwargs):
        kwargs.update({
            "model": self.state_dict(),
            "params": self.params.asdict(),
            "optim": optimizer.state_dict() if optimizer is not None else None,
        })
        torch.save(kwargs, filename)
        print("Model saved to:", filename)

    @classmethod
    def load(cls, filename="models/model.pt", optCls=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(filename, map_location=torch.device(device))
        model = cls(cls.param_cls()(**get_and_del(state, "params")))
        model.load_state_dict(get_and_del(state, "model"))
        print("Model loaded from:", filename)
        other = {k: state[k] for k in state.keys() if k != "optim"}
        if optCls is not None:
            optimizer = optCls(model.parameters())
            optimizer.load_state_dict(get_and_del(state, "optim"))
            return model, optimizer, other
        return model, None, other

    def device_(self): raise ""

    def predict(self, userIds, movieIds) -> torch.Tensor:
        device = self.device_()
        n_users = len(userIds)
        n_movies = len(movieIds)

        assert n_users != 0
        userIds = torch.repeat_interleave(
            torch.tensor(userIds), n_movies).to(device)
        movieIds = torch.tile(torch.tensor(movieIds), (n_users,)).to(device)
        with torch.no_grad():
            res, _ = self.forward(userIds, movieIds)
            return res.cpu().view(n_users, n_movies)

    def recommand(self, userIds, movieIds, start=0, count=None, ascateg=False):
        if len(movieIds) == 0:
            movieIds = torch.arange(0, self.params.nm)
        pred = self.predict(userIds, movieIds)
        if ascateg:
            pred = (pred*2).round()/2
        ratings, ids = pred.sort(-1, descending=True)
        end = start + count if count is not None else None
        ratings = ratings[:, start:end].tolist()
        ids = ids[:, start:end].tolist()
        return ids, ratings

    def train_tick(self, lossfn, metric, batch):
        self.train()
        device = self.device_()
        u, m, r = [b.to(device) for b in batch]
        pred, embeddings = self.forward(u, m)
        loss = lossfn(pred, r, embeddings, self.params)
        with torch.no_grad():
            m = metric(r.cpu().detach(), pred.cpu().detach()).item()
        return loss, m
