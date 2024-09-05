from base import BaseModel
from model import MF
from ncf import NCF
import flask as F
import torch

app = F.Flask(__name__)
MODEL_LOADERS = {
    "MF": lambda: MF.load("models/mf.pt")[0],
    "NCF": lambda: NCF.load("models/ncf1.pt")[0],
}
MODELS = {k: None for k in MODEL_LOADERS.keys()}


def legacy():
    import time
    time.sleep(.5)
    result = []
    data = F.request.json
    genres_wanted = data.get("genres", [])
    ratings = data["ratings"]
    # TODO : use genres
    # if isinstance(ratings, list):
    #     if len(ratings) == 0:
    #         result = parse(df.sort_values(
    #             ["rating"], ascending=False).iloc[0:100])
    #     else:
    #         all_genres = defaultdict(lambda: [])
    #         print("*"*20)
    #         print("Got", len(ratings), "movie ratings")
    #         for movie in ratings:
    #             movie_rating = MovieRating.new(movie)
    #             genres = movie_rating.movie.genres.split("|")
    #             for g in genres:
    #                 all_genres[g].append(movie_rating.rating)
    #         all_genres_mean = {k: (sum(v)/len(v))
    #                            for k, v in all_genres.items()}
    #         for i in ALL_GENRES:
    #             if i not in all_genres_mean:
    #                 all_genres_mean[i] = 0
    #         all_genres_mean = collections.OrderedDict(
    #             sorted(all_genres_mean.items()))
    #         print(all_genres_mean)
    return F.jsonify(result)


def get_model(name) -> BaseModel:
    assert name in MODELS.keys(), "Invalid model name."
    if MODELS[name] is None:
        print("Loading", name)
        MODELS[name] = MODEL_LOADERS[name]()
        print(f"Model '{name}' is loaded.")
    return MODELS[name]


def pprint(d: dict):
    import json
    print(json.dumps(d, indent=2))


@app.post("/movies-recom")
def get_movies_recom():
    import time
    # time.sleep(.5)
    data: dict = F.request.json
    userIds = data["userIds"]
    movieIds = data["movieIds"]
    start = data["start"]
    count = data["count"]
    ascateg = data.get("round", False)
    model_name = data.get("model", "MF")
    pprint({k: v for k, v in data.items() if k != "movieIds"})
    s = time.monotonic()
    ids, ratings = get_model(model_name).recommand(
        userIds,
        movieIds,
        start,
        count,
        ascateg=ascateg,
    )
    e = time.monotonic()
    return F.jsonify({
        "time": e-s,
        "result": [{"movieIds": i, "pred_ratings": p}for i, p in zip(ids, ratings)],
    })


ALL_GENRES = [
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'IMAX',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western',
]


# def parse(df):
#     res = []
#     for idx, row in df.iterrows():
#         res.append({
#             "name": row.title,
#             "genres": row.genres,
#             "rating": row.rating,
#             "imdbId": row.imdbId,
#         })
#     return res


if __name__ == "__main__":
    host = "localhost"
    port = 3333
    app.run(host, port, debug=True)
