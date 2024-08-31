from model import MF
import collections
from collections import defaultdict
import dataclasses
import flask as F
import torch

app = F.Flask(__name__)

# @dataclasses.dataclass
# class Movie:
#     imdbId: int
#     rating: float
#     name: str
#     genres: str

#     @staticmethod
#     def new(d: dict):
#         return Movie(
#             imdbId=d["imdbId"],
#             rating=d["rating"],
#             name=d["name"],
#             genres=d["genres"],
#         )


# @dataclasses.dataclass
# class MovieRating:
#     id: int
#     rating: float
#     movieModelId: int
#     userModelId: int
#     movie: Movie

#     @staticmethod
#     def new(d: dict):
#         return MovieRating(
#             id=d["id"],
#             rating=d["rating"],
#             movieModelId=d["movieModelId"],
#             userModelId=d["userModelId"],
#             movie=Movie.new(d["movie"]),
#         )


# @app.post("/movies-recom")
def legacy():
    import time
    time.sleep(.5)
    result = []
    data = F.request.json
    genres_wanted = data.get("genres", [])
    ratings = data["ratings"]
    # TODO : use genres
    if isinstance(ratings, list):
        if len(ratings) == 0:
            result = parse(df.sort_values(
                ["rating"], ascending=False).iloc[0:100])
        else:
            all_genres = defaultdict(lambda: [])
            print("*"*20)
            print("Got", len(ratings), "movie ratings")
            for movie in ratings:
                movie_rating = MovieRating.new(movie)
                genres = movie_rating.movie.genres.split("|")
                for g in genres:
                    all_genres[g].append(movie_rating.rating)
            all_genres_mean = {k: (sum(v)/len(v))
                               for k, v in all_genres.items()}
            for i in ALL_GENRES:
                if i not in all_genres_mean:
                    all_genres_mean[i] = 0
            all_genres_mean = collections.OrderedDict(
                sorted(all_genres_mean.items()))
            print(all_genres_mean)
    return F.jsonify(result)


@app.post("/movies-recom")
def get_movies_recom():
    import time
    time.sleep(.5)
    data = F.request.json
    userIds = data["userIds"]
    movieIds = data["movieIds"]
    start = data["start"]
    count = data["count"]
    ids, ratings = model.predict(userIds, movieIds, start, count, device)
    return F.jsonify({
        "ratings": ratings,
        "ids": ids
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


def parse(df):
    res = []
    for idx, row in df.iterrows():
        res.append({
            "name": row.title,
            "genres": row.genres,
            "rating": row.rating,
            "imdbId": row.imdbId,
        })
    return res


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(filename, device="cpu",) -> MF:
    model = MF.load(filename, opt=False).to(device)
    return model


model: MF = None
if __name__ == "__main__":
    host = "localhost"
    port = 3333
    model = load_model("models/model.pt", device)
    app.run(host, port, debug=True)
