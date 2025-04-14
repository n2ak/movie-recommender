import datetime
import asyncio
try:
    from prisma import Prisma
except:
    print("No prisma")
    pass


def user_to_dict(user):
    return {
        "id": int(user.user),
        "username": f"user{user.user}",
        "password": "password",
        "email": f"user{user.user}@gmail.com",
    }


def movie_to_dict(movie):
    return {
        "id": int(movie.movie),
        "title": movie.movie_title,
        "genres": str(movie.movie_genres).split("|"),
        "imdbId": movie.movie_imdbId,
        "href": movie.movie_href,
        "avg_rating": movie.movie_avg_rating,
        "total_ratings": movie.total_ratings,
        "year": movie.movie_year,
    }


def rating_to_dict(rating):
    return {
        # data to create a UserMovieRating record
        'rating': rating.rating,
        'movieModelId': int(rating.movie),
        'userModelId': int(rating.user),
        'timestamp': rating.timestamp
    }


def get_dfs(nrows=None):
    from core.utils import read_csv

    movie_cols = ["movie", "movie_title", "movie_genres", "movie_imdbId",
                  "movie_href", "movie_avg_rating", "total_ratings",
                  "movie_year"]
    rating_cols = ['user', 'movie', 'rating', 'timestamp']
    final = read_csv("ratings_full.csv", nrows=nrows)

    print("*"*10, "Making dfs", "*"*10)
    users = final[["user"]].drop_duplicates("user").sort_values("user")
    movies = final[movie_cols].drop_duplicates("movie").sort_values("movie")
    final["timestamp"] = final.timestamp.map(datetime.datetime.fromtimestamp)
    ratings = final[rating_cols]
    print("*"*10, "Making entities", "*"*10)

    users = users.apply(user_to_dict, axis=1).tolist()
    movies = movies.apply(movie_to_dict, axis=1).tolist()
    ratings = ratings.apply(rating_to_dict, axis=1).tolist()
    return users, movies, ratings


async def main() -> None:

    prisma = Prisma()
    await prisma.connect()

    users, movies, ratings = get_dfs()

    await prisma.usermovierating.delete_many()
    await prisma.usermodel.delete_many()
    await prisma.moviemodel.delete_many()
    print("Deleted db")

    print("*"*10, "Filling db", "*"*10)
    await create_many(prisma.usermodel, users, "users")
    await create_many(prisma.moviemodel, movies, "movies")
    await create_many(prisma.usermovierating, ratings, "ratings")

    print("Done")
    await prisma.disconnect()


# async def create_many(model, data, name, batch_size=10_000):
#     print("*"*10, "Deleting old", name, "*"*10)
#     await model.delete_many()

#     print("*"*10, "Filling db with", len(data), name, "*"*10)
#     for i in range(0, len(data), batch_size):
#         batch = data[i:i+batch_size]
#         print("Batch:", len(batch))
#         total = await model.create_many(data=batch)
#         print("Created", total, name)
async def create_many(model, data, name, batch_size=20_000):
    sema = asyncio.Semaphore(4)

    tasks = []
    print("*"*10, "Filling db with", len(data), name, "*"*10)

    async def create(batch, s):
        async with sema:
            await model.create_many(data=batch)
            return s
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        s = f"created batch: {i}:{i+len(batch)} {name}"
        tasks.append(create(batch, s))

    result = await asyncio.gather(*tasks)
    for i in result:
        print(i)
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    asyncio.run(main())
