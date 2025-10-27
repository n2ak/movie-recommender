import asyncio
import random
from datetime import datetime
from lorem_text import lorem
import pandas as pd
from prisma import Prisma

CHUNK_SIZE = 100_000


def random_string(min_words: int, max_words: int) -> str:
    if min_words > max_words:
        raise ValueError("min_words should be <= max_words")
    n = random.randint(min_words, max_words)
    return lorem.words(n)


async def chunked_insert(data, fn):
    total = 0
    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i:i + CHUNK_SIZE]
        res = await fn(chunk, skip_duplicates=True)
        total += len(chunk)
        print("\t", "Created chunk of", len(chunk), "records")
        await asyncio.sleep(0.5)
    return total


def parse_csv(path: str):
    print(f"Parsing CSV file: {path}")
    df = pd.read_csv(path)
    return df


async def create_users(data, db):
    users = [
        {
            "id": int(row.user_id),
            "username": row.username,
            "email": f"{row.username}@email.com",
            "password": row.username,
        }
        for _, row in data.iterrows()
    ]
    n = await chunked_insert(users, db.usermodel.create_many)
    print(f"Created {n} users.")


async def create_movies(data, db):
    movies = []
    for _, row in data.iterrows():
        movies.append({
            "id": int(row.movie_id),
            "imdbId": str(row.imdbId),
            "href": str(row.posters),
            "avg_rating": float(row.movie_avg_rating),
            "genres": row.genres.split("|"),
            "title": str(row.title),
            "total_ratings": int(row.movie_total_rating),
            "year": int(row.year),
            "desc": random_string(50, 150),
            "createdAt": datetime(int(row.year), 1, 1),
        })
    n = await chunked_insert(movies, db.moviemodel.create_many)
    print(f"Created {n} movies.")


async def create_ratings(data, db):
    ratings = [
        {
            "movieModelId": int(row.movie_id),
            "userModelId": int(row.user_id),
            "rating": float(row.rating),
            "timestamp": datetime.utcnow(),
        }
        for _, row in data.iterrows()
    ]
    n = await chunked_insert(ratings, db.usermovierating.create_many)
    print(f"Created {n} ratings.")


async def create_reviews(data, db):
    reviews = [
        {
            "title": random_string(5, 10),
            "text": random_string(20, 60),
            "nlikes": random.randint(0, 100),
            "ndislikes": random.randint(0, 100),
            "movieModelId": int(row.movie_id),
            "userModelId": int(row.user_id),
        }
        for _, row in data.iterrows()
    ]
    n = await chunked_insert(reviews, db.moviereview.create_many)
    print(f"Created {n} reviews.")


async def main():
    users_csv = parse_csv("users.csv")
    movies_csv = parse_csv("movies.csv")
    ratings_csv = parse_csv("ratings.csv")

    print("Connecting to database...")
    db = Prisma()
    await db.connect()

    print("Pushing to DB...")
    async with db.tx(timeout=5 * 60 * 1000) as tx:
        await create_users(users_csv, tx)
        await create_movies(movies_csv, tx)
        await create_ratings(ratings_csv, tx)
        await create_reviews(ratings_csv, tx)

    print("Done seeding.")

    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
