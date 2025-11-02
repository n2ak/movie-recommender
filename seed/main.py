import httpx
import asyncio
import random
from lorem_text import lorem
import pandas as pd
from prisma import Prisma
import os

CHUNK_SIZE = 100_000
GENRES_SPLIT_CHAR = "-"
PROD_SPLIT_CHAR = "-"
EMBEDDING_URL = os.environ["EMBEDDING_URL"]


def try_split(a, c, default=""):
    if a is None:
        return []
    return a.split(c)


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


async def create_users(data, db: Prisma):
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


async def create_movies(data, db: Prisma):
    embeddings = await get_embeddings(EMBEDDING_URL, data.overview.tolist())
    data["embeddings"] = embeddings
    movies = []
    for _, row in data.iterrows():
        movies.append({
            "tmdbId": row.tmdbId,
            "href": str(row.posters),
            "genres": try_split(row.genres, GENRES_SPLIT_CHAR),
            "title": str(row.title),
            "release_date": row.release_date,

            "production_companies": try_split(row.production_companies, PROD_SPLIT_CHAR),
            "original_language": row.original_language,
            "overview": row.overview,
            "tagline": row.tagline,
            "avg_rating": row.vote_average,
            "total_ratings": int(row.vote_count),
            "credits": try_split(row.credits, GENRES_SPLIT_CHAR),
            "keywords": try_split(row.keywords, GENRES_SPLIT_CHAR),
            "duration": float(row.runtime),
        })
    n = await chunked_insert(movies, db.moviemodel.create_many)
    for _, row in data.iterrows():
        await helper(row, db)
    print(f"Created {n} movies.")


async def get_embeddings(url: str, datas: list):
    async with httpx.AsyncClient() as client:
        async def fetch(d):
            response = await client.post(url, json=d)
            return response.json()
        tasks = [fetch(d) for d in datas]
        responses = await asyncio.gather(*tasks)
        return responses


async def helper(movie, db: Prisma):
    vector_text = "[" + \
        ",".join(map(str, movie.embeddings)) + "]"

    await db.execute_raw(
        """
    UPDATE movie
    SET overview_encoded = $1::vector
    WHERE "tmdbId" = $2
    """,
        vector_text,
        movie.tmdbId
    )


async def create_ratings(data, db: Prisma):
    ratings = [
        {
            "tmdbId": int(row.tmdbId),
            "userModelId": int(row.user_id),
            "rating": float(row.rating),
        }
        for _, row in data.iterrows()
    ]
    n = await chunked_insert(ratings, db.usermovierating.create_many)
    print(f"Created {n} ratings.")


async def create_reviews(data, db: Prisma):
    reviews = [
        {
            "title": random_string(5, 10),
            "text": random_string(20, 60),
            "nlikes": random.randint(0, 100),
            "ndislikes": random.randint(0, 100),
            "tmdbId": int(row.tmdbId),
            "userModelId": int(row.user_id),
        }
        for _, row in data.iterrows()
    ]
    n = await chunked_insert(reviews, db.moviereview.create_many)
    print(f"Created {n} reviews.")


async def main():
    users_csv = pd.read_parquet("users.parquet")
    movies_csv = pd.read_parquet("movies.parquet")
    ratings_csv = pd.read_parquet("ratings.parquet")

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
