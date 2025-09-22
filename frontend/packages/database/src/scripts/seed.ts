import type { Prisma } from "@prisma/client";
import { PrismaClient } from "@prisma/client";
import fs from "fs";
import { LoremIpsum } from "lorem-ipsum";
import Papa from "papaparse";

type MovieCSVType = {
  movie_id: number, genres: string, year: number, title: string,
  movie_avg_rating: number, movie_total_rating: number, imdbId: string
  posters: string

};
type UserCSVType = {
  user_id: number, username: string
};
type RatingCSVType = {
  user_id: number, movie_id: number, rating: number,
  // time: Date
};

const CHUNK_SIZE = 100_000;


const prisma = new PrismaClient();
type FirstParameterType<T extends (...args: any) => any> = Parameters<T>[0];
type TX = FirstParameterType<FirstParameterType<typeof prisma.$transaction>>;

const Lorem = new LoremIpsum({
  sentencesPerParagraph: {
    max: 8,
    min: 4
  },
  wordsPerSentence: {
    max: 16,
    min: 4
  }
});

function random(max: number) {
  return Math.round(Math.random() * max);
}

function randomString(min: number, max: number) {
  if (min > max) throw Error("Min should be less or equal to max")

  const range = Math.round(Math.random() * (max - min));
  const count = min + range;
  return Lorem.generateWords(count);
}

function parse_csv<T>(path: string) {
  console.log("Parsing csv file", path);
  const file = fs.readFileSync(path, "utf8");
  return Papa.parse<T>(file, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
    // preview: 3,
  });
}

async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function chunked<T>(data: T[], fn: (t: { data: T[], skipDuplicates: boolean }) => Promise<{ count: number }>) {

  let sum = 0;
  for (let i = 0; i < data.length; i += CHUNK_SIZE) {
    const slice = data.slice(i, i + CHUNK_SIZE);
    const ret = await fn({
      data: slice,
      skipDuplicates: true,
    });
    sum += ret.count;
    if (i != data.length - 1) await sleep(.5);
  }
  return sum;
}

async function createUsers(tx: TX, data: UserCSVType[]) {
  const n = await chunked(
    data.map(({ user_id, username }): Prisma.UserModelCreateWithoutMovieReviewsInput => ({
      id: user_id,
      username,
      email: `${username}@email.com`,
      password: username, // NOTE: no password encryption for simplicity
    })),
    tx.userModel.createMany
  );
  console.log(`Created ${n} users.`);
}

async function createMovies(tx: TX, data: MovieCSVType[]) {
  const n = await chunked(
    data.map(({
      genres,
      movie_avg_rating: avgRating,
      imdbId: imdbId,
      title: name,
      movie_total_rating: ratingCount,
      year,
      movie_id,
      posters
    }): Prisma.MovieModelCreateWithoutReviewsInput => ({
      id: movie_id,
      imdbId,
      href: String(posters),
      avg_rating: avgRating,
      genres: genres.split("|"),
      title: String(name),
      total_ratings: Number(ratingCount),
      year: year,
      desc: randomString(50, 150),
      createdAt: new Date(year),
    })),
    tx.movieModel.createMany
  );
  console.log(`Created ${n} movies.`);
}

async function createReviews(tx: TX, data: RatingCSVType[]) {
  const n = await chunked(
    data.map(({ movie_id, user_id }): Prisma.MovieReviewCreateManyInput => ({
      title: randomString(5, 10),
      text: randomString(20, 60),
      ndislikes: random(100),
      nlikes: random(100),
      movieModelId: movie_id,
      userModelId: user_id,
    })),
    tx.movieReview.createMany
  );
  console.log(`Created ${n} reviews.`);
}

async function createdRatings(tx: TX, data: RatingCSVType[]) {
  const n = await chunked(
    data.map((
      { movie_id, user_id, rating }
    ): Prisma.UserMovieRatingCreateManyInput => ({
      movieModelId: movie_id,
      userModelId: user_id,
      rating: rating,
      timestamp: new Date(),
    })),
    tx.userMovieRating.createMany
  );
  console.log(`Created ${n} ratings.`);
}

async function main() {
  if (process.argv.length < 5) {
    throw Error(`Invalid command: expected 3 argumens, found ${process.argv.length - 2}`)
  }
  const [ratings_file, movies_file, users_file] = process.argv.slice(-3) as any;
  const users_csv = parse_csv<UserCSVType>(users_file);
  const movies_csv = parse_csv<MovieCSVType>(movies_file);
  const ratings_csv = parse_csv<RatingCSVType>(ratings_file);

  console.log("Pushing to db...");
  await prisma.$transaction(
    async (tx) => {
      await createUsers(tx, users_csv.data);
      await createMovies(tx, movies_csv.data);
      await createdRatings(tx, ratings_csv.data);
      await createReviews(tx, ratings_csv.data);
    },
    {
      timeout: 5 * 60 * 1000,
    }
  );
  console.log("Done.");
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
