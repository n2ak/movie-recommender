import { Prisma, PrismaClient } from "@prisma/client";
import fs from "fs";
import { LoremIpsum } from "lorem-ipsum";
import Papa from "papaparse";

type MovieCSVType = {
  movieId: number, movie_genres: string, movie_year: number, title: string,
  movie_avg_rating: number, movie_total_rating: number, imdbId: number
};
type UserCSVType = {
  userId: number, username: string
};
type RatingCSVType = {
  userId: number, movieId: number, rating: number, time: Date
};

const DUMMY_HREF = "blank";
const CHUNK_SIZE = 100_000;

const USERS_FILE = "../../../back/dataset/db/users.csv";
const MOVIES_FILE = "../../../back/dataset/db/movies.csv";
const RATINGS_FILE = "../../../back/dataset/db/ratings.csv";

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
  const file = fs.readFileSync(path, "utf8");
  return Papa.parse<T>(file, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
    // preview: 3,
  });
}

async function chunked<T>(data: T[], fn: (t: { data: T[], skipDuplicates: boolean }) => Promise<{ count: number }>) {
  const sum = (numbers: number[]) => numbers.reduce((acc, curr) => acc + curr, 0);

  const chunkedArray: T[][] = [];
  for (let i = 0; i < data.length; i += CHUNK_SIZE) {
    chunkedArray.push(data.slice(i, i + CHUNK_SIZE));
  }
  const promises = await Promise.all(chunkedArray.map(d => fn({
    data: d,
    skipDuplicates: true,
  })));
  return sum(promises.map(p => p.count));
}

async function createUsers(tx: TX, data: UserCSVType[]) {
  const n = await chunked(
    data.map(({ userId, username, }): Prisma.UserModelCreateWithoutMovieReviewsInput => ({
      id: userId,
      username,
      email: `${username}@email.com`,
      password: username,
    })),
    tx.userModel.createMany
  );
  console.log(`Created ${n} users.`);
}

async function createMovies(tx: TX, data: MovieCSVType[]) {
  const n = await chunked(
    data.map(({
      movie_genres,
      movie_avg_rating: avgRating,
      imdbId: imdbId,
      title: name,
      movie_total_rating: ratingCount,
      movie_year,
      movieId,
    }): Prisma.MovieModelCreateWithoutReviewsInput => ({
      id: movieId,
      imdbId,
      href: DUMMY_HREF,
      avg_rating: avgRating,
      genres: movie_genres.split("|"),
      title: String(name),
      total_ratings: Number(ratingCount),
      year: movie_year,
      desc: randomString(200, 300),
      createdAt: new Date(movie_year),
    })),
    tx.movieModel.createMany
  );
  console.log(`Created ${n} movies.`);
}

async function createReviews(tx: TX, data: RatingCSVType[]) {
  const n = await chunked(
    data.map(({ movieId, userId }): Prisma.MovieReviewCreateManyInput => ({
      title: randomString(10, 50),
      text: randomString(100, 300),
      ndislikes: random(100),
      nlikes: random(100),
      movieModelId: movieId,
      userModelId: userId,
    })),
    tx.movieReview.createMany
  );
  console.log(`Created ${n} reviews.`);
}

async function createdRatings(tx: TX, data: RatingCSVType[]) {
  const n = await chunked(
    data.map((
      { movieId, userId, rating, time }
    ): Prisma.UserMovieRatingCreateManyInput => ({
      movieModelId: movieId,
      userModelId: userId,
      rating: rating,
      timestamp: new Date(time),
    })),
    tx.userMovieRating.createMany
  );
  console.log(`Created ${n} ratings.`);
}

async function main() {
  console.log("Parsing csv files...");
  const users_csv = parse_csv<UserCSVType>(USERS_FILE);
  const movies_csv = parse_csv<MovieCSVType>(MOVIES_FILE);
  const ratings_csv = parse_csv<RatingCSVType>(RATINGS_FILE);

  await prisma.$transaction(
    async (tx) => {
      console.log("Pushing to db...");
      await createUsers(tx, users_csv.data);
      await createMovies(tx, movies_csv.data);
      await createdRatings(tx, ratings_csv.data);
      await createReviews(tx, ratings_csv.data);
    },
    {
      timeout: 60 * 1000,
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
