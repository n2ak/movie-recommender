import { Prisma, PrismaClient } from "@prisma/client";
import fs from "fs";
import { convert } from "html-to-text";
import Papa from "papaparse";

const prisma = new PrismaClient();
const input_file = "../../../back/scraping/output/merged.csv";

function parse_csv(path: string) {
  const file = fs.readFileSync(path, "utf8");
  return Papa.parse(file, {
    header: true,
    skipEmptyLines: true,
    // preview: 3,
  });
}
function getMovie({
  genre,
  image,
  avgRating,
  imdbid,
  name,
  desc,
  ratingCount,
  movie_date,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
}: any): Prisma.MovieModelCreateWithoutReviewsInput {
  genre = JSON.parse(genre);
  desc = convert(desc);

  return {
    imdbId: imdbid,
    href: image,
    avg_rating: Number(avgRating),
    genres: genre,
    title: name,
    total_ratings: Number(ratingCount),
    year: 1000,
    desc,
    createdAt: new Date(movie_date),
  };
}
function getUser({
  author,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
}: any): Prisma.UserModelCreateWithoutMovieReviewsInput {
  return {
    username: author,
    email: `${author}@email.com`,
    password: author,
  };
}
function getReview(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  row: any,
  movies: { id: number; imdbId: string }[],
  users: { id: number; username: string }[]
): Prisma.MovieReviewCreateManyInput {
  const { body, title, dislikes, likes, imdbid, author } = row;
  const text = convert(body);
  return {
    title,
    text,
    ndislikes: Number(dislikes),
    nlikes: Number(likes),
    movieModelId: movies.find((m) => m.imdbId === imdbid)!.id,
    userModelId: users.find((u) => u.username === author)!.id,
  };
}
function getMovieRating(
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  row: any,
  movies: { id: number; imdbId: string }[],
  users: { id: number; username: string }[]
): Prisma.UserMovieRatingCreateManyInput {
  const { imdbid, author, rating, date } = row;
  return {
    movieModelId: movies.find((m) => m.imdbId === imdbid)!.id,
    userModelId: users.find((u) => u.username === author)!.id,
    rating: Number(rating),
    timestamp: new Date(date),
  };
}
async function main() {
  const reviews_csv = parse_csv(input_file);

  console.log("Keys", reviews_csv.meta.fields);

  await prisma.$transaction(
    async (tx) => {
      // console.log("Resetting db...");
      // await tx.userModel.deleteMany();
      // await tx.movieModel.deleteMany();
      // console.log("Done");
      await tx.userModel.createMany({
        data: reviews_csv.data.map(getUser),
        skipDuplicates: true,
      });
      await tx.movieModel.createMany({
        data: reviews_csv.data.map(getMovie),
        skipDuplicates: true,
      });
      const movies = await tx.movieModel.findMany();
      const users = await tx.userModel.findMany();
      console.log("Pushing to db...");
      const nreviews = (
        await tx.movieReview.createMany({
          data: reviews_csv.data.map((r) => getReview(r, movies, users)),
          skipDuplicates: true,
        })
      ).count;
      const nratings = (
        await tx.userMovieRating.createMany({
          data: reviews_csv.data.map((r) => getMovieRating(r, movies, users)),
          skipDuplicates: true,
        })
      ).count;
      console.log("Created", nreviews, "reviews.");
      console.log("Created", nratings, "ratings.");
      console.log("Done.");
    },
    {
      timeout: 20 * 1000,
    }
  );
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
