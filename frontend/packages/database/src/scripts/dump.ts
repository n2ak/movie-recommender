import { PrismaClient } from "@prisma/client";
import { createObjectCsvWriter } from "csv-writer";

type RatingType = {
  movieId: number,
  movie_genres: string[],
  movie_year: number,
  title: string,
  movie_avg_rating: number,
  movie_total_rating: number,
  imdbId: number,
  userId: number,
  rating: number,
  time: Date,
};

const prisma = new PrismaClient();
const output_file = "../../../back/dataset/db/db.csv";

async function getData() {
  const reviews = await prisma.userMovieRating.findMany({
    select: {
      user: {
        select: {
          id: true,
        },
      },
      movie: true,
      rating: true,
      timestamp: true
    },
  });
  const rows = reviews.map(({
    rating,
    timestamp: time,
    movie: {
      total_ratings: movie_total_rating,
      genres: movie_genres,
      avg_rating: movie_avg_rating,
      id: movieId,
      imdbId,
      title,
      year: movie_year
    },
    user: {
      id: userId
    },

  }): RatingType => ({
    imdbId,
    movie_avg_rating,
    movie_genres,
    movie_total_rating,
    movie_year,
    movieId,
    title,
    rating,
    userId,
    time
  }));
  return rows
}

async function main() {
  console.log("Getting data from db...");
  const rows = await getData();
  const writer = createObjectCsvWriter({
    path: output_file,
    header: Object.keys(rows[0]!).map((k) => ({
      id: k,
      title: k,
    })),
  });
  console.log("Writing to file...");
  await writer.writeRecords(rows);
  console.log("Output:", output_file);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => await prisma.$disconnect());