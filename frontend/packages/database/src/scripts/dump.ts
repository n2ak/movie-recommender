import { PrismaClient } from "@prisma/client";
import { createObjectCsvWriter } from "csv-writer";

const prisma = new PrismaClient();
const output_file = "../../../back/dataset/db.csv";

async function main() {
  const reviews = await prisma.userMovieRating.findMany({
    select: {
      user: {
        select: {
          id: true,
        },
      },
      movie: {
        select: {
          id: true,
          genres: true,
          avg_rating: true,
          total_ratings: true,
          createdAt: true,
        },
      },
      rating: true,
    },
  });
  function toRow({
    rating,
    movie: {
      id: movieId,
      genres,
      avg_rating,
      total_ratings,
      createdAt: movie_date,
    },
    user: { id: userId },
  }: (typeof reviews)[0]) {
    return {
      rating,
      movieId,
      genres,
      avg_rating,
      total_ratings,
      movie_date: movie_date.toLocaleDateString(),
      userId,
    };
  }
  const rows = reviews.map(toRow);
  const writer = createObjectCsvWriter({
    path: output_file,
    header: Object.keys(rows[0]!).map((k) => ({
      id: k,
      title: k,
    })),
  });
  await writer.writeRecords(rows);
  console.log("Output:", output_file);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
