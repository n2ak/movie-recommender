import "server-only";
import { PrismaClient } from "@prisma/client";
import Backend from "../backend_api";

const prismaClientSingleton = () => {
  const client = new PrismaClient();
  async function start() {
    // await addAllUsers(client);
    // await addAllMovies(client);
    // await addAllMovieRatings(client);
    // Backend.train();
  }
  start();
  return client;
};

declare const globalThis: {
  prismaGlobal: ReturnType<typeof prismaClientSingleton>;
} & typeof global;

export const prismaClient = globalThis.prismaGlobal ?? prismaClientSingleton();

if (process.env.NODE_ENV !== "production")
  globalThis.prismaGlobal = prismaClient;

async function __rateMovie(
  client: any,
  userId: number,
  movieId: number,
  rating: number
) {
  const a = await client.userMovieRating.findFirst({
    where: {
      movieModelId: movieId,
      userModelId: userId,
    },
  });
  if (!a) {
    return await client.userMovieRating.create({
      data: {
        movieModelId: movieId,
        userModelId: userId,
        rating: rating,
      },
    });
  } else {
    return await client.userMovieRating.update({
      where: {
        id: a.id,
        movieModelId: movieId,
        userModelId: userId,
      },
      data: {
        rating: rating,
      },
    });
  }
}
