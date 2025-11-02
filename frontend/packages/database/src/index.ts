import { Prisma } from "@prisma/client";
// import "server-only";
export type MovieGenre =
  | "Action"
  | "Adventure"
  | "Animation"
  | "Children"
  | "Comedy"
  | "Crime"
  | "Documentary"
  | "Drama"
  | "Fantasy"
  | "Film-Noir"
  | "Horror"
  | "IMAX"
  | "Musical"
  | "Mystery"
  | "Romance"
  | "Sci-Fi"
  | "Thriller"
  | "War"
  | "Western";
export type SortOrder = Prisma.SortOrder;
export * as userDB from "./authDb";
export * as movieDB from "./movieDb";
export * as reviewsDB from "./reviewsDb";
// export type RatingSortKey = "rating" | "timestamp" | "title" | "avg_rating";
export { Prisma, type MovieModel } from "@prisma/client";
export { prismaClient } from "./connect";
export type { RatedMoviesRatingSortKey as RatingSortKey } from "./movieDb";

export const {
  PrismaClientInitializationError,
  PrismaClientRustPanicError,
  PrismaClientValidationError,
  PrismaClientKnownRequestError,
  PrismaClientUnknownRequestError,
} = Prisma;
export const MAX_RATING = 5;
