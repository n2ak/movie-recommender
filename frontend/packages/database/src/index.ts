import { Prisma } from "@prisma/client";
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
export type { Movie, MovieForUser, User } from "./definitions";
export { moviesDb, type RatingWithMovie } from "./movie";
export type RatingSortBy = "rating" | "timestamp" | "title" | "avg_rating";
