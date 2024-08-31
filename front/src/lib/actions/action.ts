"use server";
import { signIn, signOut } from "@/auth";
import { AuthError } from "next-auth";
import { redirect } from "next/navigation";
import z from "zod";

// ...
export type LoginFormState =
  | {
      errors?: {
        username?: string;
        password?: string;
      };
      message?: string;
    }
  | undefined;
export async function authenticate(
  prevState: LoginFormState,
  formData: FormData
): Promise<LoginFormState> {
  try {
    console.log("Trying...");
    await signIn("credentials", formData);
    console.log("Correct crediantials");
  } catch (error) {
    if (error instanceof AuthError) {
      switch (error.type) {
        case "CredentialsSignin":
          return { message: "Invalid credentials." };
        default:
          return { message: "Something went wrong." };
      }
    }
    throw error;
  }
  redirect("/");
}
export async function logOut() {
  console.log("Logging out");
  await signOut({
    redirectTo: "/auth/login",
  });
  console.log("Logged out");
}
import moviesDb, { RatingSortBy } from "@/lib/db/movie";
import Backend from "../backend_api";
import { Prisma } from "@prisma/client";

export async function rateMovie(
  userId: number,
  movieId: number,
  rating: number
) {
  const parsed = z.number().min(1).max(5).safeParse(rating);
  if (!parsed.success) {
    throw parsed.error;
  }
  return moviesDb.rateMovie(userId, movieId, rating);
}
export async function getRecommendedMoviesForUser(
  userId: number,
  start: number,
  count: number
) {
  // const movieIds = await moviesDb.getMovieRatingForUser(userId, 0, 100);
  const resp = await Backend.getMoviesRecom(userId, start, count);
  const movies = moviesDb.getMovies(resp.ids);
  return movies;
}
export async function getRatingsForUser(
  userId: number,
  start: number,
  count: number,
  sortby: RatingSortBy,
  order: Prisma.SortOrder
) {
  // const movieIds = await moviesDb.getMovieRatingForUser(userId, 0, 100);
  const movies = moviesDb.getRatingsForUser(
    userId,
    start,
    count,
    sortby,
    order
  );
  return movies;
}
export async function getMovieForUser(userId: number, imdbId: number) {
  const movie = await moviesDb.getMovieForUser(userId, imdbId);
  return movie;
}
