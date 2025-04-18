"use server";

import {
  Backend,
  BackendGetMoviesRecomResponse,
  ModelType,
} from "@repo/backend_api";
import { movieDB, MovieGenre, RatingSortKey } from "@repo/database";
import { MovieWithUserRating } from "../../../../packages/database/src/movieDb";
import { parseRating } from "../validation";
import { actionWrapWithError, getUserId } from "./utils";

async function handleBackendResponse(
  response: BackendGetMoviesRecomResponse,
  userId: number
) {
  const ret: {
    error:
      | undefined
      | {
          [index: string]: string;
        };
    result: {
      movies: MovieWithUserRating[] | undefined;
      predictions: number[] | undefined;
    };
  } = {
    error: undefined,
    result: {
      movies: undefined,
      predictions: undefined,
    },
  };
  if (response.status_code != 200) {
    ret.error = response.error;
  } else {
    ret.result.predictions = response.result.map((p) => p.predicted_rating);
    ret.result.movies = await movieDB.getMovies(
      response.result.map((p) => p.movieId),
      userId
    );
  }
  return ret;
}

export const rateMovie = actionWrapWithError()(async (
  movieId: number,
  rating: number
) => {
  parseRating(rating);
  const { userId } = await getUserId();
  await movieDB.rateMovie(userId, movieId, rating);
});

export async function getRecommendedMoviesForUser(
  userId: number,
  model: ModelType
) {
  // TODO : get unwatched/unrated movie ids.
  const resp = await Backend.getMoviesRecom(userId, model, null);
  return handleBackendResponse(resp, userId);
}

export async function getRecommendedGenreMovies(
  userId: number,
  genre: MovieGenre,
  model: ModelType
) {
  const resp = await Backend.getGenreRecom(userId, model, [genre], null);
  return handleBackendResponse(resp, userId);
}

export const getRatedMoviesForUser = async (
  userId: number,
  start: number,
  count: number,
  sortKey: RatingSortKey,
  order: "asc" | "desc"
) => await movieDB.getRatedMoviesForUser(userId, start, count, sortKey, order);

export const getMovieReviews = async (
  currentUserId: number,
  movieId: number,
  start: number,
  count: number,
  sortKey: any,
  order: "asc" | "desc"
) =>
  await movieDB.getMovieReviews(
    currentUserId,
    movieId,
    start,
    count,
    sortKey,
    order
  );

export type MovieReview = Awaited<ReturnType<typeof getMovieReviews>>[0];

export const getNumberOfRatings = async (userId: number) =>
  await movieDB.getNumberOfRatings(userId);

export const likeMovieReview = async (userId: number, movieReviewId: number) =>
  await movieDB.likeMovieReview(userId, movieReviewId);

export const getNumberOfMovieReviews = async (movieId: number) =>
  await movieDB.getNumberOfMovieReviews(movieId);

export const getMovieForUser = async (userId: number, movieId: number) =>
  await movieDB.getMovieForUser(userId, movieId);

export const getMostWatchedGenres = async (userId: number) => {
  const genres = await getMostGenresRatings(userId);
  return genres.map((a) => a[0]).slice(0, 3);
};

export const getMostGenresRatings = async (userId: number) => {
  const genres = await movieDB.getMostWatchedGenres(userId);
  return genres;
};

export const searchMovies = async (q: string, limit: number) => {
  const genres = await movieDB.searchMovies(q, limit);
  return genres;
};

export const reviewMovie = actionWrapWithError()(async (
  movieId: number,
  review: string
) => {
  const { userId } = await getUserId();
  await movieDB.reviewMovie(userId, movieId, review);
});
