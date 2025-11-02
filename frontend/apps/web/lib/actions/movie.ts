"use server";

import * as Backend from "@repo/backend_api";
import { movieDB, reviewsDB, type MovieGenre } from "@repo/database";
import { parseRating, parseReview } from "../validation";
import { cachedQuery, clearCacheKey } from "./redisClient";
import { action, getUserId } from "./utils";

export type MovieWithPredictedRating = MovieWithUserRating & {
  predictedRating: number;
};

async function handleBackendResponse(
  response: Backend.BackendResponse,
  userId: number
) {
  const predictions = response.result.map((p) => p.predictedRating);
  const movies = await movieDB.getMovies(
    response.result.map((p) => p.movieId),
    userId
  );
  const moviesWithPred = movies.map((m, idx) => ({
    ...m,
    predictedRating: predictions[idx]!,
  }));
  return moviesWithPred;
}

export const editMovieReviewAndRating = action(
  "editMovieReviewAndRating",
  async (p: {
    movieId: number;
    reviewChanged: boolean;
    userId: number;
    rating: number;
    text: string;
    title: string;
  }) => {
    parseRating({ rating: p.rating });
    if (p.reviewChanged) parseReview({ text: p.text, title: p.title });

    await reviewsDB.editMovieReviewAndRating(p);
    await clearCacheKey(`movie:${p.userId}:${p.movieId}`, "rateMovie");
  }
);

export const getRecommendedMoviesForUser = action(
  "getRecommendedMoviesForUser",

  async ({ count, userId, temp }: { count: number; userId: number, temp: number }) => {
    const userBestMovies = await movieDB.getUserBestMovies({ userId });
    const userBestMovieIds = userBestMovies.map(m => m.tmdbId);
    return handleBackendResponse(
      await Backend.recommendMovies({ userId, count, temp, userBestMovies: userBestMovieIds }),
      userId
    )
  }
);

export const getRecommendedGenreMovies = action(
  "getRecommendedGenreMovies",
  async ({ genre, userId, temp }: { genre: MovieGenre; userId: number, temp: number }) => {
    const userBestMovies = await movieDB.getUserBestMovies({ userId });
    const userBestMovieIds = userBestMovies.map(m => m.tmdbId);
    return handleBackendResponse(
      await Backend.recommendMovies({ userId, count: 10, genres: [genre], temp, userBestMovies: userBestMovieIds }),
      userId
    )
  }
);

export const getSimilarMovies = action(
  "getRecommendedGenreMovies",
  async ({ movieIds, count }: { movieIds: number[]; count: number }) => {
    const userId = await getUserId();
    return handleBackendResponse(
      await Backend.recommendSimilarMovies({ userId, movieIds, count, temp: 0 }),
      userId
    );
  }
);

export const getRatedMoviesForUser = action(
  "getRatedMoviesForUser",
  movieDB.getRatedMoviesForUser
);

export const getMovieReviews = action(
  "getMovieReviews",
  reviewsDB.getMovieReviews
);

export const getNumberOfRatings = action(
  "getNumberOfRatings",
  movieDB.getNumberOfRatings
);

export const reactToMovieReview = action(
  "reactToMovieReview",
  reviewsDB.reactToMovieReview
);

const getMovieForUser = action(
  "getMovieForUser",
  cachedQuery(
    movieDB.getMovieForUser,
    (params) => `movie:${params.userId}:${params.movieId}`
  )
);

export const getMostWatchedGenres = action(
  "getMostWatchedGenres",
  movieDB.getMostWatchedGenres
);

export const getMovieReview = action(
  "getMovieReview",
  reviewsDB.getMovieReview
);

export const getMovieReviewById = action(
  "getMovieReviewById",
  reviewsDB.getMovieReviewById
);

export const searchMovies = action("searchMovies", movieDB.searchMovies);

export const reviewMovie = action("reviewMovie", reviewsDB.reviewMovie); //TODO clear key

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type UnwrapReturn<T extends (...args: any) => any> = NonNullable<
  Awaited<ReturnType<T>>["data"]
>;

export { getMovieForUser };

export type MovieWithUserRating = UnwrapReturn<typeof getMovieForUser>;

export type MovieReview = NonNullable<
  Awaited<ReturnType<typeof getMovieReview>>["data"]
>;
