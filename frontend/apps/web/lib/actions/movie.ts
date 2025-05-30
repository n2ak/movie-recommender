"use server";

import * as Backend from "@repo/backend_api";
import { movieDB, MovieGenre, reviewsDB } from "@repo/database";
import { parseRating } from "../validation";
import { cachedQuery, clearCacheKey } from "./redisClient";
import { getUserId, timedAction } from "./utils";

export type MovieWithPredictedRating = MovieWithUserRating & {
  predictedRating: number;
};

async function handleBackendResponse(
  response: Backend.BackendResponse,
  userId: number
) {
  const predictions = response.result.map((p) => p.predicted_rating);
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

export const editMovieReviewAndRating = timedAction(
  "editMovieReviewAndRating",
  async (p: {
    movieId: number;
    review_provided: boolean;
    userId: number;
    rating: number;
    text: string;
    title: string;
  }) => {
    parseRating(p);
    await reviewsDB.editMovieReviewAndRating(p);
    await clearCacheKey(`movie:${p.userId}:${p.movieId}`, "rateMovie");
  }
);

export const getRecommendedMoviesForUser = timedAction(
  "getRecommendedMoviesForUser",

  async ({ count, userId }: { count: number; userId: number }) =>
    handleBackendResponse(
      await Backend.getMoviesRecom({ userId, count }),
      userId
    )
);

export const getRecommendedGenreMovies = timedAction(
  "getRecommendedGenreMovies",
  async ({ genre, userId }: { genre: MovieGenre; userId: number }) =>
    handleBackendResponse(
      await Backend.getGenreRecom(userId, [genre], null),
      userId
    )
);

export const getSimilarMovies = timedAction(
  "getRecommendedGenreMovies",
  async ({ movieIds, count }: { movieIds: number[]; count: number }) => {
    const userId = await getUserId();
    return handleBackendResponse(
      await Backend.getSimilarMovies(userId, movieIds, count),
      userId
    );
  }
);

export const getRatedMoviesForUser = timedAction(
  "getRatedMoviesForUser",
  movieDB.getRatedMoviesForUser
);

export const getMovieReviews = timedAction(
  "getMovieReviews",
  reviewsDB.getMovieReviews
);

export const getNumberOfRatings = timedAction(
  "getNumberOfRatings",
  movieDB.getNumberOfRatings
);

export const reactToMovieReview = timedAction(
  "reactToMovieReview",
  reviewsDB.reactToMovieReview
);

export const getNumberOfMovieReviews = timedAction(
  "getNumberOfMovieReviews",
  reviewsDB.getNumberOfMovieReviews
);

const getMovieForUser = timedAction(
  "getMovieForUser",
  cachedQuery(
    movieDB.getMovieForUser,
    (params) => `movie:${params.userId}:${params.movieId}`
  )
);

export const getMostWatchedGenres = timedAction(
  "getMostWatchedGenres",
  movieDB.getMostWatchedGenres
);

// export const getMostGenresRatings = ttimed(
//   "getMostGenresRatings",
//   reviewsDB.getMostGenresRatings
// );

export const getMovieReview = timedAction(
  "getMovieReview",
  reviewsDB.getMovieReview
);

export const getMovieReviewById = timedAction(
  "getMovieReviewById",
  reviewsDB.getMovieReviewById
);

export const searchMovies = timedAction("searchMovies", movieDB.searchMovies);

export const reviewMovie = timedAction("reviewMovie", reviewsDB.reviewMovie); //TODO clear key

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type UnwrapReturn<T extends (...args: any) => any> = NonNullable<
  Awaited<ReturnType<T>>["data"]
>;

export { getMovieForUser };

export type MovieWithUserRating = UnwrapReturn<typeof getMovieForUser>;

export type MovieReview = NonNullable<
  Awaited<ReturnType<typeof getMovieReview>>["data"]
>;
