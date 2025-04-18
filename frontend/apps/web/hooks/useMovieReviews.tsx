import { getMovieReviews, getNumberOfMovieReviews } from "@/_lib/actions/movie";
import type { SortOrder } from "@repo/database";
import useInfinitePaging from "./usePaging";

export function useMovieReviews(
  userId: number,
  movieId: number,
  sortKey: any,
  sortOrder: SortOrder
) {
  return useInfinitePaging(
    sortKey,
    sortOrder,
    (s, c) => getMovieReviews(userId, movieId, s, c, sortKey, sortOrder),
    () => getNumberOfMovieReviews(movieId),
    "movie_reviews",
    ["nmovie_reviews", movieId],
    { movieId }
  );
}
