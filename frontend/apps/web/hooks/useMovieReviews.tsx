import { getMovieReviews, getNumberOfMovieReviews } from "@/lib/actions/movie";
import type { SortOrder } from "@repo/database";
import { usePaging } from "./usePaging";

export function useMovieReviews(
  userId: number,
  movieId: number,
  sortKey: any,
  sortOrder: SortOrder
) {
  return usePaging({
    fetchPage: (start, count) =>
      getMovieReviews({
        movieId,
        start,
        count,
        sortKey,
        order: sortOrder,
      }).then((r) => r.data ?? []),
    nRecordsFn: () =>
      getNumberOfMovieReviews({ movieId }).then((r) => r.data ?? 0),
    queryKey: "movie_reviews",
    nRecordsQKey: ["nmovie_reviews", movieId],
    keys: { movieId, sortKey, sortOrder },
  });
}
