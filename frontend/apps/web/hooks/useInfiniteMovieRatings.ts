import { getNumberOfRatings, getRatedMoviesForUser } from "@/lib/actions/movie";
import type { RatingSortKey, SortOrder } from "@repo/database";
import { usePaging } from "./usePaging";
// import usePaging from "./usePaging";

export default function useInfiniteMovieRatings(
  userId: number,
  pageNumber: number,
  count: number,
  sortKey: RatingSortKey,
  sortOrder: SortOrder
) {
  return usePaging({
    fetchPage: async (start, count) =>
      (
        await getRatedMoviesForUser({
          start,
          count,
          sortby: sortKey,
          order: sortOrder,
        })
      ).data!,
    nRecordsFn: async () => (await getNumberOfRatings(userId)).data || 0,
    pageNumber,
    rowsPerPage: count,
    queryKey: "movies_ratings",
    nRecordsQKey: ["movies_nratings", userId],
    keys: { userId, sortKey, sortOrder },
  });
}
