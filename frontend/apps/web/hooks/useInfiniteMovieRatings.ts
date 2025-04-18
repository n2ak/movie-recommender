import { getRatedMoviesForUser } from "@/_lib/actions/movie";
import type { RatingSortKey, SortOrder } from "@repo/database";
import useNumberOfRatings from "./useNumberOfRatings";
import usePaging from "./usePaging";

export default function useInfiniteMovieRatings(
  userId: number,
  sortKey: RatingSortKey,
  sortOrder: SortOrder
) {
  const nratings = useNumberOfRatings(userId);

  const {
    data,
    setPageNumber,
    setRowsPerPage,
    rowsPerPage,
    pageNumber,
    queryKey,
    nPages,
  } = usePaging({
    fetchPage: async (start, count) => {
      const res = await getRatedMoviesForUser(
        userId,
        start,
        count,
        sortKey,
        sortOrder
      );
      return res;
    },
    queryKey: "movies_ratings",
    keys: {
      userId,
      sortKey,
      sortOrder,
    },
    nRecords: nratings,
  });
  return {
    movies: data,
    setPageNumber,
    setRowsPerPage,
    rowsPerPage,
    pageNumber,
    queryKey,
    nPages,
  };
}
