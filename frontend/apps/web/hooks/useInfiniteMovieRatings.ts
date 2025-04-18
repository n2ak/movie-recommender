import {
  getNumberOfRatings,
  getRatedMoviesForUser,
} from "@/_lib/actions/movie";
import type { RatingSortKey, SortOrder } from "@repo/database";
// import usePaging from "./usePaging";
import useInfinitePaging from "./usePaging";

export default function useInfiniteMovieRatings(
  userId: number,
  sortKey: RatingSortKey,
  sortOrder: SortOrder
) {
  return useInfinitePaging(
    sortKey,
    sortOrder,
    (s, c) => getRatedMoviesForUser(userId, s, c, sortKey, sortOrder),
    () => getNumberOfRatings(userId),
    "movies_ratings",
    ["movies_nratings", userId],
    { userId }
  );
}
