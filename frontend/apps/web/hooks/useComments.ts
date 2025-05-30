//   getNumberOfReviewComments,
//   getReviewComments,
// } from "@/lib/actions/movie";
// import type { SortOrder } from "@repo/database";
// import useInfinitePaging from "./usePaging";

// export default function useComments(
//   userId: number,
//   reviewId: number,
//   sortKey: any,
//   sortOrder: SortOrder
// ) {
//   return useInfinitePaging(
//     sortKey,
//     sortOrder,
//     (s, c) => getReviewComments(userId, reviewId, s, c, sortKey, sortOrder),
//     () => getNumberOfReviewComments(reviewId),
//     "review_comments",
//     ["nreview_comments", reviewId],
//     { reviewId }
//   );
// }
