"use client";
import { FixedRating } from "@/components/Rating";
import ToggleLongParagraph from "@/components/ToggleLongParagraph";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { useAuthStore } from "@/hooks/useAuthStore";
import type { MovieReview, MovieWithUserRating } from "@/lib/actions/movie";
import {
  getMovieReviewById,
  getMovieReviews,
  reactToMovieReview,
} from "@/lib/actions/movie";
import { formatNumber, timeSince } from "@/lib/utils";
import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { ThumbsDownIcon, ThumbsUpIcon } from "lucide-react";
import { useParams } from "next/navigation";

export default function MovieReviews({
  movie,
}: {
  movie: MovieWithUserRating;
}) {
  const user = useAuthStore((s) => s.user);
  const { id } = useParams();
  const movieId = parseInt(id as string);

  const count = 10;
  const { data, fetchNextPage, hasNextPage, isFetching, isFetchingNextPage } =
    useInfiniteQuery({
      queryKey: ["movie_reviews", { movieId }],
      queryFn: ({ pageParam: start }) =>
        getMovieReviews({ movieId, start, count, order: "asc" }),
      initialPageParam: 0,
      getNextPageParam: (lastPage, pages) => {
        const hasNext = pages[pages.length - 1]?.data?.length === count;
        if (hasNext) return pages.length * count;
        return null;
      },
      enabled: !!user,
    });
  if (!movie || !data) {
    // TODO
    return null;
  }
  const reviews = data.pages.flatMap((g) => g.data).filter((r) => !!r);
  console.log(reviews.length);
  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-4xl font-bold text-center mb-10 text-gray-800">
        🎬 Movie Reviews
      </h1>
      <div className="grid gap-6">
        {reviews.map((review) => (
          <Review review={review} movieId={movie.id} key={review.id} />
        ))}
      </div>
      <div className="w-full text-center mt-2">
        <Button
          onClick={() => fetchNextPage()}
          disabled={!hasNextPage || isFetching}
        >
          {isFetchingNextPage
            ? "Loading more..."
            : hasNextPage
              ? "Load More"
              : "Nothing more to load"}
        </Button>
      </div>
    </div>
  );
}

function Review({
  movieId,
  review: r,
}: {
  review: MovieReview;
  movieId: number;
}) {
  const keys = ["movie_review", { review_id: r.id }];
  const { data: review } = useQuery({
    queryKey: keys,
    queryFn: async () =>
      (await getMovieReviewById({ reviewId: r.id, movieId })).data,
    initialData: r,
    refetchOnMount: false,
  });
  const qL = useQueryClient();
  const react = useMutation({
    mutationFn: async (type: "LIKE" | "DISLIKE") =>
      reactToMovieReview({ movieReviewId: review!.id, type }),
    onSuccess: () => {
      qL.invalidateQueries({ queryKey: keys });
    },
    onError: () => {},
  });
  if (!review) return null;
  const userRating = review.user.movieRatings[0]?.rating;
  const isLiked =
    review.reactions.length > 0 && review.reactions[0]?.type === "LIKE";
  const isDisLiked =
    review.reactions.length > 0 && review.reactions[0]?.type === "DISLIKE";
  const nLikes = formatNumber(review.nlikes);
  const nDislikes = formatNumber(review.ndislikes);
  const date = timeSince(review.createdAt);
  return (
    <Card className="shadow-xl rounded-2xl p-6 transition-transform hover:scale-[1.01]">
      <CardContent>
        <h2 className="text-2xl font-semibold">
          {review.title}{" "}
          <span className="text-sm text-gray-500">(posted {date} ago)</span>
        </h2>
        {userRating && (
          <div className="flex items-center mt-2">
            <FixedRating v={userRating} />
          </div>
        )}
        <div className="mt-4 italic">
          <ToggleLongParagraph text={review.text} />
        </div>
        <p className="mt-2 text-right italic text-sm text-gray-500">
          -by {review.user.username}
        </p>
      </CardContent>
      <CardFooter className="flex flex-col">
        <div className="flex justify-end gap-1 w-full">
          <Button
            variant={isLiked ? "default" : "outline"}
            onClick={async () => react.mutate("LIKE")}
            // disabled={!userId}
          >
            {nLikes}
            <ThumbsUpIcon />
          </Button>
          <Button
            variant={isDisLiked ? "default" : "outline"}
            onClick={async () => react.mutate("DISLIKE")}
            // disabled={!userId}
          >
            {nDislikes}
            <ThumbsDownIcon />
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
}
