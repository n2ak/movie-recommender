"use client";

import { likeMovieReview, MovieReview } from "@/_lib/actions/movie";
import { formatNumber, joinCN } from "@/_lib/utils";
import { FixedRating } from "@/components/Rating";
import { useAuthStore } from "@/hooks/useAuthStore";
import useMovie from "@/hooks/useMovie";
import { useMovieReviews } from "@/hooks/useMovieReviews";
import { Heart } from "lucide-react";
import { useParams } from "next/navigation";
import { useState } from "react";
import { MovieWithUserRating } from "../../../../../packages/database/src/movieDb";

export default function ReviewsPage() {
  const user = useAuthStore((s) => s.user);
  const { id } = useParams();
  const movieId = parseInt(id as string);

  const { movie, isLoading } = useMovie({
    movieId,
    userId: user?.id,
  });
  const { data: movie_reviews } = useMovieReviews(
    user?.id as number,
    movieId,
    "",
    "asc"
  );
  if (isLoading) {
    return null;
  }
  if (!movie) {
    // TODO
    return null;
  }
  return (
    <>
      {movie?.title}
      {movie_reviews.map((rev) => {
        return (
          <div key={rev.id}>
            <Review movie={movie} review={rev} userId={user?.id} />
          </div>
        );
      })}
    </>
  );
}

function Review({
  movie,
  review,
  userId,
}: {
  userId?: number;
  movie: MovieWithUserRating;
  review: MovieReview;
}) {
  const [showP, setShowP] = useState(false);
  const userRating = review.user.movieRatings[0]?.rating;
  const liked = !!review.likes[0];
  console.log("xdddddd");

  return (
    <div className="font-sans text-base leading-relaxed bg-gray-100/90 px-5 py-3 shadow-all-sides rounded-md ">
      <div className="flex gap-3">
        <span>By:</span>
        <span className="underline">{review.user.username}</span>
        {userRating && (
          <span className="ml-auto">
            <FixedRating v={userRating} showValue />
          </span>
        )}
      </div>
      <div className={joinCN("my-3", showP ? "" : "line-clamp-2")}>
        {review.text}
      </div>
      <div className="text-right">
        <span
          className="text-xs underline cursor-pointer hover:text-blue-800"
          onClick={() => setShowP((s) => !s)}
        >
          {showP ? "show less" : "show more"}
        </span>
      </div>
      <div className="flex justify-end gap-3">
        {userId && (
          <span
            className="mr-auto cursor-pointer hover:scale-[105%]"
            onClick={() => likeMovieReview(userId, review.id)}
          >
            {liked ? <Heart color="red" fill="red" /> : <Heart />}
          </span>
        )}
        <div>Likes: {formatNumber(review._count.likes)}</div>
        <div>Comments: {formatNumber(review._count.comments)}</div>
      </div>
    </div>
  );
}
