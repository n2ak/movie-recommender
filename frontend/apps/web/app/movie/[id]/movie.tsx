"use client";
import { MovieCarousel } from "@/components/MovieCarousel";
import EditMovieRatingAndReviewModal from "@/components/RateMovieModal";
import { FixedRating } from "@/components/Rating";
import useMovie from "@/hooks/useMovie";
import {
  MovieWithPredictedRating,
  MovieWithUserRating,
} from "@/lib/actions/movie";
import { formatNumber } from "@/lib/utils";
import { useQueryClient } from "@tanstack/react-query";
import { EditIcon } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export function MoviePage({
  movie: initialMovie,
  similarMovies,
}: {
  movie: MovieWithUserRating;
  similarMovies: MovieWithPredictedRating[];
}) {
  const qClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const { movie, queryKey } = useMovie({
    movieId: initialMovie.id,
    initialMovie,
  });
  if (!movie) return null;

  const total_ratings = formatNumber(movie.total_ratings);
  const total_reviews = formatNumber(movie._count.reviews);
  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Movie Info */}
      <div className="flex flex-col md:flex-row gap-6">
        <img
          src={movie.href}
          alt={movie.title}
          className="w-64 rounded-lg shadow-lg"
        />
        <div className="flex-1 space-y-4">
          <h1 className="text-4xl font-bold">{movie.title}</h1>
          <p className="text-gray-500 text-lg">
            {movie.year} • {movie.genres.join(", ")}
          </p>
          <div className="font-semibold flex gap-2">
            <FixedRating v={movie.avg_rating} />
            <span>({total_ratings} users)</span>
            <Link className="underline" href={"/reviews/" + movie.id}>
              ({total_reviews} reviews)
            </Link>
          </div>
          <div className="font-semibold flex gap-2">
            {movie.userRating[0] ? "Personal Rating:" : "Unrated"}
            <FixedRating v={movie.userRating[0]?.rating}></FixedRating>
            <EditIcon
              className="rounded-xs cursor-pointer hover:bg-black/50 hover:text-white hover:scale-105"
              onClick={() => setOpen((o) => !o)}
            />
          </div>

          <EditMovieRatingAndReviewModal
            onClose={() => setOpen(false)}
            onSave={() => {
              setOpen(false);
              qClient.invalidateQueries({
                queryKey,
              });
            }}
            movie={!open ? undefined : movie}
          />
          <p className="">{movie.desc}</p>
        </div>
      </div>

      <div className="mt-12">
        <h2 className="text-2xl font-semibold mb-4">Similar Movies</h2>
        <MovieCarousel movies={similarMovies} />
      </div>
    </div>
  );
}
