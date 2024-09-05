// "use client";
import { MovieRow } from "@/components/MovieCard";
import { useSnackBar } from "@/components/SnackBarProvider";
import {
  getRecommendedGenreMovies,
  getRecommendedMoviesForUser,
} from "@/lib/actions/action";
import { Model } from "@/lib/backend_api";
import { MovieGenre } from "@/lib/db";

export async function Recommended({
  userId,
  model,
}: {
  userId: number;
  model: Model;
}) {
  //   const snackBar = useSnackBar();
  const movies = await getRecommendedMoviesForUser(userId, 0, 10);
  // console.log("Recommended", model, movies);
  return (
    <MovieRow
      title={`using '${model}',Movies you would like to watch:`}
      movies={movies.movies}
      predictions={movies.predictions}
    />
  );
}
export async function ContinueWatching({ userId }: { userId: number }) {
  const movies = await getRecommendedMoviesForUser(userId, 0, 10);
  // console.log("ContinueWatching", movies);
  return (
    <MovieRow
      title="Continue watching:"
      movies={movies.movies}
      predictions={movies.predictions}
    />
  );
}
export async function RecommendedGenre({
  userId,
  genres,
  model,
}: {
  userId: number;
  genres: MovieGenre[];
  model: Model;
}) {
  const movies = await getRecommendedGenreMovies(userId, 0, 10, genres, model);
  // console.log("RecommendedGenre", model, genres, movies);
  return (
    <MovieRow
      title={`Using '${model}' model,Recommanded '${genres.join(",")}' movies:`}
      movies={movies.movies}
      predictions={movies.predictions}
    />
  );
}
