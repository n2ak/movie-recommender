import { getMovieForUser, getSimilarMovies } from "@/lib/actions/movie";
import { redirect } from "next/navigation";
import { MoviePage } from "./movie";

export default async function Page({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const movieId = parseInt((await params).id);
  if (isNaN(movieId)) return <div>Invalid id</div>;

  const [movie, similarMovies] = await Promise.all([
    getMovieForUser({ movieId }),
    getSimilarMovies({
      movieIds: [movieId],
      count: 10,
    }),
  ]);
  if (
    !movie.data ||
    movie.message ||
    !similarMovies.data ||
    similarMovies.message
  ) {
    // TODO
    return redirect("/home");
  }
  return <MoviePage movie={movie.data} similarMovies={similarMovies.data} />;
}
