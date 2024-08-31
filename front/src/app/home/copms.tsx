// "use client";
import { MovieRow } from "@/components/MovieCard";
import { useSnackBar } from "@/components/SnackBarProvider";
import { getRecommendedMoviesForUser } from "@/lib/actions/action";

export async function Recommended({ userId }: { userId: number }) {
  //   const snackBar = useSnackBar();
  const movies = await getRecommendedMoviesForUser(userId, 0, 10).catch(() => {
    return null;
  });
  if (movies === null) return <>Nothing here</>;
  return <MovieRow title="Movies you would like to watch:" movies={movies} />;
}
export async function ContinueWatching({ userId }: { userId: number }) {
  const movies = await getRecommendedMoviesForUser(userId, 10, 10);
  return <MovieRow title="Continue watching:" movies={movies} />;
}
