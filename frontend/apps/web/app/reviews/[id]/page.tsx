import { getMovieForUser } from "@/lib/actions/movie";
import { redirect } from "next/navigation";
import MovieReviews from "./reviews";

export default async function ReviewsPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const movieId = parseInt((await params).id);
  if (isNaN(movieId)) return <div>Invalid id</div>;

  const movie = await getMovieForUser({ movieId });
  if (!movie.data) {
    // TODO
    return redirect("/home");
  }

  return <MovieReviews movie={movie.data} />;
}
