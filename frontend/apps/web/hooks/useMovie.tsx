import { getMovieForUser, MovieWithUserRating } from "@/lib/actions/movie";
import { useQuery } from "@tanstack/react-query";

export default function useMovie({
  movieId,
  initialMovie,
}: {
  movieId: number;
  initialMovie?: MovieWithUserRating;
}) {
  // https://tkdodo.eu/blog/seeding-the-query-cache#pull-approach
  const queryKey = ["movie", { movieId }];
  const { data: movie, isLoading } = useQuery({
    queryKey,
    queryFn: () =>
      getMovieForUser({ movieId }).then((r) => r.data ?? undefined),
    initialData: initialMovie,
    refetchOnMount: !initialMovie,
  });
  return {
    movie,
    queryKey,
    isLoading,
  };
}
