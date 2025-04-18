import { getMovieForUser } from "@/_lib/actions/movie";
import { useQuery } from "@tanstack/react-query";

export default function useMovie({
  userId,
  movieId,
  initialMovie,
}: {
  userId: number | undefined;
  movieId: number;
  initialMovie?: Awaited<ReturnType<typeof getMovieForUser>>;
}) {
  // https://tkdodo.eu/blog/seeding-the-query-cache#pull-approach
  const queryKey = ["movie", { userId, movieId }];
  const { data: movie } = useQuery({
    queryKey,
    queryFn: async () => {
      const movie = await getMovieForUser(userId!, movieId);
      return movie;
    },
    enabled: !!userId,
    initialData: initialMovie,
  });
  return {
    movie,
    queryKey,
  };
}
