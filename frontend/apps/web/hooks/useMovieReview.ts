import { getMovieReview, reviewMovie } from "@/lib/actions/movie";
import { useQuery, useQueryClient } from "@tanstack/react-query";

export default function useMovieReview(
  movieId: number | undefined,
  onSettle?: (res: Awaited<ReturnType<typeof reviewMovie>>) => void
) {
  // https://tkdodo.eu/blog/seeding-the-query-cache#pull-approach
  const queryKey = ["movie_review", { movieId }];
  const { data: review, isLoading } = useQuery({
    queryKey: queryKey,
    queryFn: async () => (await getMovieReview({ movieId: movieId! })).data,
    enabled: !!movieId,
  });
  const qL = useQueryClient();
  const doReview = async (title: string, text: string) => {
    const res = await reviewMovie({ movieId: movieId!, title, text });

    if (onSettle) onSettle(res);
    if (res) {
      await qL.invalidateQueries({ queryKey });
    }
  };
  return {
    review,
    isLoading,
    doReview,
  };
}
