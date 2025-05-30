import { searchMovies } from "@/lib/actions/movie";
import { useQuery } from "@tanstack/react-query";

export default function useMovieSearch(search: string) {
  const { data: movies } = useQuery({
    initialData: [],
    queryKey: ["movie_search", { search }],
    queryFn: () =>
      searchMovies({ i: search, limit: 5 }).then((r) => r.data ?? []),
  });
  return movies;
}
