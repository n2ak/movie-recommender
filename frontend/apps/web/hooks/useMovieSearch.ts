import { searchMovies } from "@/_lib/actions/movie";
import { useQuery } from "@tanstack/react-query";

export default function useMovieSearch(search: string) {
  const { data: movies } = useQuery({
    initialData: [],
    queryKey: ["movie_search", { search }],
    queryFn: () => searchMovies(search, 5),
  });
  return movies;
}
