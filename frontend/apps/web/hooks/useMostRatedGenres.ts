import { getMostGenresRatings } from "@/_lib/actions/movie";
import { useQuery } from "@tanstack/react-query";

export default function useMostRatedGenres(userId: number) {
  const { data: genres } = useQuery({
    initialData: [],
    queryKey: [
      "user_most_rated_genres",
      {
        userId,
      },
    ],
    queryFn: () => getMostGenresRatings(userId),
  });
  return genres;
}
