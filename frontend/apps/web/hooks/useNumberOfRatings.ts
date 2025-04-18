import { getNumberOfRatings } from "@/_lib/actions/movie";
import { useQuery } from "@tanstack/react-query";

export default function useNumberOfRatings(userId: number) {
  const { data: nratings } = useQuery({
    queryKey: [
      "movies_nratings",
      {
        userId,
      },
    ],
    initialData: -1,
    queryFn: () => getNumberOfRatings(userId),
  });
  return nratings;
}
