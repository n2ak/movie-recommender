import { MovieGenre } from "@repo/database";
import { recommendMovies, recommendSimilarMovies } from "./client";

export const getMoviesRecom = async (params: {
  userId: number;
  count: number | null;
}) => {
  return recommendMovies({
    userId: params.userId,
    count: params.count,
    start: 0,
    genres: [],
    relation: "and",
  });
};
export const getGenreRecom = async (
  userId: number,
  genres: MovieGenre[],
  count: number | null
) => {
  return recommendMovies({
    userId,
    count,
    start: 0,
    genres,
    relation: "and",
  });
};
export const getSimilarMovies = async (
  userId: number,
  movieIds: number[],
  count: number | null
) => {
  return recommendSimilarMovies({
    userId,
    start: 0,
    count,
    movieIds,
  });
};
export { type BackendResponse } from "./client";
