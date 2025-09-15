import { recommendMovies, recommendSimilarMovies } from "./client";

export const getMoviesRecom = recommendMovies;
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
export { type BackendResponse } from "./types";

