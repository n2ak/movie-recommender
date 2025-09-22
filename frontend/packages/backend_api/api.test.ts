import { describe, expect, test } from "@jest/globals";
import { MAX_RATING, movieDB, type MovieGenre } from "@repo/database";
import { type BackendResponse, getMoviesRecom, getSimilarMovies } from "./src/index";


describe("Backend Api", () => {
  function checkPred(userId: number, predictions: BackendResponse, count: number) {
    // console.log({ predictions });
    expect(predictions.status_code).toBe(200);
    expect(predictions.result).toHaveLength(count);
    predictions.result.forEach((pred) => {
      expect(pred.userId).toBe(userId);
      expect(pred.predicted_rating).toBeGreaterThanOrEqual(0);
      expect(pred.predicted_rating).toBeLessThanOrEqual(MAX_RATING);
    });
  }

  test("movies-recom", async () => {
    const userId = 1;
    const count = 10;
    const predictions = await getMoviesRecom({
      userId,
      count,
      temp: 0,
    });
    checkPred(userId, predictions, count);
  });

  test("similar-movies", async () => {
    const userId = 1;
    const count = 10;
    const movieIds = [10, 12];
    const predictions = await getSimilarMovies({
      userId,
      count,
      movieIds,
      temp: 0,
    });
    checkPred(userId, predictions, count);
  });

  test("getGenreRecom", async () => {
    const userId = 1, count = 10;
    const genres: MovieGenre[] = ["Action", "Comedy"];
    const predictions = await getMoviesRecom({
      userId,
      genres,
      count,
      temp: 0
    }
    );
    checkPred(userId, predictions, count);
    const movieIds = predictions.result.map(p => p.movieId);
    const movies = await movieDB.getMoviesGenres(movieIds);
    movies.forEach((movie) => {
      const condition = genres.some(g => movie.genres.includes(g));
      expect(condition).toBe(true);
    })
  });

});
