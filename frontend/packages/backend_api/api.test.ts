import { describe, expect, test } from "@jest/globals";
import { MovieGenre } from "@repo/database";
import { Backend } from "./src/index";

describe("Backend Api", () => {
  test("getMoviesRecom", async () => {
    const userId = 1;
    const predictions = await Backend.getMoviesRecom(userId, "DLRM", 10);
    predictions.result.forEach((pred) => {
      expect(pred.userId).toBe(userId);
      expect(pred.predicted_rating).toBeGreaterThanOrEqual(0);
      expect(pred.predicted_rating).toBeLessThanOrEqual(5);
    });
  });
  test("getMovieRatings", async () => {
    const userId = 1;
    const movieIds = [1, 2, 3];
    const predictions = await Backend.getMovieRatings(
      userId,
      "DLRM",
      movieIds,
      10
    );
    predictions.result.forEach((pred) => {
      expect(pred.userId).toBe(userId);
      expect(movieIds).toContain(pred.movieId);
      expect(pred.predicted_rating).toBeGreaterThanOrEqual(0);
      expect(pred.predicted_rating).toBeLessThanOrEqual(5);
    });
  });
  test("getGenreRecom", async () => {
    const userId = 1;
    const genres: MovieGenre[] = ["Action", "Comedy"];
    const predictions = await Backend.getGenreRecom(userId, "DLRM", genres, 10);
    predictions.result.forEach((pred) => {
      expect(pred.userId).toBe(userId);
      expect(pred.predicted_rating).toBeGreaterThanOrEqual(0);
      expect(pred.predicted_rating).toBeLessThanOrEqual(5);
    });
  });
});
