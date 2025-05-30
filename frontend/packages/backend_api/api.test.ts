import { describe, expect, test } from "@jest/globals";
import { MovieGenre } from "@repo/database";
import { BackendResponse, getGenreRecom, getMoviesRecom } from "./src/index";

const MAX_RATING = 10;

describe("Backend Api", () => {
  function checkPred(userId: number, predictions: BackendResponse) {
    console.log({ predictions });
    expect(predictions.status_code).toBe(200);
    predictions.result.forEach((pred) => {
      expect(pred.userId).toBe(userId);
      expect(pred.predicted_rating).toBeGreaterThanOrEqual(0);
      expect(pred.predicted_rating).toBeLessThanOrEqual(MAX_RATING);
    });
  }
  test("getMoviesRecom", async () => {
    const userId = 1;
    const predictions: BackendResponse = await getMoviesRecom(userId, 10);
    checkPred(userId, predictions);
  });
  test("getGenreRecom", async () => {
    const userId = 1;
    const genres: MovieGenre[] = ["Action", "Comedy"];
    const predictions: BackendResponse = await getGenreRecom(
      userId,
      genres,
      10
    );
    checkPred(userId, predictions);
  });
});
