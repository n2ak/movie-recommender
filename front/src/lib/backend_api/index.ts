import axios from "axios";
import { MovieRating } from "../definitions";
import { MovieModel, UserMovieRating } from "@prisma/client";

const api = axios.create({
  baseURL: "http://localhost:3333",
});
export type Model = "MF" | "NCF";
const Backend = {
  getMoviesRecom: async function (
    userId: number,
    start: number,
    count: number,
    model?: Model
  ): Promise<{
    time: number;
    result: [{ movieIds: number[]; pred_ratings: number[] }];
  }> {
    return this.getMoviesRecomForMovies(userId, [], start, count, model);
  },
  getMoviesRecomForMovies: async function (
    userId: number,
    movieIds: number[],
    start: number,
    count: number,
    model?: Model,
    round?: boolean
  ): Promise<{
    time: number;
    result: [{ movieIds: number[]; pred_ratings: number[] }];
  }> {
    if (round === undefined) round = false;
    const movies = await api.post("/movies-recom", {
      userIds: [userId],
      movieIds,
      start,
      count,
      model,
      round,
    });
    return movies.data;
  },
  getMovies: async function (genres: string[]): Promise<MovieModel[]> {
    const movies = await api.get("/allmovies", {
      data: {
        genres,
      },
    });
    return movies.data;
  },
  // getMovieSuggestions: async function (
  //   ratings: UserMovieRating[]
  // ): Promise<Array<MovieModel>> {
  //   return this.getMoviesRecom([], ratings);
  // },
  // getMovieSuggestionsForGenre: async function (
  //   genre: string,
  //   ratings: UserMovieRating[]
  // ) {
  //   return this.getMoviesRecom([genre], ratings);
  // },
  train: async function () {
    return (await api.post("/train", {})).data;
  },
};

export default Backend;
