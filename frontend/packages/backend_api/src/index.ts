import { MovieGenre } from "@repo/database";
import { AxiosError } from "axios";
import { movies_recom } from "./client";

export type ModelType = "DLRM" | "NCF";

export const Backend = {
  getMoviesRecom: async function (
    userId: number,
    model: ModelType,
    count: number | null
  ) {
    return this.request({
      userId,
      model,
      count,
      start: 0,
      genres: [],
      movieIds: [],
      relation: "and",
    });
  },
  getMovieRatings: async function (
    userId: number,
    model: ModelType,
    movieIds: number[],
    count: number | null
  ) {
    return this.request({
      userId,
      model,
      count,
      start: 0,
      genres: [],
      movieIds,
      relation: "and",
    });
  },
  getGenreRecom: async function (
    userId: number,
    model: ModelType,
    genres: MovieGenre[],
    count: number | null
  ) {
    return this.request({
      userId,
      model,
      count,
      start: 0,
      genres,
      movieIds: [],
      relation: "and",
    });
  },
  request: async (
    data: GetMoviesRecomRequest
  ): Promise<GetMoviesRecomResponse> => {
    try {
      const a = (await movies_recom(data)).data;
      return a;
    } catch (e) {
      let error = "Error";
      let code: number | string = 404;
      if (e instanceof AxiosError) {
        error = e.cause?.message || "";
        code = e.code || 404;
      }
      return {
        status_code: code,
        result: [],
        time: 0,
        error: {
          error,
        },
      };
    }
  },
};

export interface GetMoviesRecomResponse {
  time: number;
  result: {
    movieId: number;
    userId: number;
    predicted_rating: number;
  }[];
  status_code: number | string;
  error:
    | {
        [index: string]: string;
      }
    | undefined;
}

type Relation = "or" | "and";

interface GetMoviesRecomRequest {
  userId: number;
  movieIds: number[];
  genres: string[];
  model: ModelType;
  start: number;
  relation: Relation;
  count: number | null;
}
