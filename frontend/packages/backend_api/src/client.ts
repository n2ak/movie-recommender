import axios, { AxiosError, AxiosResponse } from "axios";

const apiClient = axios.create({
  baseURL: process.env.BACKEND_URL || "http://127.0.0.1:8000",
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      console.error("API Error:", error.response.status, error.response.data);
    } else if (error.request) {
      console.error("Network error or timeout:", error.message);
    } else {
      console.error("Unexpected error:", error.message);
    }
    return Promise.reject(error);
  }
);

const handleErrors = async <P>(
  promise: Promise<AxiosResponse>
): Promise<BackendResponse<P>> => {
  try {
    const a = (await promise).data;
    return a;
  } catch (e) {
    let error = "Error";
    let code: number | string = 404;
    if (e instanceof AxiosError) {
      if (e.code === "ECONNREFUSED") {
        error = "Backend is down";
      }
      code = e.status || 404;
      console.log({ error: e.message });
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
};

type Relation = "or" | "and";
type Prediction = {
  movieId: number;
  userId: number;
  predicted_rating: number;
};
export type BackendResponse<P = Prediction> = {
  time: number;
  result: P[];
  status_code: number | string;
  error:
    | {
        [index: string]: string;
      }
    | undefined;
};

export const recommendMovies = async (data: {
  userId: number;
  genres: string[];
  start: number;
  relation: Relation;
  count: number | null;
}) => {
  const result = await handleErrors<Prediction>(
    apiClient.post("/movies-recom", data)
  );
  return result;
};
export const recommendSimilarMovies = async (data: {
  userId: number;
  movieIds: number[];
  start: number;
  count: number | null;
}) => {
  const result = await handleErrors<Prediction>(
    apiClient.post("/recom-similar-movies", data)
  );
  return result;
};
