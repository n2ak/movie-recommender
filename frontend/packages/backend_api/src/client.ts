import axios, { AxiosError, AxiosResponse } from "axios";
import type { BackendRequest, BackendResponse, Prediction } from "./types";

const createClient = () => {
  const baseURL = process.env.BACKEND_URL || "http://127.0.0.1:8000"
  const apiClient = axios.create({
    baseURL: baseURL,
  });
  console.log("Backend base url:", baseURL);

  return apiClient;
}
declare const globalThis: {
  apiClient: ReturnType<typeof createClient>;
} & typeof global;
const apiClient = globalThis.apiClient ?? createClient();
if (process.env.NODE_ENV !== "production") globalThis.apiClient = apiClient;

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



export const recommendMovies = async ({
  userId, count, genres, model, start, temp
}: BackendRequest) => {
  const data: Required<BackendRequest> = {
    userId, count, genres: genres || [],
    model: model || "xgb_cuda", start: start || 0,
    temp: temp || 0
  };
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
