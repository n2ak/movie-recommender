import type { AxiosResponse } from "axios";
import axios, { AxiosError } from "axios";
import type { BackendRequest, BackendResponse, SimilarMoviesRequest, SinglePrediction } from "./types";

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

const handleErrors = async (
  promise: Promise<AxiosResponse>
): Promise<BackendResponse> => {
  const resp: BackendResponse = {
    result: [],
  }
  try {
    resp.result = (await promise).data as SinglePrediction[];
    resp.statusCode = 200

  } catch (e) {
    let error = "Error calling backend service";
    // let code: number | string = 404;
    if (e instanceof AxiosError) {
      if (e.code === "ECONNREFUSED") {
        error = "Backend is down";
        resp.statusCode = e.status || 404;
      }
      console.error({ error: e.message });
    }
    resp.error = error;
  }
  return resp;
};


export const recommendMovies = async ({
  userId, temp, count, genres
}: Omit<BackendRequest, "type">) => {
  const data: Required<BackendRequest> = {
    userId,
    type: "recommend",
    temp: temp || 0,
    count: count || 10,
    genres: genres || [],
  };
  const result = await handleErrors(
    apiClient.post("/predict", data)
  );
  return result;
};

export const recommendSimilarMovies = async (req: SimilarMoviesRequest) => {
  const data: Required<SimilarMoviesRequest> = {
    userId: req.userId,
    count: req.count,
    model: req.model || "xgb_cuda",
    start: req.start || 0,
    temp: req.temp || 0,
    movieIds: req.movieIds
  };
  const result = await handleErrors(
    apiClient.post("/predict", data)
  );
  return result;
};
