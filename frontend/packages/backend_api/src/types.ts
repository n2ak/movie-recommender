import type { MovieGenre } from "@repo/database";

export type AvailableModels = "xgb_cpu" | "dlrm_cpu" | "xgb_cuda" | "dlrm_cuda" | "best";

export type BaseRequest = {
    userId: number
    type: "recommend" | "similar"
    temp?: number,
    count?: number,
}
export interface RecomRequest extends BaseRequest {
    genres?: MovieGenre[]
}
export interface SimilarMoviesRequest extends BaseRequest {
    movieIds: number[]
}
export type SinglePrediction = {
    predictedRating: number,
    movieId: number,
    userId: number
};

export type BackendResponse = {
    result: SinglePrediction[];
    error?: string;
    statusCode?: number
};