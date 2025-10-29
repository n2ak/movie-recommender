import type { MovieGenre } from "@repo/database";

export type AvailableModels = "xgb_cpu" | "dlrm_cpu" | "xgb_cuda" | "dlrm_cuda" | "best";

export type BackendRequest = {
    userId: number
    type: "recommend" | "similar"
    temp?: number,
    count?: number,
    genres?: MovieGenre[]
}
export type SimilarMoviesRequest = {
    userId: number
    start?: number
    count: number
    movieIds: number[]
    model?: AvailableModels
    temp: number
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