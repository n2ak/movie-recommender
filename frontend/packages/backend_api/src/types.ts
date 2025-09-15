import type { MovieGenre } from "@repo/database";

export type AvailableModels = "xgb_cpu" | "dlrm_cpu" | "xgb_cuda" | "dlrm_cuda";

export type BackendRequest = {
    userId: number
    count: number
    genres?: MovieGenre[]
    start?: number
    model?: AvailableModels,
    temp: number

}
export type Prediction = {
    movieId: number;
    userId: number;
    predicted_rating: number;
};

export type BackendResponse<P = Prediction> = {
    result: P[];
    time: number;
    status_code: number | string;
    error:
    | {
        [index: string]: string;
    }
    | undefined;
};