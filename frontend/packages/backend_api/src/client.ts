import axios from "axios";

const apiClient = axios.create({
  baseURL: "http://localhost:3333",
});

// apiClient.interceptors.response.use(
//   (response) => response,
//   (error) => {
//     if (error.response) {
//       console.error("API Error:", error.response.status, error.response.data);
//     } else if (error.request) {
//       console.error("Network error or timeout:", error.message);
//     } else {
//       console.error("Unexpected error:", error.message);
//     }
//     return Promise.reject(error);
//   }
// );

export const movies_recom = async (data: any) => {
  return await apiClient.post("/movies-recom", data);
};
