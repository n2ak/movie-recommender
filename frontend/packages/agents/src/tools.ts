import { tool } from "langchain";
import * as z from "zod";
import { similaritySearch } from "./vectore_store";


const searchMovies = tool(
    async ({ query }, config) => {
        const movies = await similaritySearch(query, 5);
        console.log("\nQuery:", query);

        return movies.map(movie => JSON.stringify({
            id: movie.tmdbId,
            title: movie.title,
            description: movie.overview,
        })).join("****\n");
    },
    {
        name: "search_movies",
        description: "Search the database for movies matching the query. The query must be descriptive",
        schema: z.object({
            query: z.string().describe("User query"),
        }),
    }
);

export const tools = [searchMovies];