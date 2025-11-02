import { prismaClient } from "@repo/database";

const EMBEDDING_URL: string = process.env.EMBEDDING_URL || "";

if (!EMBEDDING_URL || EMBEDDING_URL === "") {
  throw Error("No EMBEDDING_URL");
}

async function getEmbedding(text: string) {
  const res = await fetch(EMBEDDING_URL, {
    method: "POST",
    body: JSON.stringify(text)
  }).then(a => a.json());
  return res;
}

export const similaritySearch = async (query: string, n: number) => {
  console.log("Getting the embeddings");
  const embedding = await getEmbedding(query)
  console.log("Got the embeddings");

  const vectorQuery = `[${embedding.join(',')}]`
  const movies = await prismaClient.$queryRaw`
      SELECT
        "tmdbId",
        "title",
        "overview",
        (overview_encoded <=> ${vectorQuery}::vector) as similarity
      FROM movie
    --   where (overview_encoded <=> ${vectorQuery}::vector) < .5
      ORDER BY  similarity ASC
      LIMIT ${n};
    `;
  return movies as {
    tmdbId: number,
    title: string,
    overview: string,
    similarity: number
  }[];
}