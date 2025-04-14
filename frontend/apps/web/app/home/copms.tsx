// "use client";
import {
  getMostWatchedGenres,
  getRecommendedGenreMovies,
  getRecommendedMoviesForUser,
} from "@/_lib/actions/action";
import { MovieRow } from "@/components/MovieCard";
import type { ModelType } from "@repo/backend_api";
import type { MovieGenre } from "@repo/database";

export async function Recommended({
  userId,
  model,
}: {
  userId: number;
  model: ModelType;
}) {
  const resp = await getRecommendedMoviesForUser(userId, model);
  if (!!resp.error) {
    console.error(resp.error);
    return <>Error getting recommendations</>;
  }
  return (
    <MovieRow
      title={"Movies we thing you would like:"}
      movies={resp.result.movies!}
      predictions={resp.result.predictions!}
    />
  );
}
export async function RecommendedGenre({
  userId,
  genre,
  model,
}: {
  userId: number;
  genre: MovieGenre;
  model: ModelType;
}) {
  const resp = await getRecommendedGenreMovies(userId, genre, model);

  if (!!resp.error) {
    console.error(resp.error);
    return <>Error getting recommendations</>;
  }
  return (
    <>
      {resp.result.movies!.length > 0 && (
        <MovieRow
          title={`Recommended '${genre}' movies:`}
          movies={resp.result.movies!}
          predictions={resp.result.predictions!}
        />
      )}
    </>
  );
}
export async function RecommendedGenres({ userId }: { userId: number }) {
  const genres = await getMostWatchedGenres(userId);
  return (
    <>
      {genres.map((g) => (
        <div key={g}>
          <RecommendedGenre
            userId={userId}
            genre={g as MovieGenre}
            model="DLRM"
          />
        </div>
      ))}
    </>
  );
}
