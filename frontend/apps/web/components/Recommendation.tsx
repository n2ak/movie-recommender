// "use client";
import {
  getMostWatchedGenres,
  getRecommendedGenreMovies,
  getRecommendedMoviesForUser,
} from "@/_lib/actions/movie";
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
  return (
    <Recommendation
      userId={userId}
      title={"Movies we thing you would like:"}
      func={() => getRecommendedMoviesForUser(userId, model)}
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
  return (
    <Recommendation
      title={
        <span>
          Recommended {<span className="text-red-600">{genre}</span>} movies:
        </span>
      }
      func={() => getRecommendedGenreMovies(userId, genre, model)}
      userId={userId}
    />
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

async function Recommendation({
  userId,
  title,
  func,
}: {
  userId: number;
  title: any;
  func: () => ReturnType<typeof getRecommendedMoviesForUser>;
}) {
  const resp = await func();

  if (resp.error) {
    console.error(resp.error);
    return <>Error getting recommendations</>;
  }
  return (
    <>
      {resp.result.movies!.length > 0 && (
        <MovieRow
          title={title}
          movies={resp.result.movies!}
          userId={userId}
          predictions={resp.result.predictions!}
        />
      )}
    </>
  );
}
