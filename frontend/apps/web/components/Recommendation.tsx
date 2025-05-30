// "use client";
import { MovieRow } from "@/components/MovieCard";
import {
  getMostWatchedGenres,
  getRecommendedGenreMovies,
  getRecommendedMoviesForUser,
} from "@/lib/actions/movie";
import type { MovieGenre } from "@repo/database";

export async function Recommended({ userId }: { userId: number }) {
  return (
    <Recommendation
      userId={userId}
      title={"Movies we thing you would like:"}
      func={() => getRecommendedMoviesForUser({ count: 10 })}
    />
  );
}

export async function RecommendedGenre({
  userId,
  genre,
}: {
  userId: number;
  genre: MovieGenre;
}) {
  return (
    <Recommendation
      title={
        <span>
          Recommended {<span className="text-red-600">{genre}</span>} movies:
        </span>
      }
      func={() => getRecommendedGenreMovies({ genre })}
      userId={userId}
    />
  );
}

export async function RecommendedGenres({ userId }: { userId: number }) {
  const genres = (await getMostWatchedGenres(userId)).data!.map(([g, _]) => g);
  return (
    <>
      {genres.map((g) => (
        <div key={g}>
          <RecommendedGenre userId={userId} genre={g as MovieGenre} />
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
  const movies = (await func()).data!;
  return (
    <>{movies!.length > 0 && <MovieRow title={title} movies={movies} />}</>
  );
}
