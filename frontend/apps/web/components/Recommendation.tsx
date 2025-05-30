// "use client";
import { MovieCarousel } from "@/components/MovieCarousel";
import {
  getMostWatchedGenres,
  getRecommendedGenreMovies,
  getRecommendedMoviesForUser,
} from "@/lib/actions/movie";
import type { MovieGenre } from "@repo/database";

export async function Recommended() {
  return (
    <Recommendation
      title={"Movies we thing you would like:"}
      func={() => getRecommendedMoviesForUser({ count: 10 })}
    />
  );
}

async function RecommendedGenre({ genre }: { genre: MovieGenre }) {
  return (
    <Recommendation
      title={
        <span>
          Recommended {<span className="text-red-600">{genre}</span>} movies:
        </span>
      }
      func={() => getRecommendedGenreMovies({ genre })}
    />
  );
}

export async function RecommendedGenres() {
  const genres = (await getMostWatchedGenres({})).data!.map(([g]) => g);
  return (
    <>
      {genres.map((g) => (
        <div key={g}>
          <RecommendedGenre genre={g as MovieGenre} />
        </div>
      ))}
    </>
  );
}

async function Recommendation({
  title,
  func,
}: {
  title: string | React.ReactNode;
  func: () => ReturnType<typeof getRecommendedMoviesForUser>;
}) {
  const movies = (await func()).data!;
  return (
    <>{movies!.length > 0 && <MovieCarousel title={title} movies={movies} />}</>
  );
}
