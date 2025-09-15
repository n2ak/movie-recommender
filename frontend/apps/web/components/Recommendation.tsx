"use client"
import { useTemperatureStore } from "@/app/(home)/temp-input";
import { MovieCarousel } from "@/components/MovieCarousel";
import {
  getRecommendedGenreMovies,
  getRecommendedMoviesForUser
} from "@/lib/actions/movie";
import type { MovieGenre } from "@repo/database";
import { useQuery } from "@tanstack/react-query";
import { Suspense } from "react";
import Skeleton from "./Skeleton";

export function Recommended() {
  const store = useTemperatureStore();
  if (!store || store.temp === null)
    return null;
  const temp = store.temp;
  return (
    <Recommendation
      title={"Movies we think you would like:"}
      func={() => getRecommendedMoviesForUser({ count: 10, temp })}
      qkey={["Recommended", String(temp)]}
    />
  );
}

function RecommendedGenre({ genre }: { genre: MovieGenre }) {
  const store = useTemperatureStore();
  if (!store || store.temp === null)
    return null;
  const temp = store.temp;
  return (
    <Recommendation
      title={
        <span>
          Recommended {<span className="text-primary">{genre}</span>} movies:
        </span>
      }
      func={() => getRecommendedGenreMovies({ genre, temp })}
      qkey={["RecommendedGenre", genre, String(temp)]}
    />
  );
}

export function RecommendedGenres({ genres }: { genres: string[] }) {
  return (
    <>
      {genres.map((g) => (
        <div key={g}>
          <Suspense fallback={<Skeleton nBoxes={5} />}>
            <RecommendedGenre genre={g as MovieGenre} />
          </Suspense>
        </div>
      ))}
    </>
  );
}

function Recommendation({
  title,
  func,
  qkey,
}: {
  title: string | React.ReactNode;
  func: () => ReturnType<typeof getRecommendedMoviesForUser>;
  qkey: string[]
}) {
  const q = useQuery({
    queryFn: async () => (await func()).data!,
    queryKey: qkey,
    initialData: []
  });
  return (
    <>{q.data!.length > 0 && <MovieCarousel title={title} movies={q.data} />}</>
  );
}
