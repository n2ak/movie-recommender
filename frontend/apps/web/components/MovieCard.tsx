"use client";

import { Movie } from "@repo/database";
import Link from "next/link";
import { useRef } from "react";
import { ColStack } from "./Container";
import { FixedRating } from "./Rating";

export default function MovieCard({
  movie,
  pred,
}: {
  movie: Movie;
  pred: number;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  // const [showToolTip, setShowToolTip] = useState(true);
  return (
    <Link
      href={`/movie/${movie.id}`}
      style={{
        width: "100%",
      }}
    >
      <div className="group relative h-full w-full shadow-xl w-full hover:scale-[110%] duration-[.2s] hover:z-10">
        <img className="rounded-lg h-full" src={movie.href} alt={movie.title} />
        <ColStack className="absolute rounded-b-lg bg-black opacity-0 bottom-0 left-0 w-full pl-[10px] truncate group-hover:opacity-[70%] group-hover:">
          <div className="w-full  shadow-2xl inline-block text-ellipsis overflow-hidden whitespace-nowrap text-left">
            <span ref={ref} className="font-bold text-white">
              {movie.title}
            </span>
          </div>

          <FixedRating v={movie.avg_rating} className="text-left" />
          <div className="w-full inline-block text-ellipsis overflow-hidden whitespace-nowrap text-white">
            Prediction:
            <br />
            <FixedRating v={pred} />
          </div>
        </ColStack>
      </div>
    </Link>
  );
}

export function MovieRow({
  title,
  movies,
  predictions,
}: {
  title: string;
  movies: Movie[];
  predictions: number[];
}) {
  return (
    <div className="mt-3">
      <h2 className="text-3 font-sans text-3xl">{title}</h2>
      <div className="overflow-y-scroll scroll h-full px-1 py-10 scrollbar-hidden">
        <div className="flex justify-between space-x-1 ">
          {movies.map((movie, i) => (
            <div key={i} className="inline-flex min-w-[200px] h-[300px]">
              <MovieCard movie={movie} pred={predictions[i] as number} />
            </div>
          ))}
        </div>
      </div>
      {/* <Divider /> */}
    </div>
  );
}
