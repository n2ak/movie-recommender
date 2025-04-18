"use client";

import Link from "next/link";
import type { MovieWithUserRating } from "../../../packages/database/src/movieDb";
import { ColStack } from "./Container";
import { FixedRating } from "./Rating";

export default function MovieCard({
  movie,
  pred,
}: {
  movie: MovieWithUserRating;
  pred: number;
  userId: number;
}) {
  // const ref = useRef<HTMLSpanElement>(null);
  // const [showToolTip, setShowToolTip] = useState(true);

  // TODO Register movie in queryClient ?

  return (
    <Link
      href={`/movie/${movie.id}`}
      style={{
        width: "100%",
      }}
    >
      <div className="group relative h-full w-full shadow-xl hover:scale-[110%] duration-[.2s] hover:z-10">
        <img className="rounded-lg h-full" src={movie.href} alt={movie.title} />
        <ColStack className="absolute rounded-b-lg bg-black opacity-0 bottom-0 left-0 w-full pl-[10px] truncate group-hover:opacity-[70%] group-hover:">
          <div className="w-full  shadow-2xl inline-block text-ellipsis overflow-hidden whitespace-nowrap text-left">
            <span className="font-bold text-white">{movie.title}</span>
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
  userId,
}: {
  title: any;
  movies: MovieWithUserRating[];
  predictions: number[];
  userId: number;
}) {
  return (
    <div className="">
      <h2 className="font-sans text-2xl font-medium tracking-tight">{title}</h2>
      <div className="overflow-y-scroll scroll h-full px-1 py-5 scrollbar-hidden">
        <div className="flex justify-between gap-1">
          {movies.map((movie, i) => (
            <div key={i} className="inline-flex min-w-[200px] h-[300px]">
              <MovieCard
                userId={userId}
                movie={movie}
                pred={predictions[i] as number}
              />
            </div>
          ))}
        </div>
      </div>
      {/* <Divider /> */}
    </div>
  );
}
