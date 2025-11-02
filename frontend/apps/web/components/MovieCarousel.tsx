"use client";

import type { MovieWithPredictedRating } from "@/lib/actions/movie";
import Link from "next/link";
import { ColStack } from "./Container";
import { FixedRating } from "./Rating";
import * as Carousel from "./ui/carousel";

export function MovieCarousel({
  title,
  movies,
  scroll,
}: {
  title?: string | React.ReactNode;
  movies: MovieWithPredictedRating[];
  scroll?: boolean;
}) {
  return (
    <div className="">
      {title && (
        <h2 className="font-mono text-2xl font-medium tracking-tight">
          {title}
        </h2>
      )}
      {scroll ? (
        <div className="overflow-y-scroll scroll h-full px-1 py-5 scrollbar-hidden">
          <div className="flex justify-between gap-0.5 z-50">
            {movies.map((movie, i) => (
              <div key={i} className="inline-flex min-w-[200px] h-[300px]">
                <MovieCard movie={movie} />
              </div>
            ))}
          </div>
        </div>
      ) : (
        <Carousel.Carousel
          opts={{
            align: "start",
          }}
          className="w-full mx-auto"
        >
          <Carousel.CarouselContent>
            {movies.map((movie) => (
              <Carousel.CarouselItem
                key={movie.tmdbId}
                className="xl:basis-1/5 sm:basis-1/3 md:basis-1/4"
              >
                <div className="p-1 min-w-[200px] h-[300px]">
                  <MovieCard movie={movie} />
                </div>
              </Carousel.CarouselItem>
            ))}
          </Carousel.CarouselContent>
          <Carousel.CarouselPrevious />
          <Carousel.CarouselNext />
        </Carousel.Carousel>
      )}
    </div>
  );
}

function MovieCard({ movie }: { movie: MovieWithPredictedRating }) {
  return (
    <Link
      href={`/movie/${movie.tmdbId}`}
      style={{
        width: "100%",
      }}
    >
      <div className="group relative h-full w-full shadow-xl hover:scale-[110%] duration-[.2s] hover:z-50">
        <div className="rounded-md !h-full !w-full bg-black">
          <img className="rounded-md !h-full !w-full" src={movie.href} alt={movie.title} />
        </div>
        <ColStack className="absolute rounded-b-lg bottom-0 left-0 w-full pl-[10px] truncate backdrop-blur-sm bg-black/[1%] hidden group-hover:block text-shadow-2xs text-shadow-black">
          <div className="w-full inline-block text-ellipsis overflow-hidden whitespace-nowrap text-left">
            <span className="font-bold text-white">{movie.title}</span>
          </div>
          <div className="text-white">
            <FixedRating
              v={movie.avg_rating}
              className="text-left text-white"
            />
          </div>
          <div className="w-full inline-block text-ellipsis overflow-hidden whitespace-nowrap text-white">
            Prediction:
            <br />
            <FixedRating v={movie.predictedRating} />
          </div>
        </ColStack>
      </div>
    </Link>
  );
}
