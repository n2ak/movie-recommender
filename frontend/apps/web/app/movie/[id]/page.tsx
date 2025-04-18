"use client";
import Button from "@/components/Button";
import { ColStack } from "@/components/Container";
import RateMovieModal from "@/components/RateMovieModal";
import { FixedRating } from "@/components/Rating";
import { useAuthStore } from "@/hooks/useAuthStore";
import useMovie from "@/hooks/useMovie";
import { useQueryClient } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { useState } from "react";

export default function MoviePage() {
  const user = useAuthStore((s) => s.user);
  const { id } = useParams();
  const movieId = parseInt(id as string);
  const [open, setOpen] = useState(false);
  const { movie, queryKey } = useMovie({ userId: user?.id, movieId });
  const qClient = useQueryClient();
  if (!user) {
    return <>Unauthenticated</>;
  }
  if (!movie) return <>Loading</>;

  return (
    <ColStack className="justify-center items-center">
      <div className="w-2/4">
        <div className="justify-center flex">
          <img src={movie.href} />
        </div>
        <ColStack className="justify-center mt-4 gap-2">
          <h1 className="text-center text-xl font-sans">{movie.title}</h1>
          <div className="flex text-center gap-2 justify-items-center justify-center mt-4 mb-4">
            <span style={{ height: "100%" }}>Rating:</span>
            <FixedRating v={movie.avg_rating} />
            <span>({movie.total_ratings} users)</span>
          </div>
          {!!movie.userRating[0] && (
            <div className="flex text-center justify-center gap-2">
              Personal Rating:
              <FixedRating v={movie.userRating[0]?.rating}></FixedRating>
            </div>
          )}
          <Button onClick={() => setOpen((o) => !o)} className="!w-20 mx-auto">
            Rate
          </Button>

          <RateMovieModal
            onClose={() => setOpen(false)}
            onSave={() => {
              setOpen(false);
              qClient.invalidateQueries({
                queryKey,
              });
            }}
            movie={!open ? undefined : movie}
          />
        </ColStack>
      </div>
    </ColStack>
  );
}
