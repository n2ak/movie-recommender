"use client";
import { getMovieForUser, rateMovie } from "@/_lib/actions/action";
import { ColStack } from "@/components/Container";
import { FixedRating, VarRating } from "@/components/Rating";
import { useSnackBar } from "@/components/providers/SnackBarProvider";
import { useAuthStore } from "@/hooks/useAuthStore";
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "@mui/material";
import { MovieForUser } from "@repo/database";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

export default function MoviePage() {
  const user = useAuthStore((s) => s.user);
  const { id } = useParams();
  const movieId = parseInt(id as string);
  const [movie, setMovie] = useState<MovieForUser | null>(null);
  const [open, setOpen] = useState(false);
  const [personalRating, setPersonalRating] = useState(-1);
  const [tempPersonalRating, setTempPersonalRating] = useState(0);
  useEffect(() => {
    (async () => {
      try {
        if (!user) return;
        const result = await getMovieForUser(user.id, movieId);
        if (!!result) {
          setMovie(result);
          const pr = result.userRating?.rating;
          if (pr) {
            setPersonalRating(pr);
            setTempPersonalRating(pr);
          }
        }
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    })();
  }, [id, user]);
  const snackBar = useSnackBar();
  if (!user) {
    return <>Unauthenticated</>;
  }
  if (!movie) return <>Loading</>;

  const rate = async () => {
    console.log("********Rating", movie.title, tempPersonalRating);
    const a = await snackBar.handlePromise(
      rateMovie(user.id, movieId, tempPersonalRating).then(() => {
        setPersonalRating(tempPersonalRating);
        setOpen(false);
      }),
      `Movie ${movie.title} rated.`,
      "Problem rating."
    );
  };
  return (
    <ColStack className="justify-center items-center">
      <div className="w-2/4">
        <div className="justify-center flex">
          <img src={movie.href} />
        </div>
        <ColStack className="justify-center mt-4">
          <h1 className="text-center text-xl font-sans">{movie.title}</h1>
          <div className="flex text-center gap-2 justify-items-center justify-center mt-4 mb-4">
            <span style={{ height: "100%" }}>Rating:</span>
            <FixedRating v={movie.avg_rating} />
            <span>({movie.total_ratings} users)</span>
          </div>
          {personalRating !== -1 && (
            <div className="flex text-center text-center justify-center gap-2">
              Personal Rating:
              <FixedRating v={personalRating}></FixedRating>
            </div>
          )}
          <Button variant="outlined" onClick={() => setOpen((o) => !o)}>
            Rate
          </Button>
          <Dialog
            // selectedValue={selectedValue}
            open={open}
            // onClose={handleClose}
          >
            <DialogTitle>Give '{movie.title}' your rating.</DialogTitle>
            <DialogContent>
              <Box
                noValidate
                component="form"
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  m: "auto",
                  width: "fit-content",
                }}
              >
                <VarRating
                  v={tempPersonalRating}
                  onChange={(_, v) => setTempPersonalRating(v as number)}
                />
              </Box>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => rate()}>Done</Button>
              <Button
                onClick={() => {
                  setOpen(false);
                  setTempPersonalRating(personalRating);
                }}
              >
                Close
              </Button>
            </DialogActions>
          </Dialog>
        </ColStack>
      </div>
    </ColStack>
  );
}
