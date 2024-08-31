"use client";
import { useParams } from "next/navigation";
import {
  Typography,
  Container,
  Rating,
  Button,
  Dialog,
  DialogTitle,
  Box,
  DialogContent,
  DialogActions,
} from "@mui/material";
import { useEffect, useState } from "react";
import { getMovieForUser, rateMovie } from "@/lib/actions/action";
import { useSession } from "next-auth/react";
import { MovieForUser } from "@/lib/definitions";
import { useSnackBar } from "@/components/SnackBarProvider";
import { roundRating } from "@/lib/utils";

export default function MoviePage() {
  const session = useSession();
  if (session.status === "loading") return <>Loading</>;
  if (session.status === "unauthenticated") return <>No session</>;
  const { user } = session.data as any;
  console.log("session", user, typeof user.id);

  const { id } = useParams();
  const movieId = parseInt(id as string);
  const userId = parseInt(user.id);
  const [movie, setMovie] = useState<MovieForUser | null>(null);
  const [open, setOpen] = useState(false);
  const [personalRating, setPersonalRating] = useState(-1);
  const [tempPersonalRating, setTempPersonalRating] = useState(0);
  useEffect(() => {
    async function fetch() {
      try {
        const result = await getMovieForUser(userId, movieId);
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
    }
    fetch();
  }, [id]);
  if (!movie) return <>Loading</>;
  const snackBar = useSnackBar();

  const rate = async () => {
    console.log("********Rating", movie.title, tempPersonalRating);
    const a = await snackBar.handlePromise(
      rateMovie(userId, movieId, tempPersonalRating).then(() => {
        setPersonalRating(tempPersonalRating);
        setOpen(false);
      }),
      `Movie ${movie.title} rated.`,
      "Problem rating."
    );
    console.log("Result", a);
  };
  const avg_rating = roundRating(movie.avg_rating);
  return (
    <>
      <Container maxWidth="sm" sx={{ marginTop: 10 }}>
        <Box justifyContent={"center"} display={"flex"}>
          <img src={movie.href} />
        </Box>
        <Box
          display={"flex"}
          justifyContent={"center"}
          flexDirection={"column"}
        >
          <Typography textAlign={"center"} padding={2}>
            {movie.title}
          </Typography>
          <Typography
            textAlign={"center"}
            padding={2}
            justifyItems={"center"}
            justifyContent={"center"}
          >
            <span style={{ height: "100%" }}>Rating:</span>
            <Rating
              readOnly
              value={avg_rating}
              precision={0.5}
              sx={{
                // paddingTop: 1,
                alignItems: "center",
              }}
            ></Rating>
            <span>({movie.total_ratings} users )</span>
          </Typography>
          {personalRating !== -1 && (
            <Typography textAlign={"center"} padding={2}>
              Personal Rating:
              <Rating readOnly value={personalRating} precision={0.5}></Rating>
              {personalRating}
            </Typography>
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
                <Rating
                  value={tempPersonalRating}
                  onChange={(_, v) => setTempPersonalRating(v as number)}
                  precision={0.5}
                ></Rating>
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
        </Box>
      </Container>
    </>
  );
}
