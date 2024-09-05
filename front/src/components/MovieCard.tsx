"use client";
import Card from "@mui/material/Card";
import Typography from "@mui/material/Typography";
import "./row.css";

import { alpha, styled } from "@mui/material/styles";
import { Movie } from "@/lib/definitions";
import Link from "next/link";
import { Fragment, useLayoutEffect, useRef, useState } from "react";
import { Box, List, Rating, Stack } from "@mui/material";
import Tooltip from "@mui/material/Tooltip";
import { roundRating } from "@/lib/utils";

const Cardd = styled(Card)(({ theme }) => ({
  width: 300,
  color: theme.palette.success.main,
}));

export default function MovieCard({
  movie,
  pred,
}: {
  movie: Movie;
  pred: number;
}) {
  // const avg_rating = Math.round(movie.avg_rating * 2) / 2;
  const ref = useRef<HTMLSpanElement>(null);
  const [showToolTip, setShowToolTip] = useState(true);
  // useLayoutEffect(() => {
  //   const current = ref.current;
  //   if (!!current) {
  //     if (current.clientWidth < current.scrollWidth) {
  //       setShowToolTip(true);
  //     }
  //   }
  // }, [ref]);
  return (
    <Tooltip title={showToolTip ? movie.title : ""} enterDelay={1000}>
      <Link
        href={`/movie/${movie.movieId}`}
        style={{
          width: "100%",
        }}
      >
        <Box
          className="img_container"
          sx={{
            width: "100%",
          }}
        >
          <img className="poster" src={movie.href} alt={movie.title} />
          <Box
            sx={
              {
                // position: "absolute",
                // left: 0,
                // botttm: 0,
                // display: "none",
              }
            }
            className="text"
          >
            <Typography
              color="white"
              // className="row__posterName"
              sx={{
                width: "100%",
                display: "inline-block",
                textAlign: "left",
                textOverflow: "ellipsis",
                overflow: "hidden !important",
                whiteSpace: "nowrap",
              }}
              // ref={ref}
            >
              <span ref={ref}>{movie.title}</span>
            </Typography>
            <Typography
              sx={{
                width: "100%",
                display: "inline-block",
                textOverflow: "ellipsis",
                overflow: "hidden !important",
                whiteSpace: "nowrap",
              }}
              color="white"
              // className="row__posterName"
            >
              <Rating
                readOnly
                value={roundRating(movie.avg_rating)}
                precision={0.5}
              ></Rating>
            </Typography>
            <Typography
              sx={{
                width: "100%",
                display: "inline-block",
                textOverflow: "ellipsis",
                overflow: "hidden !important",
                whiteSpace: "nowrap",
              }}
              color="white"
              // className="row__posterName"
            >
              Prediction:
              <Rating
                readOnly
                value={roundRating(pred)}
                precision={0.5}
              ></Rating>
            </Typography>
          </Box>
        </Box>
      </Link>
    </Tooltip>
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
    <div>
      <h2>{title}</h2>
      <Stack
        className="row"
        direction={"row"}
        spacing={2}
        sx={{
          // display: "flex",
          overflowY: "hidden",
          overflowX: "scroll",
          // m: 300,
          minHeight: 300,
        }}
      >
        {movies.map((movie, i) => (
          <Box
            key={i}
            display={"inline-flex"}
            sx={{
              minWidth: 200,
              // width: 200,
              height: 300,
            }}
          >
            <MovieCard movie={movie} pred={predictions[i]} />
          </Box>
        ))}
      </Stack>
    </div>
  );
}
