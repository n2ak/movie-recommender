"use client";
import { getRatingsForUser } from "@/lib/actions/action";
import { RatingWithMovie } from "@/lib/db/movie";
import { roundRating, timeSince } from "@/lib/utils";
import { ArrowDownward, ArrowUpward } from "@mui/icons-material";
import {
  Box,
  Container,
  FormControl,
  InputLabel,
  MenuItem,
  Rating,
  Select,
  Skeleton,
  Stack,
  Typography,
} from "@mui/material";
import { useSession } from "next-auth/react";
import Link from "next/link";
import { useEffect, useState } from "react";
function capitalizeFirstLetter(s: string) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
export default function RatingsPage() {
  const session = useSession();
  if (session.status === "loading") return <>Loading</>;
  if (session.status === "unauthenticated") return <>No session</>;
  const { user } = session.data as any;
  const userId = parseInt(user?.id as string);
  const [ratings, setRatings] = useState<RatingWithMovie[] | undefined>(
    undefined
  );
  const [sortValue, setSortValue] = useState(1);
  const [sortOrderValue, setSortOrderValue] = useState(2);
  const sort = {
    rating: "Rating",
    title: "Title",
    avg_rating: "Avg Rating",
    timestamp: "Time",
  };
  useEffect(() => {
    (async function () {
      const o = sortOrderValue == 1 ? "asc" : "desc";
      const s = Object.keys(sort)[sortValue] as any;
      const r = await getRatingsForUser(userId, 0, 10, s, o);
      if (!!r) {
        console.log("****", o, s);
        setRatings(r);
      }
    })();
  }, [sortValue, userId, sortOrderValue]);
  return (
    <Container>
      <Box>
        <FormControl>
          <InputLabel id="demo-simple-select-label">Sort</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={sortValue}
            label="Sort by"
            onChange={(e, a) => {
              setSortValue(e.target.value as number);
            }}
          >
            {Object.keys(sort).map((st) => (
              <MenuItem value={Object.keys(sort).indexOf(st)}>
                {(sort as any)[st]}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl>
          <InputLabel id="demo-simple-select-label">Order</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={sortOrderValue}
            label="Sort by"
            onChange={(e, a) => {
              setSortOrderValue(e.target.value as number);
            }}
          >
            <MenuItem value={1}>
              Ascending
              <ArrowUpward />
            </MenuItem>
            <MenuItem value={2}>
              Descending
              <ArrowDownward />
            </MenuItem>
          </Select>
        </FormControl>
      </Box>
      {!!ratings ? <List ratings={ratings} /> : <Skeleton_ nbox={5} />}
    </Container>
  );
}
function List({ ratings }: { ratings: RatingWithMovie[] }) {
  return (
    <>
      <Stack direction="column" spacing={1}>
        {ratings.map((rating, i) => (
          <Box
            key={i}
            sx={{ display: "flex", flexDirection: "row", height: "200px" }}
          >
            <Box>
              <Link
                href={`/movie/${rating.movie.movieId}`}
                style={{ textDecoration: "none" }}
              >
                <img
                  src={rating.movie.href}
                  style={{ width: "100%", height: "100%" }}
                />
              </Link>
            </Box>
            <Box sx={{ paddingLeft: 10 }}>
              <Link
                href={`/movie/${rating.movie.movieId}`}
                style={{ textDecoration: "none" }}
              >
                <Typography
                  color="white"
                  fontSize={30}
                  fontWeight={50}
                  sx={{
                    marginBottom: 1,
                    ":hover": {
                      textDecoration: "underline",
                    },
                  }}
                >
                  {rating.movie.title}
                </Typography>
              </Link>
              <Typography>
                Avg rating:
                <Rating value={roundRating(rating.movie.avg_rating)} readOnly />
                ({rating.movie.total_ratings} users)
              </Typography>
              <Typography>
                Your rating:
                <Rating value={roundRating(rating.rating)} readOnly />
              </Typography>
              <Typography>Made: {timeSince(rating.timestamp)} ago</Typography>
            </Box>
          </Box>
        ))}
      </Stack>
    </>
  );
}

function Skeleton_({ nbox }: { nbox: number }) {
  return (
    <Container>
      <Stack direction={"column"}>
        {Array(nbox)
          .fill(0)
          .map((_, i) => (
            <Box sx={{ width: 210, marginRight: 0.5, my: 5 }} key={i}>
              <Skeleton variant="rectangular" sx={{ height: 100 }} />
            </Box>
          ))}
      </Stack>
    </Container>
  );
}
