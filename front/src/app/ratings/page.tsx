"use client";
import { getRatingsForUser } from "@/lib/actions/action";
import { RatingSortBy, RatingWithMovie } from "@/lib/db/movie";
import { Sorting, useDBList } from "@/lib/useDbList";
import { roundRating, timeSince } from "@/lib/utils";
import { ArrowDownward, ArrowUpward } from "@mui/icons-material";
import {
  Box,
  Container,
  Divider,
  FormControl,
  InputLabel,
  MenuItem,
  Rating,
  Select,
  Skeleton,
  Stack,
  Typography,
} from "@mui/material";
import { Prisma } from "@prisma/client";
import { useSession } from "next-auth/react";
import Link from "next/link";

const sort = {
  rating: "Rating",
  title: "Title",
  avg_rating: "Avg Rating",
  timestamp: "Time",
};
export default function RatingsPage() {
  const session = useSession();
  if (session.status === "loading") return <>Loading</>;
  if (session.status === "unauthenticated") return <>No session</>;
  const { user } = session.data as any;
  const userId = parseInt(user?.id as string);

  const {
    setSortValue,
    sortValue,
    values: ratings,
  } = useDBList(
    (s: Sorting<RatingSortBy>) =>
      getRatingsForUser(userId, 0, 10, s.key as any, s.order),
    {
      key: "timestamp",
      order: "asc",
    }
  );
  return (
    <Container sx={{ marginTop: 5 }}>
      <Box display="flex" flexDirection="row" sx={{ marginBottom: 5 }}>
        <FormControl
          sx={{
            width: "50%",
            // height: "60px",
            display: "flex",
          }}
        >
          <InputLabel id="demo-simple-select-label">Sort</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={sortValue.key}
            label="Sort by"
            onChange={(e, a) => {
              setSortValue({
                ...sortValue,
                key: e.target.value as RatingSortBy,
              });
            }}
          >
            {Object.keys(sort).map((st) => (
              <MenuItem value={st}>{(sort as any)[st]}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl
          sx={{
            width: "50%",
            // height: "60px",
          }}
        >
          <InputLabel id="demo-simple-select-label">Order</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={sortValue.order}
            label="Sort by"
            onChange={(e, a) => {
              setSortValue({
                ...sortValue,
                order: e.target.value as Prisma.SortOrder,
              });
            }}
          >
            <MenuItem
              value={"asc"}
              sx={{
                verticalAlign: "middle",
              }}
            >
              low to high
              <ArrowUpward
                sx={{
                  height: "100%",
                  verticalAlign: "middle",
                  marginLeft: 1,
                }}
              />
            </MenuItem>
            <MenuItem value={"desc"}>
              high to low
              <ArrowDownward
                sx={{
                  height: "100%",
                  verticalAlign: "middle",
                  marginLeft: 1,
                }}
              />
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
          <>
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
                  <Rating
                    value={roundRating(rating.movie.avg_rating)}
                    readOnly
                  />
                  ({rating.movie.total_ratings} users)
                </Typography>
                <Typography>
                  Your rating:
                  <Rating value={roundRating(rating.rating)} readOnly />
                </Typography>
                <Typography>
                  Rated : {timeSince(rating.timestamp)} ago
                </Typography>
              </Box>
            </Box>
            <Divider variant="middle" />
          </>
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
