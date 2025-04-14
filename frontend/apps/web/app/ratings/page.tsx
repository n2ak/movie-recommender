"use client";
import { getMostGenresRatings } from "@/_lib/actions/action";
import { Container, RowStack } from "@/components/Container";
import { RatingsList } from "@/components/RatingsList";
import { useAuthStore } from "@/hooks/useAuthStore";
import { ArrowDownward, ArrowUpward } from "@mui/icons-material";
import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import { PieChart } from "@mui/x-charts/PieChart";
import { RatingSortBy, SortOrder } from "@repo/database";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

const sort = {
  timestamp: "Time",
  rating: "Rating",
  avg_rating: "Avg Rating",
  title: "Title",
};

export default function RatingsPage() {
  const user = useAuthStore((s) => s.user);
  const [sortValue, setSortValue] = useState<{
    order: "asc" | "desc";
    key: keyof typeof sort;
  }>({
    order: "asc",
    key: "timestamp",
  });
  if (!user) {
    return null;
  }
  return (
    <Container>
      <MostRated userId={user.id} />
      <RowStack className="mb-5 w-full justify-between">
        <FormControl
          sx={{
            width: "45%",
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
            {Object.entries(sort).map(([k, v]) => (
              <MenuItem value={k}>{v}</MenuItem>
            ))}
          </Select>
        </FormControl>
        <FormControl
          sx={{
            width: "45%",
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
                order: e.target.value as SortOrder,
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
      </RowStack>
      <RatingsList userId={user.id} sortValue={sortValue} />
      {/* 
      {!ratings || isLoading ? (
        <RatingsListSkeleton nbox={5} />
      ) : (
      )}
      */}
    </Container>
  );
}

function MostRated({ userId }: { userId: number }) {
  const {
    data: genres,
    isLoading,
    isError,
  } = useQuery({
    initialData: [],
    queryKey: [
      "user_most_rated_genres",
      {
        userId,
      },
    ],
    queryFn: () => getMostGenresRatings(userId),
  });
  const k = 5;
  const genres_truncated = genres.slice(0, k);
  if (genres.length > k) {
    const sum = genres_truncated.slice(k).reduce((c, a) => a[1] + c, 0);
    genres_truncated.push(["other..", sum]);
  }
  const data = genres_truncated.map((a, id) => {
    return { id, value: a[1], label: a[0] };
  });
  return (
    <div className="w-full  flex mb-1">
      {genres.length > 0 && (
        <PieChart
          series={[
            {
              data,
            },
          ]}
          width={400}
          height={200}
          title="Most rated genred"
        />
      )}
    </div>
  );
}
