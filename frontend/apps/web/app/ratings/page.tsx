"use client";
import { Container, RowStack } from "@/components/Container";
import { RatingsTable } from "@/components/MovieRatingsList";
import { useAuthStore } from "@/hooks/useAuthStore";
import useMostRatedGenres from "@/hooks/useMostRatedGenres";
import { PieChart } from "@mui/x-charts/PieChart";

export default function RatingsPage() {
  const user = useAuthStore((s) => s.user);
  // const [sortValue, setSortValue] = useState<{
  //   order: "asc" | "desc";
  //   key: keyof typeof sort;
  // }>({
  //   order: "asc",
  //   key: "timestamp",
  // });
  if (!user) {
    return null;
  }
  return (
    <Container>
      {/* <MostRatedGenres userId={user.id} /> */}
      <RowStack className="mb-5 w-full justify-between">
        {/* <FormControl
          sx={{
            width: "45%",
            // height: "60px",
            display: "flex",
          }}
        >
          <InputLabel
            id="demo-simple-select-label"
            className="dark:!text-white"
          >
            Sort
          </InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={sortValue.key}
            label="Sort by"
            className="!border-white"
            onChange={(e, a) => {
              setSortValue({
                ...sortValue,
                key: e.target.value as RatingSortKey,
              });
            }}
          >
            {Object.entries(sort).map(([k, v]) => (
              <MenuItem key={k} value={k}>
                {v}
              </MenuItem>
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
        </FormControl> */}
      </RowStack>
      <RatingsTable userId={user.id} />
      {/* <RatingsList userId={user.id} sortValue={sortValue} /> */}
    </Container>
  );
}

function MostRatedGenres({ userId }: { userId: number }) {
  const genres = useMostRatedGenres(userId);
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
