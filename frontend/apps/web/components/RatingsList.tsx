import { getNumberOfRatings, getRatingsForUser } from "@/_lib/actions/action";
import { timeSince } from "@/_lib/utils";
import usePaging from "@/hooks/usePaging";
import { Skeleton, TablePagination } from "@mui/material";
import { RatingWithMovie } from "@repo/database";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { ColStack, Container, Divider, RowStack } from "./Container";
import { FixedRating } from "./Rating";

export function RatingsListSkeleton({ nbox }: { nbox: number }) {
  return (
    <Container>
      <ColStack>
        {Array(nbox)
          .fill(0)
          .map((_, i) => (
            <div key={i}>
              <div
                className="w-[210px] mr-0.5 my-5"
                // sx={{ width: 210, marginRight: 0.5, my: 5 }}
                key={i}
              >
                <Skeleton variant="rectangular" sx={{ height: 100 }} />
              </div>
            </div>
          ))}
      </ColStack>
    </Container>
  );
}

const sort = {
  timestamp: "Time",
  rating: "Rating",
  avg_rating: "Avg Rating",
  title: "Title",
};
export function RatingsList({
  userId,
  sortValue,
}: {
  userId: number;
  sortValue: {
    order: "asc" | "desc";
    key: keyof typeof sort;
  };
}) {
  const {
    data: ratings,
    setPageNumber,
    setRowsPerPage,
    rowsPerPage,
    pageNumber,
  } = usePaging({
    fetchPage: (start, count) =>
      getRatingsForUser(userId, start, count, sortValue.key, sortValue.order),
    queryKey: "movies_ratings",
    keys: {
      userId,
    },
  });
  const { data: nratings } = useQuery({
    queryKey: [
      "movies_nratings",
      {
        userId,
      },
    ],
    initialData: -1,
    queryFn: () => getNumberOfRatings(userId),
  });
  return (
    <ColStack>
      <div className="w-full flex my-5">
        <TablePagination
          component="div"
          disabled={ratings.length == 0}
          onPageChange={(_, page) => {
            setPageNumber(page);
          }}
          count={nratings}
          color="primary"
          className="mx-auto"
          page={pageNumber}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(r) => setRowsPerPage(parseInt(r.target.value))}
        />
      </div>
      {ratings.length > 0 ? (
        ratings.map((rating) => (
          <div key={rating.id}>
            <RatingsListItem rating={rating} />
            <Divider className="dark:bg-white" />
          </div>
        ))
      ) : (
        <RatingsListSkeleton nbox={5} />
      )}
    </ColStack>
  );
}
function RatingsListItem({ rating }: { rating: RatingWithMovie }) {
  return (
    <RowStack className="h-40 hover:bg-gray-100 hover:dark:bg-black">
      <div>
        <Link
          href={`/movie/${rating.movie.id}`}
          style={{ textDecoration: "none" }}
        >
          <img
            src={rating.movie.href}
            className="hover:scale-105 duration-100"
            style={{ width: "100%", height: "100%" }}
          />
        </Link>
      </div>
      <div className="pl-10">
        <Link
          href={`/movie/${rating.movie.id}`}
          style={{ textDecoration: "none" }}
        >
          <div className="mb-1 font-sans font-bold text-black text-xl hover:underline dark:text-white">
            {rating.movie.title}
          </div>
        </Link>
        <div className="flex gap-2">
          <span>Avg rating:</span>
          <FixedRating showValue v={rating.movie.avg_rating} />
          <span className="italic">, ({rating.movie.total_ratings} users)</span>
        </div>
        <div className="flex gap-2">
          <span className="pr-1">Your rating:</span>
          <FixedRating showValue v={rating.rating} />
        </div>
        <div>
          <span className="pr-2">Rated:</span>
          <span className="italic">{timeSince(rating.timestamp)} ago</span>
        </div>
      </div>
    </RowStack>
  );
}
