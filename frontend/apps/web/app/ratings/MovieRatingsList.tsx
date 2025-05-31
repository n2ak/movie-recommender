import useInfiniteMovieRatings from "@/hooks/useInfiniteMovieRatings";
import { MovieWithUserRating } from "@/lib/actions/movie";
import { joinCN } from "@/lib/utils";
import { Table } from "@radix-ui/themes";
import { RatingSortKey } from "@repo/database";
import { useQueryClient } from "@tanstack/react-query";
import { ArrowDownIcon, ArrowUpIcon, EditIcon } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import Pagination from "../../components/Pagination";
import EditMovieRatingAndReviewModal from "../../components/RateMovieModal";
import { FixedRating } from "../../components/Rating";

export function RatingsTable({ userId }: { userId: number }) {
  const [sortKey, setSortKey] = useState<RatingSortKey>("title");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");

  const {
    data: movies,
    setPageNumber,
    queryKey,
    pageNumber,
    nPages,
    setRowsPerPage,
  } = useInfiniteMovieRatings(userId, sortKey, sortOrder);

  const [selectedRating, setSelectedRating] = useState<MovieWithUserRating>();
  const qClient = useQueryClient();
  return (
    <>
      <div className="flex items-center flex-col">
        <Pagination
          nPages={nPages}
          pageNumber={pageNumber}
          setNumberOfRows={setRowsPerPage}
          values={[10, 25, 50]}
          defaultNumberOfRows={10}
        />
      </div>
      <Table.Root>
        <TableHeader
          setSortOrder={setSortOrder}
          setSortKey={setSortKey as (s: string) => void}
          sortOrder={sortOrder}
          sortKey={sortKey}
        />

        <Table.Body>
          {movies.map((movie) => {
            return (
              <Table.Row
                key={movie.id}
                className="hover:bg-gray-100 group dark:hover:bg-gray-600"
              >
                <TableRow movie={movie} setSelectedRating={setSelectedRating} />
              </Table.Row>
            );
          })}
        </Table.Body>
      </Table.Root>
      <EditMovieRatingAndReviewModal
        movie={selectedRating}
        onClose={() => setSelectedRating(undefined)}
        onSave={() => {
          setSelectedRating(undefined);
          qClient.invalidateQueries({
            queryKey: queryKey,
          });
        }}
      />
    </>
  );
}

function TableHeader({
  sortKey,
  setSortOrder,
  setSortKey,
  sortOrder,
}: {
  sortKey: RatingSortKey;
  setSortOrder: (_: "asc" | "desc") => void;
  setSortKey: (_: string) => void;
  sortOrder: "asc" | "desc";
}) {
  const defaultOrder = "asc";

  const cols = {
    title: "Title",
    timestamp: "Time of rating",
    rating: "Rating",
    // year: "Time of rating",
    // avg_rating: "Avg Rating",
  };
  return (
    <Table.Header>
      <Table.Row>
        {Object.entries(cols).map(([key, name]) => {
          const isTheOne = sortKey === key;
          return (
            <Table.ColumnHeaderCell key={key}>
              <span
                className={joinCN(
                  "flex gap-3 cursor-pointer items-center hover:text-gray-500",
                  !isTheOne ? "text-gray-400" : ""
                )}
                onClick={() => {
                  if (!isTheOne) {
                    setSortKey(key);
                    setSortOrder(defaultOrder);
                  } else {
                    setSortOrder(sortOrder === "asc" ? "desc" : "asc");
                  }
                }}
              >
                <span>{name}</span>
                {isTheOne && sortOrder === "desc" ? (
                  <ArrowDownIcon />
                ) : (
                  <ArrowUpIcon />
                )}
              </span>
            </Table.ColumnHeaderCell>
          );
        })}
      </Table.Row>
    </Table.Header>
  );
}
function TableRow({
  movie,
  setSelectedRating,
}: {
  movie: MovieWithUserRating;
  setSelectedRating: (_: MovieWithUserRating) => void;
}) {
  const userRating = movie.userRating[0]!;
  return (
    <>
      <Table.Cell>
        <Link
          href={`/movie/${movie.id}`}
          className="flex flex-row items-center gap-3"
        >
          <img
            alt={movie.title}
            src={movie.href}
            className="hover:scale-105 duration-100 max-h-15"
          />
          <div className="max-w-[200px] overflow-ellipsis overflow-hidden group-hover:underline">
            {movie.title}
          </div>
        </Link>
      </Table.Cell>
      <Table.Cell>{userRating.timestamp.toDateString()}</Table.Cell>
      <Table.Cell className="relative">
        {/* {userRating.rating} */}
        <FixedRating v={userRating.rating} showValue />
        <div
          className="absolute hidden group-hover:block top-0.5 right-0.5 hover:bg-gray-400/40 p-1 rounded-sm cursor-pointer dark:hover:bg-black"
          onClick={() => setSelectedRating(movie)}
        >
          <EditIcon />
        </div>
      </Table.Cell>
    </>
  );
}
