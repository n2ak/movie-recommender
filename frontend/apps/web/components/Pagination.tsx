import { RowStack } from "./Container";
import Select from "./Select";
// import Select from "./Select";
import * as Paging from "./ui/pagination";

export default function Pagination({
  nPages,
  values,
  pageNumber,
  setNumberOfRows,
  defaultNumberOfRows,
  linkBuilder,
}: {
  nPages: number;
  values: number[];
  pageNumber: number;
  setNumberOfRows: (n: number) => void;
  defaultNumberOfRows: number;
  linkBuilder: (page: number) => string;
}) {
  const hasPrevious = pageNumber > 0;
  const hasNext = pageNumber < nPages - 1;
  const nLinksToShow = 5; // better to be odd
  const n = Math.floor((nLinksToShow - 1) / 2);
  return (
    <RowStack>
      <Select
        values={[{ gValues: values.map((i) => i.toString()) }]}
        defaultValue={defaultNumberOfRows.toString()}
        onValueChange={(v) => setNumberOfRows(parseInt(v))}
        placeholder="Number of rows"
      />
      <Paging.Pagination>
        <Paging.PaginationContent>
          <Paging.PaginationItem>
            <Paging.PaginationPrevious
              className={
                !hasPrevious ? "pointer-events-none opacity-50" : undefined
              }
              aria-disabled={!hasPrevious}
              href={linkBuilder(pageNumber - 1)}
            />
          </Paging.PaginationItem>
          {new Array(nPages).fill(0).map((_, index) => {
            // TODO could be done better
            if (index < pageNumber - n || index > pageNumber + n) return;
            return (
              <Paging.PaginationItem key={index}>
                <Paging.PaginationLink
                  href={linkBuilder(index)}
                  isActive={index === pageNumber}
                >
                  {index + 1}
                </Paging.PaginationLink>
              </Paging.PaginationItem>
            );
          })}
          {pageNumber < nPages - (n + 1) && (
            <Paging.PaginationItem>
              <Paging.PaginationEllipsis />
            </Paging.PaginationItem>
          )}
          <Paging.PaginationItem>
            <Paging.PaginationNext
              className={
                !hasNext ? "pointer-events-none opacity-50" : undefined
              }
              aria-disabled={!hasNext}
              href={linkBuilder(pageNumber + 1)}
            />
          </Paging.PaginationItem>
        </Paging.PaginationContent>
      </Paging.Pagination>
    </RowStack>
  );
}
