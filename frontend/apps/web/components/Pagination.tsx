import { Pagination as Paging } from "@mui/material";
import { RowStack } from "./Container";
import Select from "./Select";

export default function Pagination({
  pageNumber,
  nPages,
  setPageNumber,
  values,
  setNumberOfRows,
  defaultNumberOfRows,
}: {
  nPages: number;
  setPageNumber: (n: number) => void;
  setNumberOfRows: (n: number) => void;
  pageNumber: number;
  values: number[];
  defaultNumberOfRows: number;
}) {
  return (
    <RowStack>
      <Select
        defaultValue={"" + defaultNumberOfRows}
        label="Number of rows"
        onValueChange={(v) => setNumberOfRows(parseInt(v))}
        values={values.map((n) => n.toString())}
      />
      <Paging
        className="dark:bg-gray-400 dark:rounded-sm"
        count={nPages}
        onChange={(a, b) => {
          setPageNumber(b);
        }}
        page={pageNumber}
      />
    </RowStack>
  );
}
