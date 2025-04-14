import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

export default function usePaging<T>({
  rowsPerPage: rPP,
  pageNumber: pN,
  keys,
  queryKey,
  fetchPage,
}: {
  rowsPerPage?: number;
  pageNumber?: number;
  queryKey: string;
  keys: { [_: string]: any };
  fetchPage: (start: number, count: number) => Promise<T[]>;
}) {
  const [pageNumber, setPageNumber] = useState<number>(pN || 0);
  const [rowsPerPage, setRowsPerPage] = useState<number>(rPP || 10);
  const { data, isLoading, isError } = useQuery({
    queryKey: [
      queryKey,
      {
        pageNumber,
        rowsPerPage,
        ...keys,
      },
    ],
    initialData: [],
    queryFn: () => fetchPage(pageNumber * rowsPerPage, rowsPerPage),
  });

  return {
    data,
    pageNumber,
    rowsPerPage,
    setPageNumber,
    setRowsPerPage,
  };
}
