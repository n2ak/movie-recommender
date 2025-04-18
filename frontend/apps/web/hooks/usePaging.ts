import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

export default function usePaging<T>({
  rowsPerPage: rPP,
  pageNumber: pN,
  keys,
  queryKey: qKey,
  fetchPage,
  nRecords,
}: {
  rowsPerPage?: number;
  pageNumber?: number;
  queryKey: string;
  keys: { [_: string]: any };
  fetchPage: (start: number, count: number) => Promise<T[]>;
  nRecords: number;
}) {
  const [pageNumber, setPageNumber] = useState<number>(pN || 1);
  const [rowsPerPage, setRowsPerPage] = useState<number>(rPP || 10);
  const queryKey = [
    qKey,
    {
      pageNumber,
      rowsPerPage,
      ...keys,
    },
  ];
  const { data } = useQuery({
    queryKey: queryKey,
    initialData: [],
    queryFn: () => fetchPage((pageNumber - 1) * rowsPerPage, rowsPerPage),
  });

  return {
    data,
    pageNumber,
    rowsPerPage,
    setPageNumber,
    setRowsPerPage,
    queryKey,
    nPages: Math.floor(nRecords / rowsPerPage) + 1, // it starts from zero
  };
}
