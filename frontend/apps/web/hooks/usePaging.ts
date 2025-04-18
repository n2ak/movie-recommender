import { SortOrder } from "@repo/database";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

function usePaging<T>({
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

export default function useInfinitePaging<T, SK>(
  sortKey: SK,
  sortOrder: SortOrder,
  func: (start: number, count: number) => Promise<T[]>,
  nRecordsFn: () => Promise<number>,
  qKey: string,
  nRecordsQKey: any[],
  extra_keys?: object
) {
  const { data: nrecords } = useQuery({
    queryKey: nRecordsQKey,
    queryFn: nRecordsFn,
    initialData: 0,
  });
  return usePaging({
    fetchPage: async (start, count) => {
      const res = await func(start, count);
      return res;
    },
    queryKey: qKey,
    keys: {
      sortKey,
      sortOrder,
      ...extra_keys,
    },
    nRecords: nrecords,
  });
}
