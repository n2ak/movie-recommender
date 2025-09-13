import { useQuery } from "@tanstack/react-query";

export function usePaging<T>({
  rowsPerPage,
  pageNumber,
  keys,
  queryKey: qKey,
  fetchPage,
  nRecordsFn,
  nRecordsQKey,
}: {
  nRecordsQKey: any[];
  rowsPerPage: number;
  pageNumber: number;
  queryKey: string;
  keys: { [_: string]: any };
  fetchPage: (start: number, count: number) => Promise<T[]>;
  nRecordsFn: () => Promise<number>;
}) {
  const { data: nRecords } = useQuery({
    queryKey: nRecordsQKey,
    queryFn: nRecordsFn,
    initialData: 0,
  });

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
    queryFn: () => fetchPage(pageNumber * rowsPerPage, rowsPerPage),
  });
  return {
    data,
    pageNumber,
    rowsPerPage,
    queryKey,
    nPages: Math.floor(nRecords / rowsPerPage) + 1, // it starts from zero
  };
}
