import { Prisma } from "@prisma/client";
import { useEffect, useState } from "react";

export interface Sorting<ST> {
  key: ST;
  order: Prisma.SortOrder;
}

export function useDBList<T, ST>(
  func: (s: Sorting<ST>) => Promise<T>,
  setValue: (value: T) => void,
  defaultSorting: Sorting<ST>
) {
  const [sortValue, setSortValue] = useState<Sorting<ST>>(defaultSorting);

  useEffect(() => {
    (async function () {
      const r = await func(sortValue);
      if (!!r) {
        setValue(r);
      }
    })();
  }, [sortValue]);
  return {
    setSortValue,
    sortValue,
  };
}
