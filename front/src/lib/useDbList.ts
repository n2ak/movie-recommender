import { Prisma } from "@prisma/client";
import { useEffect, useState } from "react";

export interface Sorting<K> {
  key: K;
  order: Prisma.SortOrder;
}

export function useDBList<T, K>(
  func: (s: Sorting<K>) => Promise<T>,
  defaultSorting: Sorting<K>
) {
  const [sortValue, setSortValue] = useState<Sorting<K>>(defaultSorting);
  const [values, setValues] = useState<T | undefined>(undefined);
  useEffect(() => {
    (async function () {
      const r = await func(sortValue);
      if (!!r) {
        setValues(r);
      }
    })();
  }, [sortValue]);
  return {
    setSortValue,
    sortValue,
    values,
  };
}
