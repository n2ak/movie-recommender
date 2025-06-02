import { usePathname, useSearchParams } from "next/navigation";
import { useCallback } from "react";

export default function useSearchParamsBuilder() {
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const updateQueryString = useCallback(
    (name: string, value: string | number) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set(name, `${value}`);
      return pathname + "?" + params.toString();
    },
    [searchParams, pathname]
  );
  return updateQueryString;
}
