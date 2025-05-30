import { joinCN } from "@/lib/utils";
import { Skeleton as S } from "@mui/material";
import { ColStack, RowStack } from "./Container";

export default function Skeleton({
  row,
  nbox,
}: {
  nbox: number;
  row?: boolean;
}) {
  row = true;
  const Stack = row ? RowStack : ColStack;
  return (
    <Stack className="w-full gap-2 justify-evenly">
      {Array(nbox)
        .fill(0)
        .map((_, i) => (
          <S
            key={i}
            variant="rectangular"
            className={joinCN(
              row ? "!min-h-[200px]" : "!w-full",
              "rounded-md w-full !bg-gray-700"
            )}
          />
        ))}
    </Stack>
  );
}
