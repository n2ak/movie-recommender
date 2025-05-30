import { ColStack, RowStack } from "./Container";
import { Skeleton as S } from "./ui/skeleton";

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
          <S key={i} className="h-[300px] w-[250px] rounded-xl" />
        ))}
    </Stack>
  );
}
