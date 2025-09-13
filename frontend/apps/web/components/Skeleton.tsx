import { ColStack, RowStack } from "./Container";
import { Skeleton as S } from "./ui/skeleton";

export default function Skeleton({
  row,
  nBoxes: NBoxes,
}: {
  nBoxes: number;
  row?: boolean;
}) {
  row = true;
  const Stack = row ? RowStack : ColStack;
  return (
    <Stack className="w-full gap-2 justify-evenly">
      {Array(NBoxes)
        .fill(0)
        .map((_, i) => (
          <S key={i} className="h-[300px] w-[250px] rounded-xl" />
        ))}
    </Stack>
  );
}
