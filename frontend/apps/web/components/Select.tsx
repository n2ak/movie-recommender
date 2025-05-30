import { Select as S } from "@radix-ui/themes";

export default function Select({
  defaultValue,
  onValueChange,
  label,
  values,
}: {
  defaultValue: string;
  label: string;
  onValueChange: (s: string) => void;
  values: readonly string[];
}) {
  return (
    <S.Root defaultValue={defaultValue} onValueChange={onValueChange}>
      <S.Trigger />
      <S.Content>
        <S.Group>
          <S.Label>{label}</S.Label>
          {values.map((l) => (
            <S.Item key={l} value={l}>
              {l}
            </S.Item>
          ))}
        </S.Group>
      </S.Content>
    </S.Root>
  );
}
