// import Select from "./Select";
import * as S from "./ui/select";

export default function Select({
  onValueChange,
  values,
  defaultValue,
  placeholder,
}: {
  placeholder: string;
  onValueChange: (a: string) => void;
  values: { gLabel?: string; gValues: string[] }[];
  defaultValue?: string;
}) {
  return (
    <S.Select onValueChange={onValueChange} defaultValue={defaultValue}>
      <S.SelectTrigger className="w-[180px]">
        <S.SelectValue placeholder={placeholder} />
      </S.SelectTrigger>
      <S.SelectContent>
        {values.map(({ gLabel, gValues }, gidx) => (
          <S.SelectGroup key={gidx}>
            {gLabel && <S.SelectLabel>{gLabel}</S.SelectLabel>}
            {gValues.map((value, vidx) => (
              <S.SelectItem value={value} key={vidx}>
                {value}
              </S.SelectItem>
            ))}
          </S.SelectGroup>
        ))}
      </S.SelectContent>
    </S.Select>
  );
}
