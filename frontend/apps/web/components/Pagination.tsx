import { RowStack } from "./Container";
// import Select from "./Select";
import * as Paging from "./ui/pagination";
import * as S from "./ui/select";

function Select({
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

export default function Pagination({
  nPages,
  values,
  pageNumber,
  setNumberOfRows,
  defaultNumberOfRows,
}: {
  nPages: number;
  values: number[];
  pageNumber: number;
  setNumberOfRows: (n: number) => void;
  defaultNumberOfRows: number;
}) {
  return (
    <RowStack>
      <Select
        values={[{ gValues: values.map((i) => i.toString()) }]}
        defaultValue={defaultNumberOfRows.toString()}
        onValueChange={(v) => setNumberOfRows(parseInt(v))}
        placeholder="Number of rows"
      />
      <Paging.Pagination>
        <Paging.PaginationContent>
          <Paging.PaginationItem>
            <Paging.PaginationPrevious href="#" />
          </Paging.PaginationItem>
          {new Array(nPages).fill(0).map((_, index) => (
            <Paging.PaginationItem key={index}>
              <Paging.PaginationLink href="#" isActive={index === pageNumber}>
                {index + 1}
              </Paging.PaginationLink>
            </Paging.PaginationItem>
          ))}
          <Paging.PaginationItem>
            <Paging.PaginationEllipsis />
          </Paging.PaginationItem>
          <Paging.PaginationItem>
            <Paging.PaginationNext href="#" />
          </Paging.PaginationItem>
        </Paging.PaginationContent>
      </Paging.Pagination>
    </RowStack>
  );
}
