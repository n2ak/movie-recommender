"use client";
import { roundRating } from "@/_lib/utils";
import { useUIStore } from "@/hooks/useUIStore";
import { Star } from "@mui/icons-material";
import { Rating as BaseRating } from "@mui/material";

interface Type {
  v: number;
  onChange: ((value: number) => void) | undefined;
  className?: string;
  showValue?: boolean;
}
export function FixedRating(props: Omit<Type, "onChange" | "ro">) {
  return <Base {...props} onChange={undefined} />;
}

export function VarRating(props: Omit<Type, "ro">) {
  return <Base {...props} />;
}

function Base({ v, onChange, className, showValue }: Type) {
  v = roundRating(v);
  const { isDarkMode: darkMode } = useUIStore();
  return (
    <span className="flex gap-1">
      <BaseRating
        classes={"d"}
        className={className}
        readOnly={!onChange}
        value={v}
        precision={0.5}
        onChange={(_, v) => {
          if (!!onChange && !!v) onChange(v);
        }}
        emptyIcon={darkMode ? <Star className="text-white/70" /> : null}
      />
      {showValue && <span className="h-full my-auto">({v})</span>}
    </span>
  );
}
