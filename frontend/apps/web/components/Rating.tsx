"use client";
import { roundRating } from "@/_lib/utils";
import { Star } from "@mui/icons-material";
import { Rating as BaseRating } from "@mui/material";

interface Type {
  v: number;
  onChange?: ((event: any, value: number | null) => void) | undefined;
  ro?: boolean;
  className?: string;
  showValue?: boolean;
}
export function FixedRating(props: Omit<Type, "onChange" | "ro">) {
  return <Base {...props} ro={true} />;
}
export function VarRating(props: Omit<Type, "ro">) {
  return <Base {...props} ro={false} />;
}
function Base({ v, onChange, ro, className, showValue }: Type) {
  v = roundRating(v);
  const darkMode = document.documentElement.classList.contains("dark");
  return (
    <span className="flex">
      {showValue && <span>({v})</span>}
      <BaseRating
        classes={"d"}
        className={className}
        readOnly={!!ro}
        value={v}
        precision={0.5}
        onChange={onChange}
        emptyIcon={darkMode ? <Star className="text-white/70" /> : null}
      />
    </span>
  );
}
