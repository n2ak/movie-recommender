"use client";
import { useUIStore } from "@/hooks/useUIStore";
import { MAX_RATING } from "@/lib/constants";
import { roundRating } from "@/lib/utils";
import { Rating as BaseRating } from "@mui/material";
import { Star } from "lucide-react";

interface Type {
  v: number;
  onChange: ((value: number) => void) | undefined;
  className?: string;
  showValue?: boolean;
}
export function FixedRating(props: Omit<Type, "onChange" | "ro">) {
  const v = Math.floor(props.v * 10) / 10;
  return (
    <div className="flex items-center">
      <Star className="text-yellow-500" />
      <span className="text-center content-center items-center center">
        {v}/{MAX_RATING}
      </span>
    </div>
  );
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
        max={10}
      />
      {showValue && <span className="h-full my-auto">({v})</span>}
    </span>
  );
}
