"use client";
import { MAX_RATING } from "@/lib/constants";
import { roundRating } from "@/lib/utils";
import { Rating as BaseRating } from "@mui/material";
import { Star } from "lucide-react";
import { useTheme } from "next-themes";

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

export function VarRating({
  v,
  onChange,
  className,
  showValue,
}: Omit<Type, "ro">) {
  v = roundRating(v);
  const { theme } = useTheme();
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
        emptyIcon={theme === "dark" ? <Star className="text-white/70" /> : null}
        max={10}
      />
      {showValue && <span className="h-full my-auto">({v})</span>}
    </span>
  );
}
