"use client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useEffect, useState } from "react";
import { create } from "zustand";
import { persist } from "zustand/middleware";

type Temperature = {
  temp: number | null;
  setTemp: (temp: number) => void;
}

export const useTemperatureStore = create<Temperature>()(
  persist(
    (set) => ({
      temp: 0,
      setTemp: (temp) => set({ temp })
    }),
    {
      name: "temperature",
    }
  )
);

function clamp(v: number, min: number, max: number) {
  return Math.min(Math.max(v, min), max);
}

export function TemperatureInput() {
  const tempStore = useTemperatureStore();
  const [t, setT] = useState(tempStore.temp || 0);
  console.log("tempStore.temp", tempStore.temp);
  useEffect(() => {
    if (tempStore.temp)
      setT(tempStore.temp);
  }, [tempStore.temp]);

  return <div
    className="flex justify-center gap-2 items-center"
  >
    Temperature
    <Input
      type="number"
      min={0}
      max={1}
      step={0.01}
      className="dark:border-white border-2 border-neutral-900 rounded px-2 py-1 w-24"
      placeholder="0 - 1"
      value={t}
      onChange={(e) => setT(Number(e.target.value))}
    />
    <Button
      onClick={() => tempStore.setTemp(clamp(t, 0, 1))}
      className="cursor-pointer"
    >
      Save
    </Button>
  </div>
}