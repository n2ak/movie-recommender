"use client";
import { useUIStore } from "@/hooks/useUIStore";
import { Theme } from "@radix-ui/themes";
import { PropsWithChildren, useEffect } from "react";

export default function ThemeProvider({ children }: PropsWithChildren) {
  const { isDarkMode, __loaded: loaded } = useUIStore();
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);
  if (!loaded) {
    return null;
  }
  return <Theme appearance={isDarkMode ? "dark" : "light"}>{children}</Theme>;
}
