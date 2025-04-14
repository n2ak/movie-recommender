"use client";
import { useUIStore } from "@/hooks/useUIStore";
import { Theme } from "@radix-ui/themes";
import { PropsWithChildren, useEffect } from "react";

export default function ThemeProvider({ children }: PropsWithChildren) {
  const darkMode = useUIStore((s) => s.isDarkMode);
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);
  return <Theme appearance={darkMode ? "dark" : "light"}>{children}</Theme>;
}
