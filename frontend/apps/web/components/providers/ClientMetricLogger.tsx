// components/ClientMetricsLogger.tsx
"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

export function ClientMetricsLogger() {
  const pathname = usePathname();
  const navStart = useRef(performance.now());

  useEffect(() => {
    const now = performance.now();
    const duration = now - navStart.current;
    console.log("hydration", duration);
    navStart.current = now;
  }, [pathname]);

  return null;
}
