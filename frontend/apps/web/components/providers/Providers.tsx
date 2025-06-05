"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtoolsPanel } from "@tanstack/react-query-devtools";
import type { Session } from "next-auth";
import { SessionProvider } from "next-auth/react";
import type { PropsWithChildren } from "react";
import { Toaster } from "../ui/sonner";
import AuthSyncProvider from "./AuthSyncProvider";
import { ClientMetricsLogger } from "./ClientMetricLogger";
import ThemeProvider from "./ThemeProvider";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

export default function Providers({
  session,
  children,
}: PropsWithChildren & { session: Session | null }) {
  return (
    <SessionProvider session={session}>
      <ThemeProvider>
        <QueryClientProvider client={queryClient}>
          <AuthSyncProvider />
          <ClientMetricsLogger />
          {children}
          {process.env.NODE_ENV === "development" && (
            <ReactQueryDevtoolsPanel />
          )}
        </QueryClientProvider>
        <Toaster />
      </ThemeProvider>
    </SessionProvider>
  );
}
