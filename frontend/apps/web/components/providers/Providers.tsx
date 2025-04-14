"use client";
import { CssBaseline } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactQueryDevtoolsPanel } from "@tanstack/react-query-devtools";
import type { Session } from "next-auth";
import { SessionProvider } from "next-auth/react";
import { PropsWithChildren } from "react";
import AuthSyncProvider from "./AuthSyncProvider";
import { SnackBarProvider } from "./SnackBarProvider";
import ThemeProvider from "./ThemeProvider";

const queryClient = new QueryClient({});

export default function Providers({
  session,
  children,
}: PropsWithChildren & { session: Session | null }) {
  return (
    <SessionProvider session={session}>
      <ThemeProvider>
        <CssBaseline />
        <QueryClientProvider client={queryClient}>
          <SnackBarProvider>
            <AuthSyncProvider />
            {children}
          </SnackBarProvider>
          <ReactQueryDevtoolsPanel />
        </QueryClientProvider>
      </ThemeProvider>
    </SessionProvider>
  );
}
