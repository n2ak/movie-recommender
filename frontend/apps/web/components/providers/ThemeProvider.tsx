"use client";
import { ThemeProvider as NextThemesProvider } from "next-themes";

export default function ThemeProvider({
  children,
}: React.ComponentProps<typeof NextThemesProvider>) {
  return (
    <NextThemesProvider
      defaultTheme="system"
      enableSystem
      attribute="class"
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  );
}
