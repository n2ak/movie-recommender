import Appp from "./Appp";
import { auth } from "@/auth";
import { PropsWithChildren } from "react";
import { SessionProvider } from "next-auth/react";
import { SnackBarProvider } from "@/components/SnackBarProvider";

export default async function RootLayout({ children }: PropsWithChildren) {
  const session = await auth();
  return (
    <>
      <html lang="en" suppressHydrationWarning>
        <head />
        <body>
          <SessionProvider session={session}>
            <SnackBarProvider>
              <Appp>{children}</Appp>
            </SnackBarProvider>
          </SessionProvider>
        </body>
      </html>
    </>
  );
}
