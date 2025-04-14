import { PropsWithChildren } from "react";
import App from "./App";

export default async function RootLayout({ children }: PropsWithChildren) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body>
        <App>{children}</App>
      </body>
    </html>
  );
}
