import { auth } from "@/auth";
import NavBar from "@/components/Navbar";
import Providers from "@/components/providers/Providers";
import "@radix-ui/themes/styles.css";
import "../globals.css";

export default async function App({ children }: any) {
  const session = await auth();
  return (
    <Providers session={session}>
      <NavBar />
      <div className="pt-20 bg-back bg-fixed">{children}</div>
    </Providers>
  );
}
