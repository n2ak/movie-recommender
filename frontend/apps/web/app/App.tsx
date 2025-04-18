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
      <div className="min-h-screen w-full pt-25 mx-auto px-2 lg:px-20 bg-gray-200 dark:bg-black">
        {children}
      </div>
    </Providers>
  );
}
