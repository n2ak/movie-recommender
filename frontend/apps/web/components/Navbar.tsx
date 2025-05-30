"use client";
import { useAuthStore, UserInfo } from "@/hooks/useAuthStore";
import { useDictionary } from "@/hooks/useLanguageStore";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Avatar from "./Avatar";
import { RowStack } from "./Container";
import Search from "./Search";

export default function NavBar() {
  const user = useAuthStore((s) => s.user);

  if (user) return <OnNavbar user={user} />;
  return <OffNavbar />;
}

function SwitchModeIcon() {
  const { theme, setTheme } = useTheme();
  const Component = theme === "dark" ? Moon : Sun;
  return (
    <div className="content-center">
      <Component
        className="cursor-pointer border border-black p-1 w-9 h-9 rounded-sm dark:border-white"
        onClick={() => setTheme(theme === "light" ? "dark" : "light")}
      />
    </div>
  );
}
function OnNavbar({ user }: { user: UserInfo }) {
  const dict = useDictionary();
  const links = [
    { name: dict.home, href: "/home" },
    { name: "Ratings", href: "/ratings" },
  ];
  return (
    <nav className="fixed top-0 w-full z-50 shadow bg-white/60 backdrop-blur-sm border-gray-200 dark:bg-black/60">
      <div className="max-w-screen-xl flex justify-between mx-auto p-4 max-h-[70px]">
        <Logo />
        <Search />
        <Links links={links} />
        <SwitchModeIcon />
        <Avatar username={user.username} />
      </div>
    </nav>
  );
}
function OffNavbar() {
  const { setTheme, theme } = useTheme();

  return (
    <nav className="fixed top-0 w-full z-50 shadow bg-white/60 backdrop-blur-sm border-gray-200 dark:bg-black/60">
      <div className="max-w-screen-xl flex justify-between mx-auto p-4 max-h-[70px]">
        <Logo />
        <div
          className="cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-500 px-1 py-1 rounded-sm"
          onClick={() => setTheme(theme === "light" ? "dark" : "light")}
        >
          {theme === "dark" ? <Sun /> : <Moon />}
        </div>
      </div>
    </nav>
  );
}

function Links({ links }: { links: { name: string; href: string }[] }) {
  const patnName = usePathname();

  return (
    <div
      className="items-center justify-between hidden w-full md:flex md:w-auto"
      id="navbar-user"
    >
      <RowStack className="font-medium gap-2">
        {links.map((link) => (
          <div key={link.href} className="">
            {link.href !== patnName ? (
              <Link
                className="hover:scale-105 py-1 hover:bg-2 hover:text-4 dark:hover:text-black dark:hover:bg-4 px-2 rounded-xs"
                href={link.href}
              >
                {link.name}
              </Link>
            ) : (
              <span className="text-base py-1 text-2 bg-gray-200 dark:bg-2 dark:text-gray-200 px-2 rounded-xs">
                {link.name}
              </span>
            )}
          </div>
        ))}
      </RowStack>
    </div>
  );
}

function Logo() {
  return (
    <Link href="/home" className="hidden sm:block relative">
      <img
        src="/res/movies.png"
        className="h-15 w-15 -translate-y-3"
        alt="Flowbite Logo"
      />
    </Link>
  );
}
