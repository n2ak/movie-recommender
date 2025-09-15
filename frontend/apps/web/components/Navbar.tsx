"use client";
import { useAuthStore, type UserInfo } from "@/hooks/useAuthStore";
import { useDictionary } from "@/hooks/useLanguageStore";
import { MoonIcon, SunIcon } from "lucide-react";
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
  const Component = theme === "dark" ? MoonIcon : SunIcon;
  return (
    <div className="content-center">
      <Component
        className="cursor-pointer border text-primary border-primary p-1 w-9 h-9 rounded-sm "
        onClick={() => setTheme(theme === "light" ? "dark" : "light")}
      />
    </div>
  );
}
function OnNavbar({ user }: { user: UserInfo }) {
  const dict = useDictionary();
  const links = [
    { name: dict.home, href: "/" },
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
  return (
    <nav className="fixed top-0 w-full z-50 shadow bg-white/60 backdrop-blur-sm border-gray-200 dark:bg-black/60">
      <div className="max-w-screen-xl flex justify-between mx-auto p-4 max-h-[70px]">
        <Logo />
        <SwitchModeIcon />
      </div>
    </nav>
  );
}

function Links({ links }: { links: { name: string; href: string }[] }) {
  const pathname = usePathname();

  return (
    <div
      className="items-center justify-between hidden w-full md:flex md:w-auto"
      id="navbar-user"
    >
      <RowStack className="font-medium gap-2">
        {links.map((link) => (
          <div key={link.href} className="">
            {link.href !== pathname ? (
              <Link
                className="font-mono py-1 text-primary rounded-xs"
                href={link.href}
              >
                {link.name}
              </Link>
            ) : (
              <span className="font-mono py-1 px-3 text-white rounded-xs bg-primary">
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
    <Link href="/" className="hidden sm:block relative">
      <img
        src="/res/movies.png"
        className="h-15 w-15 -translate-y-3"
        alt="Flowbite Logo"
      />
    </Link>
  );
}
