"use client";
import { useAuthStore, User } from "@/hooks/useAuthStore";
import { useDictionary } from "@/hooks/useLanguageStore";
import { useUIStore } from "@/hooks/useUIStore";
import { Avatar } from "@radix-ui/themes";
import { Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";
import { RowStack } from "./Container";
import NavbarMenu from "./NavbarMenu";
import Search from "./Search";

export default function NavBar() {
  const user = useAuthStore((s) => s.user);

  if (user) return <OnNavbar user={user} />;
  return <OffNavbar />;
}

function OnNavbar({ user }: { user: User }) {
  const dict = useDictionary();
  const links = [
    { name: dict.home, href: "/home" },
    { name: "Ratings", href: "/ratings" },
  ];
  const [menuOpen, setMenuOpen] = React.useState(false);
  const ref = React.useRef<HTMLElement | null>(null);
  return (
    <nav className="fixed top-0 w-full z-50 shadow bg-white/60 backdrop-blur-sm border-gray-200 dark:bg-black/60">
      <div className="max-w-screen-xl flex justify-between mx-auto p-4 max-h-[70px]">
        <Logo />
        <Search userId={user.id} />
        <Links links={links} />
        <div className="relative">
          <Avatar
            src="https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?&w=256&h=256&q=70&crop=focalpoint&fp-x=0.5&fp-y=0.3&fp-z=1&fit=crop"
            fallback="A"
            className="cursor-pointer "
            size={"3"}
            radius="full"
            variant="soft"
            onClick={() => setMenuOpen(!menuOpen)}
            ref={ref}
          />
          <NavbarMenu
            email={user.email}
            username={user.name}
            menuOpen={menuOpen}
            setMenuOpen={setMenuOpen}
            avatarRef={ref}
          />
        </div>
      </div>
    </nav>
  );
}
function OffNavbar() {
  const { isDarkMode, toggleDarkMode } = useUIStore();

  return (
    <nav className="fixed top-0 w-full z-50 shadow bg-white/60 backdrop-blur-sm border-gray-200 dark:bg-black/60">
      <div className="max-w-screen-xl flex justify-between mx-auto p-4 max-h-[70px]">
        <Logo />
        <div
          className="cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-500 px-1 py-1 rounded-sm"
          onClick={toggleDarkMode}
        >
          {isDarkMode ? <Sun /> : <Moon />}
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
        // src="https://flowbite.com/docs/images/logo.svg"
        src="/res/movies.png"
        className="h-15 w-15 -translate-y-3"
        alt="Flowbite Logo"
      />
    </Link>
  );
}
