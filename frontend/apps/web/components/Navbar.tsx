"use client";
import { useAuthStore } from "@/hooks/useAuthStore";
import { useDictionary } from "@/hooks/useLanguageStore";
import { Avatar } from "@radix-ui/themes";
import Link from "next/link";
import { usePathname } from "next/navigation";
import * as React from "react";
import { ColStack } from "./Container";
import NavbarMenu from "./NavbarMenu";
import Search from "./Search";

export default function NavBar() {
  const dict = useDictionary();
  const user = useAuthStore((s) => s.user);
  const [menuOpen, setMenuOpen] = React.useState(false);
  const patnName = usePathname();

  if (!user) return <>unauthenticated</>;

  const links = [
    { name: dict.home, href: "/home" },
    { name: "Ratings", href: "/ratings" },
  ];

  return (
    <nav className="fixed top-0 w-full z-50 dark:bg-2 shadow  border-gray-200">
      <div className="max-w-screen-xl flex justify-between mx-auto p-4  max-h-[70px]">
        <a
          href="/home"
          className="flex items-center space-x-3 rtl:space-x-reverse"
        >
          <img
            src="https://flowbite.com/docs/images/logo.svg"
            className="h-8"
            alt="Flowbite Logo"
          />
        </a>
        <div className="relative md:order-2">
          {/* <button
            type="button"
            className="cursor-pointer hover:scale-[105%] flex text-sm md:me-0 dark:focus:ring-gray-600"
            id="user-menu-button"
            aria-expanded="false"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <AccountBoxIcon />
          </button> */}
          <Avatar
            src="https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?&w=256&h=256&q=70&crop=focalpoint&fp-x=0.5&fp-y=0.3&fp-z=1&fit=crop"
            fallback="A"
            className="cursor-pointer"
            size={"3"}
            radius="full"
            variant="soft"
            onClick={() => setMenuOpen(!menuOpen)}
          />
          <NavbarMenu
            email={user.email}
            username={user.name}
            menuOpen={menuOpen}
            setMenuOpen={setMenuOpen}
          />
        </div>
        <Search userId={user.id} />

        <div
          className="items-center justify-between hidden w-full md:flex md:w-auto md:order-1"
          id="navbar-user"
        >
          <ColStack className="font-medium p-4 md:p-0 mt-4 border border-gray-100 rounded-lg  md:space-x-8 rtl:space-x-reverse md:flex-row md:mt-0 md:border-0  ">
            {links.map((link) => (
              <div key={link.href}>
                {link.href !== patnName ? (
                  <Link
                    className="block hover:text-3 hover:scale-105 "
                    href={link.href}
                  >
                    {link.name}
                  </Link>
                ) : (
                  <button className="block text-blue-500" disabled>
                    {link.name}
                  </button>
                )}
              </div>
            ))}
          </ColStack>
        </div>
      </div>
    </nav>
  );
}
