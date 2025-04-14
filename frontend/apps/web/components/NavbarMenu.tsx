"use client";
import { logOut } from "@/_lib/actions/action";
import { useAuthStore } from "@/hooks/useAuthStore";
import { useOnClickOutside } from "@/hooks/useClickedOutside";
import { useDictionary } from "@/hooks/useLanguageStore";
import { useUIStore } from "@/hooks/useUIStore";
import { Button, Separator } from "@radix-ui/themes";
import { signOut } from "next-auth/react";
import Link from "next/link";
import * as React from "react";
import { ColStack } from "./Container";

export default function NavbarMenu({
  menuOpen,
  setMenuOpen,
  username,
  email,
}: {
  menuOpen: boolean;
  username: string;
  email: string;
  setMenuOpen: (open: boolean) => void;
}) {
  const dict = useDictionary();
  const { isDarkMode, toggleDarkMode } = useUIStore();
  const clearUser = useAuthStore((s) => s.clearUser);
  const ref = React.useRef<HTMLDivElement>(null);
  useOnClickOutside(ref, () => {
    setMenuOpen(false);
  });
  const profileLinks = [
    {
      href: "/profile",
      name: "Profile",
    },
  ];
  if (!menuOpen) {
    return null;
  }
  return (
    <div
      ref={ref}
      className="absolute right-0 z-50 min-w-44 my-4 text-base list-none bg-white divide-y divide-gray-100 rounded-lg shadow dark:bg-gray-700 dark:divide-gray-600"
      id="user-dropdown"
    >
      <ColStack className="px-4 py-3 space-y-1">
        <div className="block text-sm text-gray-900 dark:text-white">
          {username}
        </div>
        <div className="block text-sm  text-gray-500 truncate dark:text-gray-400">
          {email}
        </div>
      </ColStack>
      <Separator decorative orientation="horizontal" size="4" />

      <ColStack className="gap-0.5">
        {profileLinks.map((link) => (
          <Link
            href={link.href}
            key={link.href}
            className="w-full mx-auto pl-3  py-1 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
          >
            {link.name}
          </Link>
        ))}
      </ColStack>
      <ul
        className="py-2 flex flex-col items-center"
        aria-labelledby="user-menu-button"
      >
        <li className="w-10/12">
          <label className="inline-flex items-center cursor-pointer">
            <input
              onChange={() => toggleDarkMode()}
              type="checkbox"
              checked={isDarkMode}
              className="sr-only peer"
            />
            <div className="relative w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600 dark:peer-checked:bg-blue-600"></div>
            <span className="ms-3 text-sm font-medium text-gray-900 dark:text-gray-300">
              {dict.darkMode}
            </span>
          </label>
        </li>

        <li className="w-10/12">{/* <LanguageSelect /> */}</li>

        <li className="w-10/12">
          <Button
            onClick={() => {
              logOut();
              signOut();
              clearUser();
            }}
            className="!w-full cursor-pointer px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 dark:text-gray-200 dark:hover:text-white"
          >
            {dict.signOut}
          </Button>
        </li>
      </ul>
    </div>
  );
}
