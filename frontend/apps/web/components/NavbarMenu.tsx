"use client";
import { logOut } from "@/_lib/actions/user";
import { useAuthStore } from "@/hooks/useAuthStore";
import { useOnClickOutside } from "@/hooks/useClickedOutside";
import { useDictionary } from "@/hooks/useLanguageStore";
import { useUIStore } from "@/hooks/useUIStore";
import { signOut } from "next-auth/react";
import Link from "next/link";
import * as React from "react";
import Button from "./Button";
import { RowStack } from "./Container";
import LanguageSelect from "./LanguageSelect";

export default function NavbarMenu({
  menuOpen,
  setMenuOpen,
  username,
  email,
  avatarRef,
}: {
  menuOpen: boolean;
  username: string;
  email: string;
  setMenuOpen: (open: boolean) => void;
  avatarRef: React.RefObject<HTMLElement | null>;
}) {
  const dict = useDictionary();
  const { isDarkMode, toggleDarkMode } = useUIStore();
  const clearUser = useAuthStore((s) => s.clearUser);
  const ref = React.useRef<HTMLDivElement>(null);
  useOnClickOutside(ref, () => {
    setMenuOpen(false);
  }, [avatarRef]);
  if (!menuOpen) return null;
  const profileLinks = [
    {
      href: "/profile",
      name: "Profile",
    },
    {
      href: "/profile?section=settings",
      name: "Settings",
    },
  ];
  return (
    <div
      ref={ref}
      className="flex flex-col gap-2 top-[100%] right-0 absolute max-w-60 min-w-50 rounded-md bg-gray-100 border border-gray-900/30 dark:border-white/30 dark:bg-slate-950 px-2 py-2 text-sm font-medium"
    >
      <div className="flex flex-row gap-3 pl-2">
        <div className="w-10 h-10 bg-white">
          <img className="" />
        </div>
        <div className="flex flex-col justify-evenly">
          <div className="text-gray-400 dark:text-gray-200/60 text-xs">
            {username}
          </div>
          <div className="text-gray-400 dark:text-gray-200/60 text-xs">
            {email}
          </div>
        </div>
      </div>
      <div className="mx-auto h-[1px] w-[50%] bg-gray-900/30 dark:bg-white/30" />
      <div>
        {profileLinks.map((link) => (
          <div
            key={link.name}
            className="cursor-pointer rounded-sm py-1 pl-2 dark:text-gray-200 dark:hover:bg-gray-700 text-gray-900 hover:bg-gray-300"
          >
            <Link href={link.href}>{link.name}</Link>
          </div>
        ))}
      </div>

      <div className="mx-auto h-[1px] w-[50%] bg-gray-900/30 dark:bg-white/30" />
      <label className="mx-auto inline-flex items-center cursor-pointer">
        <input
          onChange={() => toggleDarkMode()}
          type="checkbox"
          checked={isDarkMode}
          className="sr-only peer"
        />
        <div className="relative w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white  after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600 dark:peer-checked:bg-blue-600 focus:outline-hidden"></div>
        <span className="ms-3 text-sm font-medium dark:text-gray-200 text-gray-900">
          {dict.darkMode}
        </span>
      </label>
      <RowStack className="gap-1 items-center">
        <span className="h-full text-center items-center">Language: </span>
        <LanguageSelect />
      </RowStack>
      <div className="text-center">
        <Button
          className="cursor-pointer rounded-sm py-1 text-center mx-auto"
          onClick={() => {
            logOut();
            signOut();
            clearUser();
          }}
        >
          Sign Out
        </Button>
      </div>
    </div>
  );
}
