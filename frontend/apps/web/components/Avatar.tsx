import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuPortal,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuthStore } from "@/hooks/useAuthStore";
import { languages, useLanguageStore } from "@/hooks/useLanguageStore";
import { LanguagesIcon, LogOutIcon, UserIcon } from "lucide-react";
import { signOut } from "next-auth/react";
import Link from "next/link";
import { Avatar as A, AvatarFallback, AvatarImage } from "./ui/avatar";

export default function Avatar({ username }: { username: string }) {
  const clearUser = useAuthStore((s) => s.clearUser);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <A className="cursor-pointer !border">
          <AvatarImage src="" alt={username} />
          <AvatarFallback className="">
            {username[0]?.toUpperCase()}
          </AvatarFallback>
        </A>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56">
        <DropdownMenuGroup>
          <Link href="/profile">
            <DropdownMenuItem className="cursor-pointer">
              <UserIcon className="text-black dark:text-white" />
              Profile
            </DropdownMenuItem>
          </Link>
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <LanguageGroup />
        <DropdownMenuSeparator />
        <DropdownMenuItem
          className="cursor-pointer"
          onClick={() => {
            signOut();
            clearUser();
          }}
        >
          <LogOutIcon className="text-black dark:text-white" />
          Log out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
const LanguageGroup = () => {
  const { language, setLanguage } = useLanguageStore();
  return (
    <DropdownMenuGroup>
      <DropdownMenuSub>
        <DropdownMenuSubTrigger>
          <LanguagesIcon className="text-black dark:text-white" />
          Language
        </DropdownMenuSubTrigger>
        <DropdownMenuPortal>
          <DropdownMenuSubContent>
            {languages.map((l) => (
              <DropdownMenuItem
                className="cursor-pointer"
                key={l}
                disabled={l === language}
                onClick={() => setLanguage(l)}
              >
                {l}
              </DropdownMenuItem>
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuPortal>
      </DropdownMenuSub>
    </DropdownMenuGroup>
  );
};
