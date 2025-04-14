import { Settings as SettingsIcon } from "@mui/icons-material";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback } from "react";

export default function ProfileSidePanel({ user }: any) {
  const fakeUser = {
    name: "Jane Doe",
    username: "janedoe",
    avatar: "https://i.pravatar.cc/150?img=47",
    bio: "Frontend developer & UI/UX enthusiast. Building cool stuff with React.",
    location: "San Francisco, CA",
    website: "https://janedoe.dev",
    joined: "January 2022",
  };
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const createQueryString = useCallback(
    (name: string, value: string) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set(name, value);

      return params.toString();
    },
    [searchParams]
  );
  const sections = [
    {
      name: "Overview",
      icon: "🏠",
    },
    {
      name: "Settings",
      icon: <SettingsIcon />,
    },
    {
      name: "Stats",
      icon: "⛽",
    },
  ];
  return (
    <aside className="w-full md:w-1/4 bg-white p-6 border-r">
      <div className="flex flex-col items-center md:items-start space-y-4">
        <img
          src={fakeUser.avatar}
          alt={user.name}
          className="w-24 h-24 rounded-full object-cover"
        />
        <div className="text-center md:text-left">
          <h2 className="text-xl font-semibold">{user.name}</h2>
          <p className="text-sm text-gray-500">@{user.name}</p>
        </div>
        <nav className="mt-6 space-y-2 w-full">
          {sections.map((s) => (
            <span
              key={s.name}
              className="block cursor-pointer text-gray-700 hover:text-blue-600"
              onClick={() =>
                router.push(
                  pathname +
                    "?" +
                    createQueryString("section", s.name.toLowerCase())
                )
              }
            >
              {s.icon} {s.name}
            </span>
          ))}
        </nav>
      </div>
    </aside>
  );
}
