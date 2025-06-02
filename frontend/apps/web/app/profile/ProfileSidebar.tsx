// export default function ProfileSidePanel({ user }: any) {
//   const fakeUser = {
//     name: "Jane Doe",
//     username: "janedoe",
//     avatar: "https://i.pravatar.cc/150?img=47",
//     bio: "Frontend developer & UI/UX enthusiast. Building cool stuff with React.",
//     location: "San Francisco, CA",
//     website: "https://janedoe.dev",
//     joined: "January 2022",
//   };
//   const router = useRouter();
//   const pathname = usePathname();
//   const searchParams = useSearchParams();
//
//   const sections = [
//     {
//       name: "Overview",
//       icon: "🏠",
//     },
//     {
//       name: "Settings",
//       icon: <Settings />,
//     },
//     {
//       name: "Stats",
//       icon: "⛽",
//     },
//   ];
//   return (
//     <aside className="w-full md:w-1/4 py-6 px-3 dark:bg-gray-900">
//       <div className="flex flex-col space-y-4 w-full">
//         <div className="">
//           <img
//             src={fakeUser.avatar}
//             alt={user.name}
//             className="w-24 h-24 rounded-full object-cover mx-auto"
//           />
//         </div>
//         <div className="pl-3">
//           <h2 className="text-xl font-bold">{user.name}</h2>
//           <p className="text-sm text-gray-500">{user.email}</p>
//         </div>
//         <nav className="mt-3 flex flex-col gap-2 w-full">
//           {sections.map((s) => (
//             <span
//               key={s.name}
//               className="block cursor-pointer font-medium dark:text-gray-300 dark:hover:bg-gray-600 rounded-sm pl-2 py-0.5"
//               onClick={() =>
//                 router.push(
//                   pathname +
//                     "?" +
//                     createQueryString("section", s.name.toLowerCase())
//                 )
//               }
//             >
//               {s.icon} {s.name}
//             </span>
//           ))}
//         </nav>
//       </div>
//     </aside>
//   );
// }

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { HomeIcon, SettingsIcon } from "lucide-react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { useCallback } from "react";

// Menu items.
const items = [
  {
    title: "Overview",
    icon: HomeIcon,
  },
  {
    title: "Settings",
    icon: SettingsIcon,
  },
];

export default function ProfileSidebar({
  selectedSection,
}: {
  selectedSection: Lowercase<string>;
}) {
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const createQueryString = useCallback(
    (name: string, value: string) => {
      const params = new URLSearchParams(searchParams.toString());
      params.set(name, value);
      return pathname + "?" + params.toString();
    },
    [searchParams, pathname]
  );
  return (
    <Sidebar className="mt-[70px]" collapsible="icon">
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel className="font-bold">Profile</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton
                    asChild
                    isActive={selectedSection === item.title.toLowerCase()}
                  >
                    <Link
                      href={createQueryString(
                        "section",
                        item.title.toLowerCase()
                      )}
                    >
                      <item.icon />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
