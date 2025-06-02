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
import useSearchParamsBuilder from "@/hooks/useSearchParamsBuilder";
import { HomeIcon, SettingsIcon } from "lucide-react";
import Link from "next/link";

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
  const createQueryString = useSearchParamsBuilder();
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
