"use client";

import { SidebarProvider } from "@/components/ui/sidebar";
import { useAuthStore } from "@/hooks/useAuthStore";
import { useSearchParams } from "next/navigation";
import OverviewSection from "./OvervieSection";
import ProfileSidebar from "./ProfileSidebar";
import SettingsSection from "./SettingsSection";
import StatsSection from "./StatsSection";

export default function ProfilePage() {
  return <Profile />;
}

type Section = "overview" | "settings" | "stats";
function Profile() {
  const user = useAuthStore((s) => s.user);
  const params = useSearchParams();
  const section = (params.get("section") as Section) || "overview";
  if (!user) {
    // TODO
    return null;
  }
  function getSelection() {
    if (!user) return null;
    switch (section) {
      case "overview":
        return <OverviewSection user={user} />;
      case "settings":
        return <SettingsSection user={user} />;
      case "stats":
        return <StatsSection />;
      default:
        return null;
    }
  }
  return (
    <div className="h-full">
      <SidebarProvider>
        <div className="flex flex-col md:flex-row max-w-6xl mx-auto h-full">
          <ProfileSidebar selectedSection={section} />
          <main className="flex-1 p-6">{getSelection()}</main>
        </div>
      </SidebarProvider>
    </div>
  );
}
