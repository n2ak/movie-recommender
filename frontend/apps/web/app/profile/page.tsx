"use client";

import OverviewSection from "@/components/OvervieSection";
import ProfileSidePanel from "@/components/ProfileSidePanel";
import RecentLoginActivity from "@/components/RecentLoginActivity";
import SettingsSection from "@/components/SettingsSection";
import StatsSection from "@/components/StatsSection";
import { useAuthStore } from "@/hooks/useAuthStore";
import { useSearchParams } from "next/navigation";

export default function ProfilePage() {
  return <Profile />;
}

type Section = "loginactivity" | "overview" | "settings" | "stats";
function Profile() {
  const user = useAuthStore((s) => s.user);
  const params = useSearchParams();
  const section = (params.get("section") as Section) || "overview";
  if (!user) {
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
        return <StatsSection user={user} />;
      case "loginactivity":
        return <RecentLoginActivity />;
      default:
        return null;
    }
  }
  return (
    <div className="h-full">
      <div className="flex flex-col md:flex-row max-w-6xl mx-auto h-full">
        <ProfileSidePanel user={user} />
        <main className="flex-1 p-6">{getSelection()}</main>
      </div>
    </div>
  );
}
