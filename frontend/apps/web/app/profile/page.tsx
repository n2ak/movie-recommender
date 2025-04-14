"use client";

import OverviewSection from "@/components/OvervieSection";
import ProfileSidePanel from "@/components/ProfileSidePanel";
import SettingsSection from "@/components/SettingsSection";
import StatsSection from "@/components/StatsSection";
import { useAuthStore } from "@/hooks/useAuthStore";
import { useSearchParams } from "next/navigation";

export default function ProfilePage({}) {
  return <Profile />;
}

type Section = "overview" | "settings" | "stats";
export function Profile() {
  const user = useAuthStore((s) => s.user);
  const params = useSearchParams();
  const section: Section = (params.get("section") as Section) || "overview";
  if (!user) {
    return null;
  }
  function getSelection() {
    switch (section) {
      case "overview":
        return <OverviewSection user={user} />;
      case "settings":
        return <SettingsSection user={user as any} />;
      case "stats":
        return <StatsSection user={user as any} />;
      default:
        return null;
    }
  }
  return (
    <div className="min-h-screen bg-gray-100">
      <div className="flex flex-col md:flex-row max-w-6xl mx-auto">
        <ProfileSidePanel user={user} />
        <main className="flex-1 p-6">{getSelection()}</main>
      </div>
    </div>
  );
}
