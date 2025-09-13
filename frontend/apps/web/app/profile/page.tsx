"use client";

import { SidebarProvider } from "@/components/ui/sidebar";
import { useAuthStore } from "@/hooks/useAuthStore";
import { useSearchParams } from "next/navigation";
import type { PropsWithChildren } from "react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../../components/ui/card";
import OverviewSection from "./OverviewSection";
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
        return {
          title: "Profile Overview",
          section: <OverviewSection user={user} />,
        };
      case "settings":
        return { title: "Settings", section: <SettingsSection user={user} /> };
      case "stats":
        return { title: "Stats", section: <StatsSection /> };
      default:
        return null;
    }
  }
  const selection = getSelection();
  return (
    <div className="h-full">
      <SidebarProvider>
        <div className="flex flex-col md:flex-row max-w-6xl h-full w-full">
          <ProfileSidebar selectedSection={section} />
          <main className="w-full">
            {selection && (
              <MainSection title={selection.title}>
                {selection.section}
              </MainSection>
            )}
          </main>
        </div>
      </SidebarProvider>
    </div>
  );
}

function MainSection({
  title,
  children,
}: PropsWithChildren<{
  title: string;
}>) {
  return (
    <>
      <Card className="gap-2 dark:bg-secondary-foreground p-4 rounded-lg shadow-sm">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold mb-4">{title}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">{children}</CardContent>
      </Card>
    </>
  );
}
