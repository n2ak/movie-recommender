"use client";
import { getUserInfo } from "@/lib/actions/user";
import { useSession } from "next-auth/react";
import { useEffect } from "react";
import { useAuthStore } from "../../hooks/useAuthStore";

export default function AuthSyncProvider() {
  const { setUser, clearUser, setLoading } = useAuthStore();
  const { data, status } = useSession();
  useEffect(() => {
    if (status === "authenticated") {
      const userId = parseInt(data.user!.id as string);
      (async () => {
        const userInfo = await getUserInfo(userId);
        console.log("New user info", userInfo);
        if (userInfo.data) {
          setUser(userInfo.data);
          setLoading(false);
        }
      })();
    } else if (status === "unauthenticated") {
      clearUser();
      setLoading(false);
    } else if (status === "loading") {
      setLoading(true);
    }
  }, [status, clearUser, setLoading, setUser, data?.user]);

  return null;
}
