"use client";
import { useSession } from "next-auth/react";
import { useEffect } from "react";
import { useAuthStore } from "../../hooks/useAuthStore";

export default function AuthSyncProvider() {
  const { setUser, clearUser, setLoading } = useAuthStore();
  const { data, status } = useSession();
  useEffect(() => {
    if (status === "authenticated") {
      setUser({
        name: data.user!.name as string,
        email: data.user!.email as string,
        id: parseInt(data.user!.id as string),
      });
      setLoading(false);
    } else if (status === "unauthenticated") {
      clearUser();
    } else if (status === "loading") {
      setLoading(true);
    }
  }, [status, data]);
  return null;
}
