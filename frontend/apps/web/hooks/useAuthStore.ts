// import { User } from "next-auth";
import { create } from "zustand";
// import type { User } from "@repo/database";
export interface UserInfo {
  id: number;
  email: string;
  username: string;
  createdAt: Date;
}
interface Auth {
  user: UserInfo | null;
  token: string | null;
  loading: boolean | null;
  setUser: (user: UserInfo | null) => void;
  setToken: (token: string | null) => void;
  setLoading: (user: boolean | null) => void;
  clearUser: () => void;
}

export const useAuthStore = create<Auth>((set) => ({
  user: null,
  token: null,
  loading: false,
  setUser: (user) => set({ user, loading: false }),
  setToken: (token) => set({ token }),
  setLoading: (loading) => set({ loading }),
  clearUser: () => {
    set({ user: null, token: null, loading: false });
  },
}));
