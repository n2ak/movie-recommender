import { create } from "zustand";
import { persist } from "zustand/middleware";

interface UIState {
  isDarkMode: boolean;
  toggleDarkMode: () => void;
  __loaded: boolean;
  __setLoaded: (hydrated: boolean) => void;
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      isDarkMode: false,
      __loaded: false,
      toggleDarkMode: () => set((state) => ({ isDarkMode: !state.isDarkMode })),
      __setLoaded: (hydrated) => set({ __loaded: hydrated }),
    }),
    {
      name: "theme-storage", // localStorage key
      onRehydrateStorage: () => (state) => {
        state?.__setLoaded(true);
      },
      partialize: (state) => ({
        isDarkMode: state.isDarkMode,
      }),
    }
  )
);
