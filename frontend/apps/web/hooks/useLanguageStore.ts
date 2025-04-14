import { create } from "zustand";
import { persist } from "zustand/middleware";
import { englishDict } from "./dicts";

export const languages = [
  "English",
  //  "French",
] as const;
type Language = (typeof languages)[number];

export const useLanguageStore = create<{
  language: Language;
  setLanguage: (language: Language) => void;
}>()(
  persist(
    (set) => ({
      language: "English",
      setLanguage: (language) => {
        language = (language.charAt(0).toUpperCase() +
          language.slice(1)) as Language;
        if (languages.includes(language)) {
          console.log({ language });
          set({ language });
        }
      },
    }),
    {
      name: "theme-storage", // localStorage key
    }
  )
);
export function useDictionary() {
  const language = useLanguageStore((s) => s.language);
  switch (language) {
    case "English":
      return englishDict;
    // case "French":
    //   return frenchDict;
    default:
      return englishDict;
  }
}
