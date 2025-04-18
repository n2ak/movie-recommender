import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      backgroundImage: {
        back: "linear-gradient(to left bottom, #393e46, #544a60, #7f5269, #a55d5d, #b37645);",
      },
      colors: {
        "1": "#211C84",
        "2": "#4D55CC",
        "3": "#7A73D1",
        "4": "#B5A8D5",
      },
    },
    fontFamily: {
      sans: ["Graphik", "sans-serif"],
      serif: ["Merriweather", "serif"],
      noto: ["Noto Sans"],
    },
  },
  plugins: [],
};
export default config;
