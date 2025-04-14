import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      backgroundImage: {
        back: "linear-gradient(to left bottom, #393e46, #544a60, #7f5269, #a55d5d, #b37645);",
      },
      colors: {
        "1": "#222831",
        "2": "#393e46",
        "3": "#f96d00",
        "4": "#f2f2f2",
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
