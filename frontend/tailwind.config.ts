import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        void: "#0D1210",
        panel: "#141B19",
        panel2: "#1B2422",
        hairline: "#2A3532",
        steel: "#7C8B88",
        steel2: "#A8B5B2",
        paper: "#ECEDE9",
        paperPanel: "#DDDFD9",
        signal: "#33FFB2",
        signalDim: "#1C8F66",
        amber: "#FFB020",
        red: "#FF5C5C",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      borderRadius: {
        none: "0px",
        sm: "2px",
        DEFAULT: "3px",
      },
    },
  },
  plugins: [],
};

export default config;
