/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "media",
  theme: {
    extend: {
      // Only the things with no Tailwind default: the warm palette, the fonts,
      // and the two animations. Everything else uses the default scale.
      colors: {
        ink: "#0a0a0f",
        parchment: { DEFAULT: "#e8e0d4", dim: "#b8b0a4" },
        meta: "#8a7f72",
        rule: "#2a2520",
        gold: "#c8aa78",
      },
      fontFamily: {
        serif: ['"Cormorant Garamond"', "Georgia", "serif"],
        sc: ['"Cormorant SC"', "Georgia", "serif"],
        sans: ['"Instrument Sans"', "system-ui", "sans-serif"],
      },
      keyframes: {
        rise: {
          from: { opacity: "0", transform: "translateY(7px)" },
          to: { opacity: "1", transform: "none" },
        },
        breathe: {
          "0%,100%": { transform: "scale(0.972)", opacity: "0.97" },
          "50%": { transform: "scale(1)", opacity: "1" },
        },
      },
      animation: {
        rise: "rise 0.7s ease both",
        breathe: "breathe 15s ease-in-out infinite",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};
