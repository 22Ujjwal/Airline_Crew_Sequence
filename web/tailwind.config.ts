import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        aa: {
          red: "#b61f23",
          "red-dark": "#8f1619",
          blue: "#0d73b1",
          "blue-light": "#14acdc",
          navy: "#07192e",
          "navy-mid": "#0d2444",
          muted: "#c7d0d7",
          "muted-dark": "#8a9aa6",
        },
      },
      fontFamily: {
        serif: ["Instrument Serif", "Georgia", "serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      backgroundImage: {
        "aa-gradient": "linear-gradient(160deg, #0a1628 0%, #0d2444 40%, #0d3a60 75%, #0a2036 100%)",
        "card-glass": "linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%)",
      },
      animation: {
        "fade-up": "fadeUp 0.6s ease-out forwards",
        "fade-in": "fadeIn 0.8s ease-out forwards",
        drift: "drift 22s ease-in-out infinite",
        fly: "fly 18s cubic-bezier(0.2,0,0.8,1) infinite",
        "pulse-ring": "pulseRing 2s ease-out infinite",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        drift: {
          "0%, 100%": { transform: "translate(0,0) scale(1)" },
          "33%": { transform: "translate(18px,-6px) scale(1.01)" },
          "66%": { transform: "translate(8px,4px) scale(0.99)" },
        },
        fly: {
          "0%": { transform: "translate(-120px,80px) rotate(-3deg)", opacity: "0" },
          "8%": { opacity: "0.9" },
          "92%": { opacity: "0.9" },
          "100%": { transform: "translate(1400px,-60px) rotate(-3deg)", opacity: "0" },
        },
        pulseRing: {
          "0%": { transform: "scale(1)", opacity: "1" },
          "100%": { transform: "scale(1.5)", opacity: "0" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
      boxShadow: {
        glass: "0 4px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1)",
        "glass-lg": "0 8px 64px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08)",
        glow: "0 0 24px rgba(20,172,220,0.25)",
        "glow-red": "0 0 24px rgba(182,31,35,0.3)",
      },
    },
  },
  plugins: [],
};

export default config;
