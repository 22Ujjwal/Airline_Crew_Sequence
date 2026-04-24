import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { RiskLevel } from "@/types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtPct(v: number, decimals = 1) {
  return `${(v * 100).toFixed(decimals)}%`;
}

export function fmtScore(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export function scoreToColor(score: number): string {
  const CAP = 0.5;
  const t = Math.max(0, Math.min(1, score / CAP));
  let r: number, g: number, b: number;
  if (t <= 0.5) {
    const s = t * 2;
    r = Math.round(44 + s * (255 - 44));
    g = Math.round(160 + s * (200 - 160));
    b = Math.round(44 + s * (0 - 44));
  } else {
    const s = (t - 0.5) * 2;
    r = Math.round(255 + s * (214 - 255));
    g = Math.round(200 + s * (39 - 200));
    b = Math.round(0 + s * 40);
  }
  return `rgb(${r},${g},${b})`;
}

export function riskLabel(score: number): RiskLevel {
  if (score >= 0.3) return "HIGH RISK";
  if (score >= 0.2) return "MODERATE RISK";
  return "LOW RISK";
}

export function riskClass(label: RiskLevel | string) {
  if (label === "HIGH RISK") return "risk-high";
  if (label === "MODERATE RISK") return "risk-moderate";
  return "risk-low";
}

export function riskHex(label: RiskLevel | string) {
  if (label === "HIGH RISK") return "#ef4444";
  if (label === "MODERATE RISK") return "#f97316";
  return "#22c55e";
}

export function debounce<T extends (...args: unknown[]) => void>(fn: T, ms: number) {
  let timer: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
