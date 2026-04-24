import { cn } from "@/lib/utils";
import type { RiskLevel } from "@/types";

interface RiskBadgeProps {
  label: RiskLevel | string;
  score?: number;
  size?: "sm" | "md" | "lg";
}

const sizeStyles = {
  sm: "text-[10px] px-2 py-0.5",
  md: "text-xs px-3 py-1",
  lg: "text-sm px-4 py-1.5",
};

export function RiskBadge({ label, score, size = "md" }: RiskBadgeProps) {
  const isHigh = label === "HIGH RISK" || label === "HIGH";
  const isMod = label === "MODERATE RISK" || label === "MODERATE";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full font-semibold border tracking-wider uppercase",
        sizeStyles[size],
        isHigh && "risk-high",
        isMod && "risk-moderate",
        !isHigh && !isMod && "risk-low"
      )}
    >
      <span className={cn("w-1.5 h-1.5 rounded-full", isHigh ? "bg-red-400" : isMod ? "bg-orange-400" : "bg-emerald-400")} />
      {label}
      {score !== undefined && ` · ${(score * 100).toFixed(1)}%`}
    </span>
  );
}
