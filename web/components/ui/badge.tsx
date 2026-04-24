import * as React from "react";
import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "high" | "moderate" | "low" | "blue" | "outline";
}

const variantStyles: Record<string, string> = {
  default: "bg-white/10 text-white/70 border-white/15",
  high: "bg-red-500/15 text-red-400 border-red-500/30",
  moderate: "bg-orange-500/15 text-orange-400 border-orange-500/30",
  low: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
  blue: "bg-aa-blue-light/15 text-aa-blue-light border-aa-blue-light/30",
  outline: "bg-transparent text-white/50 border-white/15",
};

export function Badge({ variant = "default", className, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-0.5 text-[10px] font-semibold tracking-wider uppercase",
        variantStyles[variant],
        className
      )}
      {...props}
    />
  );
}
