import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  accent?: "red" | "blue" | "green" | "orange" | "default";
  delay?: number;
}

const accentStyles: Record<string, string> = {
  red: "border-t-aa-red text-aa-red",
  blue: "border-t-aa-blue-light text-aa-blue-light",
  green: "border-t-emerald-400 text-emerald-400",
  orange: "border-t-orange-400 text-orange-400",
  default: "border-t-white/20 text-white",
};

export function MetricCard({ label, value, sub, accent = "default", delay = 0 }: MetricCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      className={cn("glass rounded-xl p-4 border-t-2", accentStyles[accent])}
    >
      <div className={cn("font-serif text-3xl tracking-tight", accentStyles[accent])}>
        {value}
      </div>
      <div className="text-[10px] font-medium tracking-widest uppercase text-white/40 mt-1">
        {label}
      </div>
      {sub && <div className="text-[11px] text-white/25 mt-0.5">{sub}</div>}
    </motion.div>
  );
}
