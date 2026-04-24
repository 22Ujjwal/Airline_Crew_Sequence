"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart3, Home } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/", label: "Home", icon: <Home className="w-3.5 h-3.5" /> },
  { href: "/dashboard", label: "Dashboard", icon: <BarChart3 className="w-3.5 h-3.5" /> },
];

export function Navbar() {
  const pathname = usePathname();
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center px-6 border-b border-white/[0.06] backdrop-blur-md bg-aa-navy/80"
    >
      {/* Logo */}
      <Link href="/" className="flex items-center gap-2.5 mr-8">
        <div className="relative w-6 h-6">
          <svg viewBox="0 0 24 24" className="w-full h-full">
            <path
              d="M2 12 L8 5 L14 12 L20 7 L22 12 L12 20 Z"
              fill="#b61f23"
              opacity="0.9"
            />
          </svg>
        </div>
        <span className="text-sm font-semibold text-white/90 tracking-wide">AA Crew Risk</span>
      </Link>

      {/* Nav links */}
      <nav className="flex items-center gap-1">
        {NAV_LINKS.map((l) => {
          const active = pathname === l.href || (l.href !== "/" && pathname.startsWith(l.href));
          return (
            <Link
              key={l.href}
              href={l.href}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors",
                active
                  ? "text-aa-blue-light bg-aa-blue-light/10"
                  : "text-white/50 hover:text-white/80 hover:bg-white/5"
              )}
            >
              {l.icon}
              {l.label}
            </Link>
          );
        })}
      </nav>

      {/* Right side: model badge */}
      <div className="ml-auto flex items-center gap-3">
        <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-full border border-white/8 bg-white/4">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[10px] text-white/40 tracking-wider uppercase">XGBoost v7</span>
        </div>
        <div className="text-[10px] text-white/25 hidden md:block">AUC 0.825</div>
      </div>
    </motion.header>
  );
}
