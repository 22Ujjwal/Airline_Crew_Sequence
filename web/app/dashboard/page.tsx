"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BarChart3, Search, Zap, BookOpen } from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { DashboardTab } from "@/components/tabs/DashboardTab";
import { QueryTab } from "@/components/tabs/QueryTab";
import { OptimizerTab } from "@/components/tabs/OptimizerTab";
import { MethodologyTab } from "@/components/tabs/MethodologyTab";
import { cn } from "@/lib/utils";

const TABS = [
  { id: "dashboard", label: "Risk Dashboard", icon: <BarChart3 className="w-4 h-4" />, component: DashboardTab },
  { id: "query", label: "Pair Query", icon: <Search className="w-4 h-4" />, component: QueryTab },
  { id: "optimizer", label: "Optimizer", icon: <Zap className="w-4 h-4" />, component: OptimizerTab },
  { id: "methodology", label: "Methodology", icon: <BookOpen className="w-4 h-4" />, component: MethodologyTab },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabId>("dashboard");

  const ActiveComponent = TABS.find((t) => t.id === activeTab)!.component;

  return (
    <div className="min-h-screen bg-aa-navy">
      {/* Cloud background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[#0a1628] via-[#0d2444] to-[#0a2036]" />
        <div className="absolute top-0 left-1/4 w-[800px] h-[300px] bg-aa-blue/[0.04] rounded-full blur-[80px]" />
        <div className="absolute bottom-1/3 right-1/4 w-[600px] h-[200px] bg-aa-blue-light/[0.04] rounded-full blur-[60px]" />
      </div>

      <Navbar />

      <div className="relative z-10 pt-14">
        {/* Tab bar */}
        <div className="sticky top-14 z-40 border-b border-white/[0.06] backdrop-blur-md bg-aa-navy/70">
          <div className="max-w-screen-xl mx-auto px-6">
            <div className="flex items-center gap-1 overflow-x-auto hide-scrollbar">
              {TABS.map((tab) => {
                const active = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={cn(
                      "relative flex items-center gap-2 px-4 py-3.5 text-sm font-medium transition-colors whitespace-nowrap",
                      active ? "text-aa-blue-light" : "text-white/40 hover:text-white/70"
                    )}
                  >
                    {tab.icon}
                    {tab.label}
                    {active && (
                      <motion.div
                        layoutId="tab-underline"
                        className="absolute bottom-0 left-0 right-0 h-[2px] bg-aa-blue-light"
                        transition={{ type: "spring", stiffness: 400, damping: 30 }}
                      />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Page header */}
        <div className="max-w-screen-xl mx-auto px-6 pt-8 pb-4">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <h1 className="font-serif text-3xl text-white">
              {TABS.find((t) => t.id === activeTab)?.label}
            </h1>
            <p className="text-sm text-white/35 mt-1">
              {activeTab === "dashboard" && "Airport pair risk scores · A→DFW→B tail-matched rotations · BTS 2015–2024"}
              {activeTab === "query" && "Calibrated risk score + SHAP explanation for any origin → DFW → destination pair"}
              {activeTab === "optimizer" && "Hungarian algorithm minimizes total weather risk across a day's DFW schedule"}
              {activeTab === "methodology" && "Technical report: data pipeline, 70 features, XGBoost v7, isotonic calibration, evaluation"}
            </p>
          </motion.div>
        </div>

        {/* Tab content */}
        <div className="max-w-screen-xl mx-auto px-6 pb-12">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
              <ActiveComponent />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Bottom accent bar */}
      <div className="fixed bottom-0 left-0 right-0 h-[4px] aa-accent-bar z-50" />
    </div>
  );
}
