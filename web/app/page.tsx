"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, TrendingUp, Zap, Target, BarChart3 } from "lucide-react";

const STATS = [
  { label: "Val AUC", value: "0.825", sub: "2024 holdout" },
  { label: "Sequences", value: "398K", sub: "tail-matched rotations" },
  { label: "Features", value: "70", sub: "across 8 groups" },
  { label: "Base Rate", value: "11.4%", sub: "positive class" },
];

const FEATURES = [
  {
    icon: <BarChart3 className="w-5 h-5" />,
    title: "Risk Heatmap",
    desc: "Top 15 origin airports × 12 months. Spot the hottest corridors instantly.",
  },
  {
    icon: <Target className="w-5 h-5" />,
    title: "Pair Query + SHAP",
    desc: "Pick any A→DFW→B pair and month. Get a calibrated score and full SHAP waterfall.",
  },
  {
    icon: <Zap className="w-5 h-5" />,
    title: "Sequence Optimizer",
    desc: "Hungarian algorithm minimizes total weather risk across a day's schedule.",
  },
  {
    icon: <TrendingUp className="w-5 h-5" />,
    title: "Methodology",
    desc: "Full technical report: data pipeline, 70 features, XGBoost v7, isotonic calibration.",
  },
];

export default function HeroPage() {
  return (
    <main className="relative min-h-screen overflow-hidden bg-aa-navy">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#0a1628] via-[#0d2444] to-[#0a2036]" />

      {/* Animated cloud blobs */}
      <motion.div
        className="absolute top-[5%] left-[-5%] w-[700px] h-[220px] pointer-events-none"
        animate={{ x: [0, 18, 8, 0], y: [0, -6, 4, 0] }}
        transition={{ duration: 22, repeat: Infinity, ease: "easeInOut" }}
      >
        <svg viewBox="0 0 700 220" className="w-full h-full">
          <filter id="b1">
            <feGaussianBlur stdDeviation="28" />
          </filter>
          <ellipse cx="250" cy="110" rx="220" ry="75" fill="rgba(255,255,255,0.055)" filter="url(#b1)" />
          <ellipse cx="480" cy="90" rx="170" ry="60" fill="rgba(255,255,255,0.04)" filter="url(#b1)" />
        </svg>
      </motion.div>

      <motion.div
        className="absolute top-[35%] right-[-2%] w-[600px] h-[200px] pointer-events-none"
        animate={{ x: [0, -14, 10, 0], y: [0, 8, -4, 0] }}
        transition={{ duration: 28, repeat: Infinity, ease: "easeInOut", delay: -8 }}
      >
        <svg viewBox="0 0 600 200" className="w-full h-full">
          <filter id="b2">
            <feGaussianBlur stdDeviation="22" />
          </filter>
          <ellipse cx="300" cy="100" rx="260" ry="70" fill="rgba(20,172,220,0.055)" filter="url(#b2)" />
          <ellipse cx="120" cy="80" rx="130" ry="50" fill="rgba(13,115,177,0.04)" filter="url(#b2)" />
        </svg>
      </motion.div>

      {/* Animated airplane — bottom-left third → top-right third */}
      <motion.div
        className="absolute top-[65%] left-0 pointer-events-none z-10"
        animate={{ x: [-320, 1600], y: [0, -380] }}
        transition={{ duration: 20, repeat: Infinity, ease: [0.2, 0, 0.8, 1], delay: 2 }}
        style={{ rotate: -12 }}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src="/airplane.png"
          alt=""
          width={320}
          height={245}
          style={{ opacity: 0.85 }}
        />
      </motion.div>

      {/* Content */}
      <div className="relative z-20 flex flex-col items-center justify-center min-h-screen px-6 text-center">
        {/* Eyebrow */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex items-center gap-3 mb-8"
        >
          <div className="h-px w-12 bg-aa-blue-light/50" />
          <span className="text-xs font-medium tracking-[0.22em] uppercase text-aa-blue-light/80">
            American Airlines · EPPS Data Challenge
          </span>
          <div className="h-px w-12 bg-aa-blue-light/50" />
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.25 }}
          className="font-serif text-6xl md:text-7xl lg:text-8xl leading-none tracking-tight text-white mb-4 max-w-4xl"
        >
          Risk Prediction
          <br />
          <em className="text-gradient">for Crew Sequencing</em>
        </motion.h1>

        {/* Subtext */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.45 }}
          className="text-aa-muted/60 text-base md:text-lg font-light max-w-2xl mt-6 mb-10 leading-relaxed"
        >
          XGBoost v7 + isotonic calibration. Scores ≈ observed disruption rates.
          <br className="hidden md:block" />
          A→DFW→B tail-matched sequences · 10 years BTS · 70 features.
        </motion.p>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          className="flex flex-col sm:flex-row gap-4 mb-20"
        >
          <Link
            href="/dashboard"
            className="group inline-flex items-center gap-2.5 bg-aa-red hover:bg-aa-red-dark text-white text-sm font-medium px-8 py-3.5 rounded-full transition-all duration-200 shadow-glow-red hover:shadow-none hover:scale-[0.98]"
          >
            Open Dashboard
            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </Link>
          <a
            href="https://github.com/Optimax14/AA-EPPS-Data-Challenge"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2.5 glass glass-hover text-white/70 hover:text-white text-sm font-medium px-8 py-3.5 rounded-full transition-all duration-200"
          >
            GitHub Repo
          </a>
        </motion.div>

        {/* Stats row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.75 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-3xl mb-24"
        >
          {STATS.map((s) => (
            <div key={s.label} className="glass rounded-2xl p-4 text-center">
              <div className="font-serif text-3xl md:text-4xl text-aa-blue-light tracking-tight">
                {s.value}
              </div>
              <div className="text-[10px] font-medium tracking-widest uppercase text-white/40 mt-1">
                {s.label}
              </div>
              <div className="text-[11px] text-white/30 mt-0.5">{s.sub}</div>
            </div>
          ))}
        </motion.div>

        {/* Feature grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1 }}
          className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 w-full max-w-4xl"
        >
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 1 + i * 0.1 }}
              className="glass glass-hover rounded-xl p-5 text-left"
            >
              <div className="text-aa-blue-light mb-3">{f.icon}</div>
              <div className="text-sm font-semibold text-white mb-1.5">{f.title}</div>
              <div className="text-xs text-white/40 leading-relaxed">{f.desc}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Bottom AA red accent bar */}
      <div className="fixed bottom-0 left-0 right-0 h-[5px] aa-accent-bar z-50" />
    </main>
  );
}
