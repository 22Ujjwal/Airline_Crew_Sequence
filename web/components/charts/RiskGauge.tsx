"use client";

import { useMemo } from "react";
import { scoreToColor, riskLabel } from "@/lib/utils";
import { RiskBadge } from "@/components/common/RiskBadge";

interface RiskGaugeProps {
  score: number;
  size?: number;
}

export function RiskGauge({ score, size = 200 }: RiskGaugeProps) {
  const label = riskLabel(score);
  const color = useMemo(() => scoreToColor(score), [score]);

  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.38;
  const strokeW = size * 0.075;

  // Arc from 210° to 330° (240° sweep = 2/3 of circle)
  const startAngle = (210 * Math.PI) / 180;
  const endAngle = (330 * Math.PI) / 180;
  const totalSweep = (300 * Math.PI) / 180; // 300°

  const polarToXY = (angle: number, radius: number) => ({
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  });

  const describeArc = (start: number, end: number, rad: number) => {
    const s = polarToXY(start, rad);
    const e = polarToXY(end, rad);
    const largeArc = Math.abs(end - start) > Math.PI ? 1 : 0;
    return `M ${s.x} ${s.y} A ${rad} ${rad} 0 ${largeArc} 1 ${e.x} ${e.y}`;
  };

  // Track arc (full background)
  const trackPath = describeArc(startAngle, startAngle + totalSweep, r);

  // Fill arc (proportional to score)
  const fillSweep = Math.min(1, score / 0.5) * totalSweep;
  const fillPath = fillSweep > 0.01
    ? describeArc(startAngle, startAngle + fillSweep, r)
    : "";

  // Zone arcs
  const zoneLow = describeArc(startAngle, startAngle + (0.2 / 0.5) * totalSweep, r);
  const zoneMid = describeArc(startAngle + (0.2 / 0.5) * totalSweep, startAngle + (0.3 / 0.5) * totalSweep, r);

  // Needle
  const needleAngle = startAngle + Math.min(1, score / 0.5) * totalSweep;
  const needleTip = polarToXY(needleAngle, r * 0.85);
  const needleBase = polarToXY(needleAngle + Math.PI * 0.05, r * 0.15);
  const needleBase2 = polarToXY(needleAngle - Math.PI * 0.05, r * 0.15);

  return (
    <div className="flex flex-col items-center gap-3">
      <svg width={size} height={size * 0.7} viewBox={`0 0 ${size} ${size * 0.7}`}>
        {/* Track */}
        <path
          d={trackPath}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
        {/* Low zone (green) */}
        <path
          d={zoneLow}
          fill="none"
          stroke="rgba(34,197,94,0.15)"
          strokeWidth={strokeW}
          strokeLinecap="round"
        />
        {/* Moderate zone (orange) */}
        <path
          d={zoneMid}
          fill="none"
          stroke="rgba(249,115,22,0.15)"
          strokeWidth={strokeW}
        />
        {/* Fill */}
        {fillPath && (
          <path
            d={fillPath}
            fill="none"
            stroke={color}
            strokeWidth={strokeW}
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 8px ${color}55)` }}
          />
        )}
        {/* Needle */}
        <polygon
          points={`${needleTip.x},${needleTip.y} ${needleBase.x},${needleBase.y} ${needleBase2.x},${needleBase2.y}`}
          fill="rgba(255,255,255,0.85)"
        />
        <circle cx={cx} cy={cy} r={strokeW * 0.55} fill="rgba(255,255,255,0.15)" />

        {/* Score text */}
        <text
          x={cx}
          y={cy + r * 0.1}
          textAnchor="middle"
          fill="white"
          fontSize={size * 0.14}
          fontFamily="Instrument Serif, serif"
          letterSpacing="-0.03em"
        >
          {(score * 100).toFixed(1)}%
        </text>
      </svg>

      <RiskBadge label={label} size="md" />
    </div>
  );
}
