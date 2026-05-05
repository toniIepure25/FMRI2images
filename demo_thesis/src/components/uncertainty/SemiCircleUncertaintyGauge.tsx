import { useId } from 'react';
import { motion } from 'framer-motion';

const VB_H = 180;
const CX = 150;
const CY = 160;
const R = 120;
const NEEDLE_LEN = R - 22;
const GAUGE_ARC_PATH = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`;
const GAUGE_ARC_LEN = Math.PI * R;
const STROKE_W = 18;

export interface SemiCircleUncertaintyGaugeProps {
  fraction: number;
  centerValue: string;
  percentLabel: string;
  title: string;
  description: string;
  accent: 'cyan' | 'amber';
  animationDelay?: number;
  className?: string;
  /** Extra classes for the SVG element (e.g. larger cinematic size). */
  svgClassName?: string;
}

const accentTitle: Record<SemiCircleUncertaintyGaugeProps['accent'], string> = {
  cyan: 'text-cyan-300/95',
  amber: 'text-amber-300/95',
};

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

export function SemiCircleUncertaintyGauge({
  fraction,
  centerValue,
  percentLabel,
  title,
  description,
  accent,
  animationDelay = 0,
  className = '',
  svgClassName = '',
}: SemiCircleUncertaintyGaugeProps) {
  const uid = useId().replace(/:/g, '');
  const gradId = `u-grad-${uid}`;
  const glowFId = `u-glow-${uid}`;
  const f = clamp01(fraction);
  const dashTarget = `${f * GAUGE_ARC_LEN} ${GAUGE_ARC_LEN}`;

  const accentRing =
    accent === 'cyan'
      ? 'shadow-[0_0_32px_-8px_rgba(34,211,238,0.35)] border-cyan-500/25'
      : 'shadow-[0_0_32px_-8px_rgba(251,191,36,0.3)] border-amber-500/25';

  return (
    <motion.div
      className={`glass-panel flex flex-col items-center border px-5 pb-6 pt-6 ${accentRing} ${className}`}
      initial={{ opacity: 0, y: 22 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.55,
        delay: animationDelay,
        ease: [0.22, 1, 0.36, 1],
      }}
    >
      <p
        className={`text-center text-[11px] font-bold uppercase tracking-[0.2em] ${accentTitle[accent]}`}
        style={{ textShadow: accent === 'cyan' ? '0 0 18px rgba(34,211,238,0.35)' : '0 0 18px rgba(251,191,36,0.3)' }}
      >
        {title}
      </p>

      <svg
        viewBox={`0 0 300 ${VB_H}`}
        className={`mx-auto mt-3 block w-full ${svgClassName ? svgClassName : 'h-[180px] min-w-[280px] max-w-[340px]'}`}
        role="img"
        aria-label={`${title}: ${centerValue}, ${percentLabel}`}
      >
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22c55e" />
            <stop offset="48%" stopColor="#eab308" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
          <filter id={glowFId} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <path
          d={GAUGE_ARC_PATH}
          fill="none"
          stroke="#1e2a4a"
          strokeWidth={STROKE_W}
          strokeLinecap="round"
          opacity={0.95}
        />

        <motion.path
          d={GAUGE_ARC_PATH}
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth={STROKE_W + 6}
          strokeLinecap="round"
          opacity={0.28}
          filter={`url(#${glowFId})`}
          initial={{ strokeDasharray: `0 ${GAUGE_ARC_LEN}` }}
          animate={{ strokeDasharray: dashTarget }}
          transition={{
            duration: 1.35,
            delay: animationDelay + 0.12,
            ease: [0.22, 1, 0.36, 1],
          }}
        />

        <motion.path
          d={GAUGE_ARC_PATH}
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth={STROKE_W}
          strokeLinecap="round"
          filter={`url(#${glowFId})`}
          initial={{ strokeDasharray: `0 ${GAUGE_ARC_LEN}` }}
          animate={{ strokeDasharray: dashTarget }}
          transition={{
            duration: 1.35,
            delay: animationDelay + 0.12,
            ease: [0.22, 1, 0.36, 1],
          }}
        />

        <motion.g
          style={{ transformOrigin: `${CX}px ${CY}px` }}
          initial={{ rotate: 180 }}
          animate={{ rotate: 180 - f * 180 }}
          transition={{
            duration: 1.4,
            delay: animationDelay + 0.1,
            ease: [0.22, 1, 0.36, 1],
          }}
        >
          <line
            x1={CX}
            y1={CY}
            x2={CX + NEEDLE_LEN}
            y2={CY}
            stroke="rgba(248,250,252,0.98)"
            strokeWidth={3}
            strokeLinecap="round"
          />
          <circle cx={CX + NEEDLE_LEN} cy={CY} r={6} fill="#ffffff" filter={`url(#${glowFId})`} />
        </motion.g>

        <circle cx={CX} cy={CY} r={7} fill="#0f172a" stroke="rgba(148,163,184,0.55)" strokeWidth={2} />

        <text
          x={CX}
          y={112}
          textAnchor="middle"
          fill="#ffffff"
          style={{ font: '700 34px ui-sans-serif, system-ui, sans-serif' }}
        >
          {centerValue}
        </text>
        <text
          x={CX}
          y={138}
          textAnchor="middle"
          fill="#94a3b8"
          style={{ font: '500 14px ui-monospace, monospace' }}
        >
          {percentLabel}
        </text>
      </svg>

      <p className="mt-2 max-w-sm text-center text-[12px] leading-relaxed text-slate-400">{description}</p>
    </motion.div>
  );
}
