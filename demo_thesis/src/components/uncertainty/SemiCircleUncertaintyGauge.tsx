import { useId } from 'react';
import { motion } from 'framer-motion';

export interface SemiCircleUncertaintyGaugeProps {
  fraction: number;
  centerValue: string;
  percentLabel: string;
  title: string;
  description: string;
  accent: 'cyan' | 'amber';
  animationDelay?: number;
  className?: string;
  svgClassName?: string;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

const SIZE = 160;
const STROKE = 12;
const R = (SIZE - STROKE) / 2;
const C = SIZE / 2;
const CIRCUMFERENCE = 2 * Math.PI * R;
const ARC_FRACTION = 0.75;
const ARC_LEN = CIRCUMFERENCE * ARC_FRACTION;
const START_ANGLE = 135;

const ACCENT_COLORS = {
  cyan: {
    ring: 'border-cyan-500/20 shadow-[0_0_24px_-6px_rgba(34,211,238,0.2)]',
    title: 'text-cyan-300',
    stroke: '#22d3ee',
    glow: 'rgba(34,211,238,0.4)',
    trackStroke: 'rgba(34,211,238,0.08)',
  },
  amber: {
    ring: 'border-amber-500/20 shadow-[0_0_24px_-6px_rgba(251,191,36,0.2)]',
    title: 'text-amber-300',
    stroke: '#fbbf24',
    glow: 'rgba(251,191,36,0.4)',
    trackStroke: 'rgba(251,191,36,0.08)',
  },
};

export function SemiCircleUncertaintyGauge({
  fraction,
  centerValue,
  percentLabel,
  title,
  description,
  accent,
  animationDelay = 0,
  className = '',
}: SemiCircleUncertaintyGaugeProps) {
  const uid = useId().replace(/:/g, '');
  const gradId = `gauge-grad-${uid}`;
  const f = clamp01(fraction);
  const filledLen = f * ARC_LEN;
  const gap = CIRCUMFERENCE - ARC_LEN;
  const colors = ACCENT_COLORS[accent];

  return (
    <motion.div
      className={`glass-panel flex flex-col items-center border px-6 py-6 ${colors.ring} ${className}`}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: animationDelay, ease: [0.22, 1, 0.36, 1] }}
    >
      <p className={`text-[10px] font-bold uppercase tracking-[0.22em] ${colors.title}`}>
        {title}
      </p>

      <div className="relative mt-4 mb-2" style={{ width: SIZE, height: SIZE }}>
        <svg
          width={SIZE}
          height={SIZE}
          viewBox={`0 0 ${SIZE} ${SIZE}`}
          className="block"
          role="img"
          aria-label={`${title}: ${centerValue}, ${percentLabel}`}
        >
          <defs>
            <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={colors.stroke} stopOpacity="0.3" />
              <stop offset="100%" stopColor={colors.stroke} />
            </linearGradient>
          </defs>

          {/* Track */}
          <circle
            cx={C}
            cy={C}
            r={R}
            fill="none"
            stroke={colors.trackStroke}
            strokeWidth={STROKE}
            strokeDasharray={`${ARC_LEN} ${gap}`}
            strokeDashoffset={-gap / 2}
            strokeLinecap="round"
            transform={`rotate(${START_ANGLE} ${C} ${C})`}
          />

          {/* Filled arc */}
          <motion.circle
            cx={C}
            cy={C}
            r={R}
            fill="none"
            stroke={`url(#${gradId})`}
            strokeWidth={STROKE}
            strokeLinecap="round"
            strokeDasharray={`${ARC_LEN} ${gap}`}
            strokeDashoffset={-gap / 2}
            transform={`rotate(${START_ANGLE} ${C} ${C})`}
            initial={{ strokeDasharray: `0 ${CIRCUMFERENCE}`, strokeDashoffset: -gap / 2 }}
            animate={{ strokeDasharray: `${filledLen} ${CIRCUMFERENCE - filledLen}`, strokeDashoffset: -gap / 2 }}
            transition={{ duration: 1.2, delay: animationDelay + 0.15, ease: [0.22, 1, 0.36, 1] }}
            style={{ filter: `drop-shadow(0 0 6px ${colors.glow})` }}
          />
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="font-mono text-2xl font-bold text-white"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: animationDelay + 0.3 }}
          >
            {centerValue}
          </motion.span>
          <motion.span
            className="mt-0.5 font-mono text-xs text-slate-400"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4, delay: animationDelay + 0.45 }}
          >
            {percentLabel}
          </motion.span>
        </div>
      </div>

      <p className="max-w-[220px] text-center text-[11px] leading-relaxed text-slate-500">{description}</p>
    </motion.div>
  );
}
