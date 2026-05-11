import { motion } from 'framer-motion';
import type { Provenance } from '@/lib/provenance';
import { ProvenanceBadge } from './ProvenanceBadge';
import { isMetricAvailable } from '@/lib/metrics';

interface ParameterBarProps {
  label: string;
  value: number | null | undefined;
  normalized?: number | null;
  format?: (v: number) => string;
  description?: string;
  interpretation?: string;
  provenance: Provenance;
  tone?: 'cyan' | 'violet' | 'emerald' | 'amber' | 'rose' | 'slate';
}

const BAR_COLORS: Record<string, { bg: string; text: string }> = {
  cyan:    { bg: 'bg-cyan-400/40',    text: 'text-cyan-300'    },
  violet:  { bg: 'bg-violet-400/40',  text: 'text-violet-300'  },
  emerald: { bg: 'bg-emerald-400/40', text: 'text-emerald-300' },
  amber:   { bg: 'bg-amber-400/35',   text: 'text-amber-300'   },
  rose:    { bg: 'bg-rose-400/35',    text: 'text-rose-300'    },
  slate:   { bg: 'bg-slate-400/35',   text: 'text-slate-300'   },
};

export function ParameterBar({
  label,
  value,
  normalized,
  format,
  description,
  interpretation,
  provenance,
  tone = 'cyan',
}: ParameterBarProps) {
  const available = isMetricAvailable(value);
  const frac =
    isMetricAvailable(normalized) ? Math.max(0, Math.min(1, normalized)) : null;
  const colors = BAR_COLORS[tone];

  return (
    <div className="rounded-xl bg-white/[0.025] p-4">
      <div className="flex items-center justify-between gap-2">
        <p className={`text-[10px] font-medium ${colors.text}`}>
          {label}
        </p>
        <ProvenanceBadge provenance={provenance} />
      </div>
      <p
        className={`mt-1.5 font-mono text-xl font-semibold tabular-nums ${
          available ? 'text-white' : 'text-slate-600'
        }`}
      >
        {available
          ? format
            ? format(value!)
            : value! >= 100
              ? value!.toFixed(1)
              : value!.toFixed(3)
          : '—'}
      </p>
      {frac != null && (
        <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-white/[0.04]">
          <motion.div
            className={`h-full rounded-full ${colors.bg}`}
            initial={{ width: '0%' }}
            animate={{ width: `${frac * 100}%` }}
            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          />
        </div>
      )}
      {description && (
        <p className="mt-1.5 text-[9px] text-slate-600">
          {description}
        </p>
      )}
      {interpretation && (
        <p className="mt-0.5 text-[9px] font-medium text-slate-500">
          {interpretation}
        </p>
      )}
    </div>
  );
}
