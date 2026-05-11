import type { Provenance } from '@/lib/provenance';
import { ProvenanceBadge } from './ProvenanceBadge';
import { formatMetric, metricCaption, isMetricAvailable } from '@/lib/metrics';

interface MetricCellProps {
  label: string;
  value: number | string | null | undefined;
  decimals?: number;
  description?: string;
  unavailableText?: string;
  provenance: Provenance;
  className?: string;
  valueColor?: string;
}

export function MetricCell({
  label,
  value,
  decimals = 3,
  description,
  unavailableText,
  provenance,
  className = '',
  valueColor,
}: MetricCellProps) {
  const displayValue =
    typeof value === 'string' ? value : formatMetric(value, decimals);
  const caption =
    typeof value === 'string' ? null : metricCaption(value, unavailableText);
  const available = typeof value === 'string' || isMetricAvailable(value);

  return (
    <div
      className={`rounded-xl bg-white/[0.025] p-4 ${className}`}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="text-[9px] font-medium text-slate-500">
          {label}
        </p>
        <ProvenanceBadge provenance={provenance} />
      </div>
      <p
        className={`mt-1.5 font-mono text-[22px] font-semibold tabular-nums leading-none ${
          available ? (valueColor ?? 'text-white') : 'text-slate-600'
        }`}
      >
        {displayValue}
      </p>
      {description && (
        <p className="mt-1.5 text-[9px] text-slate-600">{description}</p>
      )}
      {caption && (
        <p className="mt-0.5 text-[8px] text-slate-600">{caption}</p>
      )}
    </div>
  );
}
