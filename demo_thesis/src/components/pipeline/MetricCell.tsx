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
      className={`rounded-xl border border-white/[0.05] bg-white/[0.02] px-3.5 py-3 ${className}`}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="premium-kicker">{label}</p>
        <ProvenanceBadge provenance={provenance} />
      </div>
      <p
        className={`mt-2 font-mono text-[20px] font-semibold tabular-nums leading-none tracking-tight ${
          available ? (valueColor ?? 'text-text-primary') : 'text-text-muted'
        }`}
      >
        {displayValue}
      </p>
      {description && (
        <p className="mt-1.5 text-[10.5px] leading-snug text-text-muted">{description}</p>
      )}
      {caption && (
        <p className="mt-0.5 text-[10.5px] leading-snug text-text-muted">{caption}</p>
      )}
    </div>
  );
}
