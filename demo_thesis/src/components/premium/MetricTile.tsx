import type { ReactNode } from 'react';

type MetricTone = 'default' | 'accent' | 'success' | 'warning' | 'muted';

const toneClass: Record<MetricTone, string> = {
  default: 'text-text-primary',
  accent: 'text-accent',
  success: 'text-status-success',
  warning: 'text-status-warning',
  muted: 'text-text-secondary',
};

export function MetricTile({
  label,
  value,
  detail,
  tone = 'default',
  provenance,
  className = '',
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: MetricTone;
  provenance?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`premium-metric ${className}`}>
      <div className="flex items-center justify-between gap-3">
        <p className="premium-kicker">{label}</p>
        {provenance}
      </div>
      <div className={`mt-2 font-mono text-2xl font-semibold leading-none tabular-nums ${toneClass[tone]}`}>
        {value}
      </div>
      {detail ? <p className="mt-2 text-[11px] leading-snug text-text-muted">{detail}</p> : null}
    </div>
  );
}
