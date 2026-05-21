import type { ReactNode } from 'react';
import { ProvenanceBadge } from './ProvenanceBadge';
import type { Provenance } from '@/lib/provenance';

interface ComputationCardProps {
  title: string;
  subtitle?: ReactNode;
  provenance: Provenance;
  /** Optional tiny step indicator, e.g. "Step 04 · MLP". Renders above title. */
  step?: string;
  /** Main visualization body. */
  children: ReactNode;
  /** Optional footer for compact metadata or explanation. */
  footer?: ReactNode;
  className?: string;
}

/**
 * Shared visual grammar for every active computation step in the Pipeline
 * workbench. All five subphases (preprocessing / ROI / MLP / vMF / CSLS) use
 * the same header/body/footer structure so the rhythm reads as one
 * instrument, not five different widgets.
 */
export function ComputationCard({
  title,
  subtitle,
  provenance,
  step,
  children,
  footer,
  className = '',
}: ComputationCardProps) {
  return (
    <div className={`flex flex-col gap-5 ${className}`}>
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          {step ? <p className="premium-kicker mb-1">{step}</p> : null}
          <h3 className="text-[16px] font-semibold leading-tight text-text-primary">{title}</h3>
          {subtitle ? (
            <p className="mt-1 max-w-2xl text-[13px] leading-relaxed text-text-muted">{subtitle}</p>
          ) : null}
        </div>
        <div className="shrink-0 pt-0.5">
          <ProvenanceBadge provenance={provenance} />
        </div>
      </header>

      <div className="space-y-4">{children}</div>

      {footer ? (
        <footer className="rounded-xl border border-border-subtle/70 bg-surface-base/40 px-4 py-2.5 text-[11.5px] leading-relaxed text-text-muted">
          {footer}
        </footer>
      ) : null}
    </div>
  );
}
