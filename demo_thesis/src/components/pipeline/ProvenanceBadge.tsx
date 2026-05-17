import type { Provenance, ProvenanceKind } from '@/lib/provenance';
import { provenanceLabel, provenanceTone } from '@/lib/provenance';

const TONE_CLASSES: Record<ProvenanceKind, string> = {
  live:        'border-accent/25 bg-accent/10 text-accent',
  replay:      'border-border-subtle bg-surface-raised/85 text-text-secondary',
  derived:     'border-status-info/20 bg-status-info/10 text-status-info',
  placeholder: 'border-status-error/20 bg-status-error/10 text-status-error',
  unknown:     'border-border-subtle bg-surface-raised/70 text-text-muted',
};

const DOT_CLASSES: Record<ProvenanceKind, string> = {
  live:        'bg-accent',
  replay:      'bg-text-muted',
  derived:     'bg-status-info',
  placeholder: 'bg-status-error',
  unknown:     'bg-border-emphasis',
};

interface ProvenanceBadgeProps {
  provenance: Provenance | ProvenanceKind;
  className?: string;
}

export function ProvenanceBadge({ provenance, className = '' }: ProvenanceBadgeProps) {
  const tone = provenanceTone(provenance);
  const label = provenanceLabel(provenance);
  const detail = typeof provenance === 'object' ? provenance.detail : undefined;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[10px] font-semibold shadow-[inset_0_1px_0_rgba(255,255,255,0.035)] ${TONE_CLASSES[tone]} ${className}`}
      title={detail}
    >
      <span className={`h-1 w-1 rounded-full ${DOT_CLASSES[tone]}`} />
      {label}
    </span>
  );
}
