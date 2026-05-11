import type { Provenance, ProvenanceKind } from '@/lib/provenance';
import { provenanceLabel, provenanceTone } from '@/lib/provenance';

const TONE_CLASSES: Record<ProvenanceKind, string> = {
  live:        'bg-accent/[0.07] text-accent ring-1 ring-accent/15',
  replay:      'bg-surface-active text-text-secondary ring-1 ring-border-subtle',
  derived:     'bg-status-info/8 text-status-info ring-1 ring-status-info/12',
  placeholder: 'bg-status-error/6 text-status-error ring-1 ring-status-error/12',
  unknown:     'bg-surface-raised text-text-muted ring-1 ring-border-subtle',
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
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[9px] font-medium ${TONE_CLASSES[tone]} ${className}`}
      title={detail}
    >
      <span className={`h-1 w-1 rounded-full ${DOT_CLASSES[tone]}`} />
      {label}
    </span>
  );
}
