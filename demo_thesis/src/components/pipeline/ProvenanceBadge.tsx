import type { Provenance, ProvenanceKind } from '@/lib/provenance';
import { provenanceLabel, provenanceTone } from '@/lib/provenance';

const TONE_CLASSES: Record<ProvenanceKind, string> = {
  live: 'bg-emerald-500/8 text-emerald-400',
  replay: 'bg-cyan-500/6 text-cyan-400',
  derived: 'bg-amber-500/6 text-amber-400',
  placeholder: 'bg-rose-500/6 text-rose-400',
  unknown: 'bg-slate-500/6 text-slate-500',
};

const DOT_CLASSES: Record<ProvenanceKind, string> = {
  live: 'bg-emerald-400',
  replay: 'bg-cyan-400',
  derived: 'bg-amber-400',
  placeholder: 'bg-rose-400',
  unknown: 'bg-slate-500',
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
      className={`inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[8px] font-semibold ${TONE_CLASSES[tone]} ${className}`}
      title={detail}
    >
      <span className={`h-1 w-1 rounded-full ${DOT_CLASSES[tone]}`} />
      {label}
    </span>
  );
}
