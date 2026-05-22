import type { ReactNode } from 'react';
import type { DemoCase, RetrievedImage } from '@/types';
import { PremiumPanel } from './PremiumPanel';

function verdictForRank(rank: number | null | undefined) {
  if (rank === 1) return { label: 'Exact match', tone: 'success' as const };
  if (rank != null && rank <= 5) return { label: 'Near match', tone: 'warning' as const };
  return { label: 'Candidate identified', tone: 'default' as const };
}

/** Same-size image pair. The image is the hero; caption is a precise strip. */
function EvidenceFrame({
  title,
  subtitle,
  src,
  emphasis,
}: {
  title: string;
  subtitle: string;
  src?: string | null;
  emphasis?: boolean;
}) {
  return (
    <figure
      className={`group overflow-hidden rounded-xl border bg-surface-elevated/80 transition duration-300 ${
        emphasis ? 'border-status-success/20' : 'border-white/[0.05]'
      }`}
      style={{ boxShadow: '0 1px 0 rgb(255 255 255 / 0.022) inset, 0 14px 36px -30px rgb(0 0 0 / 0.85)' }}
    >
      <div className="relative aspect-[4/3] bg-surface-raised">
        {src ? (
          <img
            src={src}
            alt={title}
            className="h-full w-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-xs text-text-muted">
            Unavailable
          </div>
        )}
        {/* Hair-thin inner ring to soften the photo edge */}
        <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/[0.04]" aria-hidden />
        {emphasis ? (
          <span className="absolute left-2.5 top-2.5 inline-flex items-center gap-1 rounded-full bg-black/45 px-1.5 py-[2px] text-[9px] font-semibold uppercase tracking-[0.08em] text-status-success/90 ring-1 ring-status-success/28 backdrop-blur-md">
            <span className="h-[4px] w-[4px] rounded-full bg-status-success" />
            Match
          </span>
        ) : null}
      </div>
      {/* Caption — slim hairline strip, image-first weighting */}
      <figcaption className="flex items-baseline justify-between gap-3 px-3.5 py-2.5">
        <p className="truncate text-[12.5px] font-semibold tracking-tight text-text-primary">
          {title}
        </p>
        <p className="shrink-0 text-[10.5px] uppercase tracking-[0.14em] text-text-muted/80">
          {subtitle}
        </p>
      </figcaption>
    </figure>
  );
}

/** Hairline evidence strip — Rank · CSLS · Margin · κ, calibration-light. */
function EvidenceStrip({
  rank,
  csls,
  margin,
  kappa,
  tone,
}: {
  rank: number | null | undefined;
  csls: string;
  margin: string;
  kappa: string;
  tone: 'success' | 'warning' | 'default';
}) {
  const rankColor =
    tone === 'success'
      ? 'text-status-success'
      : tone === 'warning'
      ? 'text-status-warning'
      : 'text-text-primary';

  const items: { label: string; value: string; color?: string }[] = [
    { label: 'Rank',   value: rank != null ? `#${rank}` : '—', color: rankColor },
    { label: 'CSLS',   value: csls },
    { label: 'Margin', value: margin },
    { label: 'κ',      value: kappa, color: 'text-accent' },
  ];

  return (
    <div className="grid grid-cols-4 divide-x divide-white/[0.05]">
      {items.map(({ label, value, color }, i) => (
        <div key={label} className={`flex flex-col px-4 py-2 ${i === 0 ? 'pl-0' : ''}`}>
          <p className="premium-kicker">{label}</p>
          <p
            className={`mt-1.5 font-mono text-[19px] font-semibold tabular-nums leading-none tracking-tight ${
              color ?? 'text-text-primary'
            }`}
          >
            {value}
          </p>
        </div>
      ))}
    </div>
  );
}

export function HeroResultPanel({
  case_,
  top1,
  provenance,
  liveLabel,
}: {
  case_: DemoCase;
  top1?: RetrievedImage;
  provenance?: ReactNode;
  liveLabel?: ReactNode;
}) {
  const verdict = verdictForRank(case_.metrics.rank);
  const margin =
    case_.retrievedImages[0]?.csls != null && case_.retrievedImages[1]?.csls != null
      ? case_.retrievedImages[0].csls - case_.retrievedImages[1].csls
      : null;

  const verdictHeadingTone =
    verdict.tone === 'success'
      ? 'text-status-success'
      : verdict.tone === 'warning'
      ? 'text-status-warning'
      : 'text-text-primary';

  return (
    <PremiumPanel variant="hero" className="p-5 sm:p-6">
      {/* ── Report header — kicker + hairline + verdict tone + metadata ── */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2.5">
            <span className="h-px w-7 bg-accent/45" aria-hidden />
            <p className="premium-kicker">Retrieval complete</p>
          </div>
          <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className={`text-[16px] font-semibold tracking-tight ${verdictHeadingTone}`}>
              {verdict.label}
            </span>
            <span className="font-mono text-[11.5px] tabular-nums text-text-muted">
              10,000-image CSLS gallery · {case_.subject}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 sm:self-start">
          {liveLabel}
          {provenance}
        </div>
      </div>

      {/* ── Image pair: same-size, image-dominant, thin captions ── */}
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <EvidenceFrame
          src={case_.targetImage}
          title="Subject perceived"
          subtitle="NSD stimulus"
          emphasis={case_.metrics.rank === 1}
        />
        <EvidenceFrame
          src={top1?.image}
          title="Model retrieved"
          subtitle={top1 ? `Rank #${top1.rank} · gallery` : 'Awaiting candidate'}
          emphasis={case_.metrics.rank === 1}
        />
      </div>

      {/* ── Evidence strip: flat hairline columns, no surrounding card. ── */}
      <div className="mt-5 border-t border-white/[0.05] pt-3">
        <EvidenceStrip
          rank={case_.metrics.rank}
          csls={top1?.csls != null ? top1.csls.toFixed(3) : 'N/A'}
          margin={margin != null ? margin.toFixed(4) : 'N/A'}
          kappa={case_.uncertainty.kappa.toFixed(1)}
          tone={verdict.tone}
        />
      </div>
    </PremiumPanel>
  );
}
