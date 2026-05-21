import type { ReactNode } from 'react';
import type { DemoCase, RetrievedImage } from '@/types';
import { PremiumPanel } from './PremiumPanel';

function verdictForRank(rank: number | null | undefined) {
  if (rank === 1) return { label: 'Exact match', tone: 'success' as const };
  if (rank != null && rank <= 5) return { label: 'Near match', tone: 'warning' as const };
  return { label: 'Candidate identified', tone: 'default' as const };
}

/** Same-size image pair. The image is the hero; the caption is a thin strip. */
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
      className={`group overflow-hidden rounded-2xl border bg-surface-elevated/85 transition duration-300 ${
        emphasis ? 'border-status-success/22' : 'border-white/[0.05]'
      }`}
      style={{ boxShadow: '0 1px 0 rgb(255 255 255 / 0.025) inset' }}
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
          <span className="absolute left-2.5 top-2.5 inline-flex items-center gap-1 rounded-md border border-status-success/25 bg-black/45 px-2 py-0.5 text-[10px] font-semibold text-status-success backdrop-blur-md">
            <span className="h-1.5 w-1.5 rounded-full bg-status-success" />
            Match
          </span>
        ) : null}
      </div>
      {/* Compact caption — no large empty footer black */}
      <figcaption className="px-3.5 py-2.5">
        <p className="truncate text-[12.5px] font-semibold text-text-primary">{title}</p>
        <p className="mt-0.5 truncate text-[11px] text-text-muted">{subtitle}</p>
      </figcaption>
    </figure>
  );
}

/** Hairline-separated evidence strip — Rank · CSLS · Margin · κ in one row. */
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

  const items: { label: string; value: string; color?: string; mono?: boolean }[] = [
    { label: 'Rank', value: rank != null ? `#${rank}` : '—', color: rankColor, mono: true },
    { label: 'CSLS', value: csls, mono: true },
    { label: 'Margin', value: margin, mono: true },
    { label: 'κ', value: kappa, mono: true, color: 'text-accent' },
  ];

  return (
    <div className="grid grid-cols-4 divide-x divide-white/[0.05] overflow-hidden rounded-xl border border-white/[0.04] bg-white/[0.012]">
      {items.map(({ label, value, color, mono }) => (
        <div key={label} className="flex flex-col justify-center px-4 py-3">
          <p className="premium-kicker">{label}</p>
          <p
            className={`mt-1 ${mono ? 'font-mono' : ''} text-[18px] font-semibold tabular-nums leading-none tracking-tight ${
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
      {/* ── Compact header row: kicker + verdict label + provenance ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-baseline gap-3">
          <p className="premium-kicker">Retrieval complete</p>
          <span className={`text-[13px] font-semibold tracking-tight ${verdictHeadingTone}`}>
            {verdict.label}
          </span>
          <span className="text-[11.5px] text-text-muted">
            · 10,000-image CSLS gallery · {case_.subject}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {liveLabel}
          {provenance}
        </div>
      </div>

      {/* ── Image pair: same-size, image-dominant, thin captions ── */}
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <EvidenceFrame
          src={case_.targetImage}
          title="Subject perceived"
          subtitle="Reference NSD stimulus"
          emphasis={case_.metrics.rank === 1}
        />
        <EvidenceFrame
          src={top1?.image}
          title="Model retrieved"
          subtitle={top1 ? `Rank #${top1.rank} gallery image` : 'Awaiting candidate'}
          emphasis={case_.metrics.rank === 1}
        />
      </div>

      {/* ── Evidence strip: hairline-separated stats, no chunky boxes ── */}
      <div className="mt-4">
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
