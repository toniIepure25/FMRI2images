import { useState } from 'react';
import { motion } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { EvidenceImage } from './EvidenceImage';
import { KickerRule, MonoNote } from './atoms';
import {
  retrievalCandidates,
  retrievalTarget,
  type RetrievalCandidate,
} from '@/data/neuralManifoldExplorer';
import { DERIVED_PROV, REPLAY_PROV } from '@/lib/provenance';

const STATUS_TONE: Record<RetrievalCandidate['status'], {
  frame: string;
  chip:  string;
  label: string;
  dot:   string;
  row:   string;
}> = {
  match: {
    frame: 'border-status-success/30',
    chip:  'text-status-success bg-status-success/[0.10] ring-status-success/24',
    label: 'Match',
    dot:   'bg-status-success',
    row:   'bg-status-success/[0.05]',
  },
  near: {
    frame: 'border-white/[0.07]',
    chip:  'text-text-secondary bg-white/[0.03] ring-white/[0.06]',
    label: 'Near',
    dot:   'bg-text-secondary/85',
    row:   '',
  },
  distractor: {
    frame: 'border-white/[0.04]',
    chip:  'text-text-muted bg-white/[0.015] ring-white/[0.04]',
    label: 'Other',
    dot:   'bg-text-muted/70',
    row:   '',
  },
};

/**
 * RetrievalEvidencePanel
 * ─────────────────────────────────────────────────────────────
 * Premium evidence board, compact:
 *   left  28%  target stimulus, strong but not oversized
 *   right 72%  top-5 candidate strip + integrated value table
 *
 * Hovering or focusing a candidate row in the table highlights
 * the matching card, and vice versa, so the strip and table read
 * as one device rather than two stacked blocks.
 *
 * EvidenceImage guarantees no candidate can ever paint a white
 * block — its dark MissingMedia surface stays behind the image
 * before, during, and after load.
 */
export function RetrievalEvidencePanel() {
  const [activeRank, setActiveRank] = useState<number | null>(null);

  return (
    <motion.section
      id="manifold-retrieval"
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] scroll-mt-20 px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Retrieval evidence"
        title="Top-K candidates and the cosine → CSLS shift."
        description="Cosine retrieval is shaped by hubs in CLIP space; CSLS reweights each candidate by its own and the query's local neighbourhood density. Per-row deltas show which candidates the correction helps or demotes."
        action={<ProvenanceBadge provenance={DERIVED_PROV} />}
      />

      <PremiumPanel className="mt-6 p-5 sm:p-6">
        <div className="grid gap-5 lg:grid-cols-[minmax(200px,0.55fr)_minmax(0,2.3fr)]">

          {/* ── Target stimulus ── */}
          <div className="flex flex-col gap-3">
            <KickerRule label="Target stimulus" />
            <figure className="overflow-hidden rounded-xl border border-status-success/22 bg-surface-elevated">
              <EvidenceImage
                src={retrievalTarget.image}
                alt="Subject perceived NSD stimulus"
                aspect="4/5"
                overlay={
                  <span className="absolute left-2.5 top-2.5 inline-flex items-center gap-1 rounded-full bg-black/55 px-1.5 py-[2px] text-[9px] font-semibold uppercase tracking-[0.08em] text-status-success/95 ring-1 ring-status-success/28 backdrop-blur-md">
                    <span className="h-1 w-1 rounded-full bg-status-success" />
                    Target
                  </span>
                }
              />
              <figcaption className="flex items-baseline justify-between gap-3 px-3 py-2.5">
                <p className="truncate text-[12px] font-semibold tracking-tight text-text-primary">
                  Subject perceived
                </p>
                <MonoNote>{retrievalTarget.caption}</MonoNote>
              </figcaption>
            </figure>
            <p className="text-[10.5px] leading-snug text-text-muted">
              The decoder predicts one 768-D CLIP direction for this trial; the gallery is then
              ranked by CSLS-corrected similarity to that direction.
            </p>
          </div>

          {/* ── Candidate strip + table — share the right column ── */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <KickerRule label="Top-5 retrieved · CSLS" />
              <MonoNote>Gallery 10,000</MonoNote>
            </div>

            {/* 5-up candidate strip — square aspect for consistent height */}
            <div className="grid grid-cols-5 gap-2.5">
              {retrievalCandidates.map((c) => (
                <Candidate
                  key={c.rank}
                  c={c}
                  active={activeRank === c.rank}
                  onEnter={() => setActiveRank(c.rank)}
                  onLeave={() => setActiveRank((cur) => (cur === c.rank ? null : cur))}
                />
              ))}
            </div>

            {/* Integrated value table */}
            <div className="overflow-hidden rounded-xl border border-white/[0.05] bg-white/[0.012]">
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="border-b border-white/[0.05] text-text-muted">
                    <th className="px-3 py-2 text-left font-semibold uppercase tracking-[0.12em]">#</th>
                    <th className="px-3 py-2 text-left font-semibold uppercase tracking-[0.12em]">Caption</th>
                    <th className="px-3 py-2 text-right font-semibold uppercase tracking-[0.12em]">Cosine</th>
                    <th className="px-3 py-2 text-right font-semibold uppercase tracking-[0.12em]">CSLS</th>
                    <th className="px-3 py-2 text-right font-semibold uppercase tracking-[0.12em]">Δ</th>
                    <th className="px-3 py-2 text-right font-semibold uppercase tracking-[0.12em]">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {retrievalCandidates.map((c) => {
                    const delta = c.csls - c.cosine;
                    const sign = delta > 0 ? '+' : delta < 0 ? '−' : '±';
                    const deltaClass =
                      delta > 0.005
                        ? 'text-status-success'
                        : delta < -0.005
                        ? 'text-status-warning'
                        : 'text-text-muted';
                    const tone = STATUS_TONE[c.status];
                    const active = activeRank === c.rank;
                    return (
                      <tr
                        key={c.rank}
                        onMouseEnter={() => setActiveRank(c.rank)}
                        onMouseLeave={() => setActiveRank((cur) => (cur === c.rank ? null : cur))}
                        className={`border-b border-white/[0.04] last:border-b-0 transition ${
                          active
                            ? 'bg-accent/[0.06]'
                            : tone.row
                        }`}
                      >
                        <td className="px-3 py-2 font-mono text-text-secondary">#{c.rank}</td>
                        <td className="px-3 py-2 text-text-secondary">{c.caption}</td>
                        <td className="px-3 py-2 text-right font-mono tabular-nums text-text-primary">{c.cosine.toFixed(3)}</td>
                        <td className="px-3 py-2 text-right font-mono tabular-nums text-accent">{c.csls.toFixed(3)}</td>
                        <td className={`px-3 py-2 text-right font-mono tabular-nums ${deltaClass}`}>
                          {sign}{Math.abs(delta).toFixed(3)}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[9.5px] font-semibold uppercase tracking-[0.1em] ring-1 ${tone.chip}`}>
                            <span className={`h-1 w-1 rounded-full ${tone.dot}`} />
                            {tone.label}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div className="flex items-center justify-between gap-3 text-[10.5px] text-text-muted">
              <p>
                CSLS reduces high-dimensional retrieval hubness — see the next panel for population-level
                evidence on SHARED1000.
              </p>
              <ProvenanceBadge provenance={REPLAY_PROV} />
            </div>
          </div>
        </div>
      </PremiumPanel>
    </motion.section>
  );
}

function Candidate({
  c,
  active,
  onEnter,
  onLeave,
}: {
  c: RetrievalCandidate;
  active: boolean;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const tone = STATUS_TONE[c.status];
  return (
    <figure
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      className={`overflow-hidden rounded-lg border bg-surface-elevated transition ${
        active
          ? 'border-accent/40 ring-1 ring-accent/22'
          : tone.frame
      }`}
    >
      <EvidenceImage
        src={c.image}
        alt={c.caption}
        aspect="square"
        overlay={
          <>
            <span className="absolute left-1.5 top-1.5 rounded-md bg-black/55 px-1.5 py-[2px] font-mono text-[9px] font-semibold tabular-nums text-white/85 ring-1 ring-white/10 backdrop-blur">
              #{c.rank}
            </span>
            <span className={`absolute right-1.5 top-1.5 inline-flex items-center gap-1 rounded-full px-1.5 py-[2px] font-mono text-[9px] font-semibold uppercase tracking-[0.08em] ring-1 backdrop-blur-md ${tone.chip}`}>
              <span className={`h-1 w-1 rounded-full ${tone.dot}`} />
              {tone.label}
            </span>
          </>
        }
      />
      <figcaption className="flex items-baseline justify-between gap-2 px-2 py-1.5">
        <p className="truncate text-[10.5px] text-text-secondary">{c.caption}</p>
        <p className="shrink-0 font-mono text-[9.5px] tabular-nums text-accent">
          {c.csls.toFixed(2)}
        </p>
      </figcaption>
    </figure>
  );
}
