import { motion } from 'framer-motion';
import { ProvenanceBadge } from './ProvenanceBadge';
import type { Provenance } from '@/lib/provenance';

interface ScientificVerdictProps {
  rank: number;
  verdictText: string;
  verdictCls: string;
  kappa?: number | null;
  delta?: number | null;
  top1Csls?: number | null;
  top2Csls?: number | null;
  topKEntropy?: number | null;
  gallerySize?: number;
  provenance: Provenance;
}

function reliabilityLabel(rank: number, margin: number, kappa: number | null, entropy: number | null): {
  label: string; cls: string;
} {
  let score = 0;
  if (rank === 1) score += 3;
  else if (rank <= 3) score += 2;
  else if (rank <= 5) score += 1;
  if (margin > 1.0) score += 3;
  else if (margin > 0.3) score += 2;
  else if (margin > 0.1) score += 1;
  if (kappa !== null && kappa > 100) score += 2;
  else if (kappa !== null && kappa > 30) score += 1;
  if (entropy !== null && entropy < 1.0) score += 2;
  else if (entropy !== null && entropy < 2.0) score += 1;

  if (score >= 7) return { label: 'High trust', cls: 'text-status-success bg-status-success/8 ring-1 ring-status-success/20' };
  if (score >= 4) return { label: 'Moderate trust', cls: 'text-status-warning bg-status-warning/8 ring-1 ring-status-warning/15' };
  return { label: 'Suspicious', cls: 'text-status-error bg-status-error/6 ring-1 ring-status-error/12' };
}

function computeTopKEntropy(topK: Array<{ csls: number }> | undefined): number {
  if (!topK || topK.length < 2) return 0;
  const scores = topK.map(r => Math.max(1e-8, r.csls));
  const total = scores.reduce((a, b) => a + b, 0);
  const probs = scores.map(s => s / total);
  return -probs.reduce((h, p) => h + p * Math.log(p + 1e-8), 0);
}

export function ScientificVerdict({
  rank, verdictText, verdictCls, kappa, delta,
  top1Csls, top2Csls, topKEntropy, gallerySize, provenance,
}: ScientificVerdictProps) {
  const margin = (top1Csls != null && top2Csls != null) ? top1Csls - top2Csls : null;
  const reliability = reliabilityLabel(rank, margin ?? 0, kappa ?? null, topKEntropy ?? null);
  const entropyDisplay = topKEntropy != null ? topKEntropy.toFixed(2) : '—';

  return (
    <motion.div
      className="rounded-xl border border-border-subtle bg-surface-elevated p-5 sm:p-6"
      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-text-primary">Scientific verdict</h3>
        <ProvenanceBadge provenance={provenance} />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {/* Rank */}
        <div className="rounded-lg bg-surface-raised px-4 py-3">
          <p className="text-[10px] font-medium text-text-muted">Rank</p>
          <div className="flex items-baseline gap-2 mt-1">
            <span className={`font-mono text-2xl font-bold ${rank === 1 ? 'text-accent' : 'text-text-primary'}`}>
              #{rank}
            </span>
            <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${verdictCls}`}>
              {verdictText}
            </span>
          </div>
          <p className="text-[10px] text-text-muted mt-1">
            {gallerySize != null ? `of ${gallerySize.toLocaleString()} gallery images` : 'CSLS ranking'}
          </p>
        </div>

        {/* κ confidence */}
        <div className="rounded-lg bg-surface-raised px-4 py-3">
          <p className="text-[10px] font-medium text-text-muted">κ directional confidence</p>
          <p className="font-mono text-2xl font-bold text-accent mt-1">{kappa != null ? kappa.toFixed(1) : '—'}</p>
          <p className="text-[10px] text-text-muted mt-1">
            {kappa != null ? (kappa > 100 ? 'sharp vMF peak' : kappa > 30 ? 'moderate concentration' : 'broad uncertainty') : 'not available'}
          </p>
        </div>

        {/* CSLS margin */}
        <div className="rounded-lg bg-surface-raised px-4 py-3">
          <p className="text-[10px] font-medium text-text-muted">CSLS margin (1−2)</p>
          <p className="font-mono text-2xl font-bold text-text-primary mt-1">
            {margin != null ? margin.toFixed(4) : '—'}
          </p>
          <p className="text-[10px] text-text-muted mt-1">
            {margin != null ? (margin > 1.0 ? 'very well separated' : margin > 0.1 ? 'moderate separation' : 'near tie') : 'not available'}
          </p>
        </div>

        {/* Top-K entropy */}
        <div className="rounded-lg bg-surface-raised px-4 py-3">
          <p className="text-[10px] font-medium text-text-muted">Top-K entropy</p>
          <p className="font-mono text-2xl font-bold text-text-primary mt-1">{entropyDisplay}</p>
          <p className="text-[10px] text-text-muted mt-1">
            {topKEntropy != null ? (topKEntropy < 1.0 ? 'low uncertainty' : topKEntropy < 2.0 ? 'moderate' : 'high entropy') : 'not available'}
          </p>
        </div>
      </div>

      {/* Reliability verdict */}
      <div className="mt-4 flex items-center gap-3 rounded-lg bg-surface-raised px-4 py-3">
        <span className="text-[11px] font-semibold text-text-muted">Reliability:</span>
        <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold ${reliability.cls}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${
            reliability.label === 'High trust' ? 'bg-status-success' : reliability.label === 'Moderate trust' ? 'bg-status-warning' : 'bg-status-error'
          }`} />
          {reliability.label}
        </span>
        <span className="text-[10px] text-text-muted">based on rank, margin, κ, and entropy</span>
      </div>
    </motion.div>
  );
}

// Export utility for external use
export { computeTopKEntropy, reliabilityLabel };
