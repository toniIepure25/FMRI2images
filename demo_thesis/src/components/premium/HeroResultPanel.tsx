import type { ReactNode } from 'react';
import type { DemoCase, RetrievedImage } from '@/types';
import { ImageEvidenceCard } from './ImageEvidenceCard';
import { MetricTile } from './MetricTile';
import { PremiumPanel } from './PremiumPanel';

function verdictForRank(rank: number | null | undefined) {
  if (rank === 1) return { label: 'Exact match', tone: 'success' as const };
  if (rank != null && rank <= 5) return { label: 'Near match', tone: 'warning' as const };
  return { label: 'Candidate identified', tone: 'default' as const };
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

  return (
    <PremiumPanel variant="hero" className="p-5 sm:p-6 xl:p-8">
      <div className="grid gap-7 xl:grid-cols-[0.9fr_1.1fr] xl:items-stretch">
        <div className="space-y-5">
          <div className="flex flex-wrap items-center gap-3">
            <p className="premium-kicker">Retrieval complete</p>
            {liveLabel}
            {provenance}
          </div>
          <div>
            <div className="flex flex-wrap items-end gap-4">
              <span className={`font-mono text-8xl font-semibold leading-none tracking-tight tabular-nums ${
                verdict.tone === 'success' ? 'text-status-success' : verdict.tone === 'warning' ? 'text-status-warning' : 'text-text-primary'
              }`}>
                #{case_.metrics.rank ?? '?'}
              </span>
              <div className="pb-2">
                <h2 className="text-4xl font-semibold tracking-tight text-text-primary">{verdict.label}</h2>
                <p className="mt-1 text-sm text-text-secondary">
                  Top-ranked result from a 10,000-image CSLS gallery search.
                </p>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricTile label="Rank" value={`#${case_.metrics.rank ?? '?'}`} tone={verdict.tone} detail="Top visual hypothesis" />
            <MetricTile label="CSLS" value={top1?.csls != null ? top1.csls.toFixed(3) : 'n/a'} detail="Top-1 score" />
            <MetricTile label="Margin" value={margin != null ? margin.toFixed(4) : 'n/a'} detail="Top-1 minus top-2" />
            <MetricTile label="κ" value={case_.uncertainty.kappa.toFixed(1)} tone="accent" detail="Directional concentration" />
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <ImageEvidenceCard
            imageSrc={case_.targetImage}
            title="Subject perceived"
            subtitle="Reference NSD stimulus"
            emphasis={case_.metrics.rank === 1}
            aspectClass="aspect-[5/4]"
            badge={<span className="premium-rank-badge">target</span>}
          />
          <ImageEvidenceCard
            imageSrc={top1?.image}
            title="Model retrieved"
            subtitle={top1 ? `Rank #${top1.rank} gallery image` : 'Awaiting candidate'}
            emphasis={case_.metrics.rank === 1}
            aspectClass="aspect-[5/4]"
            badge={<span className="premium-rank-badge premium-rank-badge-primary">top-1</span>}
          />
        </div>
      </div>
    </PremiumPanel>
  );
}
