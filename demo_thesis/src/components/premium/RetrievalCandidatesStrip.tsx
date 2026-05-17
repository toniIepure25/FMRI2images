import { motion } from 'framer-motion';
import type { RetrievedImage } from '@/types';
import { ImageEvidenceCard } from './ImageEvidenceCard';

function candidateLabel(label: RetrievedImage['label']) {
  if (label === 'correct') return 'Exact match';
  if (label === 'semantic_neighbor') return 'Semantic neighbor';
  return 'Distractor';
}

export function RetrievalCandidatesStrip({
  candidates,
  visibleRanks,
}: {
  candidates: RetrievedImage[];
  visibleRanks?: number[];
}) {
  const visible = visibleRanks ?? candidates.map((c) => c.rank);

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
      {candidates.map((item, index) => {
        const shown = visible.includes(item.rank);
        const score = item.csls != null ? item.csls.toFixed(3) : 'n/a';
        return (
          <motion.div
            key={item.rank}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: shown ? 1 : 0.18, y: shown ? 0 : 14 }}
            transition={{ duration: 0.26, delay: Math.min(index * 0.035, 0.18) }}
          >
            <ImageEvidenceCard
              imageSrc={item.image}
              title={`Rank #${item.rank}`}
              subtitle={candidateLabel(item.label)}
              emphasis={item.rank === 1}
              badge={
                <span className={`premium-rank-badge ${item.rank === 1 ? 'premium-rank-badge-primary' : ''}`}>
                  #{item.rank}
                </span>
              }
              footer={
                <div className="flex items-center justify-between gap-2 text-[11px]">
                  <span className="font-mono text-text-secondary">CSLS {score}</span>
                  <span className={item.label === 'correct' ? 'text-status-success' : 'text-text-muted'}>
                    {item.label === 'correct' ? 'match' : 'candidate'}
                  </span>
                </div>
              }
            />
          </motion.div>
        );
      })}
    </div>
  );
}
