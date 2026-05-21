import { motion } from 'framer-motion';
import type { RetrievedImage } from '@/types';

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
    <div className="grid grid-cols-2 gap-2.5 md:grid-cols-5">
      {candidates.map((item, index) => {
        const shown = visible.includes(item.rank);
        const score = item.csls != null ? item.csls.toFixed(3) : 'N/A';
        const isTop = item.rank === 1;
        return (
          <motion.div
            key={item.rank}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: shown ? 1 : 0.18, y: shown ? 0 : 10 }}
            transition={{ duration: 0.26, delay: Math.min(index * 0.035, 0.18) }}
          >
            {/* ── Curated evidence tile — image-dominant, hair-thin chrome,
                   no shadows. The rank/CSLS read as scientific annotation, not
                   a product chip. ── */}
            <figure
              className={`group overflow-hidden rounded-xl border bg-surface-elevated/85 transition duration-300 ${
                isTop
                  ? 'border-accent/35'
                  : 'border-white/[0.05] hover:border-white/[0.10]'
              }`}
            >
              <div className="relative aspect-[4/3] bg-surface-base">
                {item.image ? (
                  <img
                    src={item.image}
                    alt={`Retrieved candidate rank ${item.rank}`}
                    className="h-full w-full object-cover transition duration-500 group-hover:scale-[1.02]"
                    loading="lazy"
                  />
                ) : (
                  <div className="flex h-full w-full items-center justify-center text-xs text-text-muted">Unavailable</div>
                )}
                {/* Hair-thin inset ring — softens the photo edge without adding chrome. */}
                <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/[0.04]" aria-hidden />
                {/* Rank annotation — compact mono badge, top-left. */}
                <div className="absolute left-2 top-2">
                  <span
                    className={`inline-flex items-center rounded-md border backdrop-blur-md px-1.5 py-0.5 font-mono text-[10px] font-semibold tabular-nums ${
                      isTop
                        ? 'border-accent/35 bg-black/45 text-accent'
                        : 'border-white/10 bg-black/45 text-white/85'
                    }`}
                  >
                    #{item.rank}
                  </span>
                </div>
              </div>
              {/* Caption — one line, kicker label + tabular CSLS. */}
              <figcaption className="flex items-baseline justify-between gap-2 px-3 py-2">
                <p
                  className={`truncate text-[11px] font-semibold tracking-tight ${
                    isTop ? 'text-accent' : 'text-text-secondary'
                  }`}
                >
                  {candidateLabel(item.label)}
                </p>
                <p
                  className={`shrink-0 font-mono text-[10.5px] tabular-nums ${
                    isTop ? 'text-text-primary' : 'text-text-muted'
                  }`}
                >
                  {score}
                </p>
              </figcaption>
            </figure>
          </motion.div>
        );
      })}
    </div>
  );
}
