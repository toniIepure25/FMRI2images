import { motion } from 'framer-motion';
import { ImagePlaceholder } from '@/components/ImagePlaceholder';
import type { DemoCase } from '@/types';

export type ChallengeVerdict =
  | 'both_correct'
  | 'model_only'
  | 'human_only'
  | 'both_wrong';

export interface ChallengeResultProps {
  verdict: ChallengeVerdict;
  committeeCorrect: boolean;
  modelCorrect: boolean;
  case_: DemoCase;
  committeeChoiceSrc: string | null;
  committeeChoiceLabel: string;
}

const VERDICT_COPY: Record<ChallengeVerdict, string> = {
  both_correct: 'The model and the committee agree!',
  model_only:
    'The model found the correct image, but humans were misled by semantic similarity.',
  human_only: "The committee got it right, but the model's top-1 was a semantic neighbor.",
  both_wrong:
    'Neither the model nor the committee found the correct image — this was a genuinely ambiguous case.',
};

function resolveVerdict(committeeCorrect: boolean, modelCorrect: boolean): ChallengeVerdict {
  if (committeeCorrect && modelCorrect) return 'both_correct';
  if (modelCorrect && !committeeCorrect) return 'model_only';
  if (committeeCorrect && !modelCorrect) return 'human_only';
  return 'both_wrong';
}

export function deriveVerdict(committeeCorrect: boolean, modelCorrect: boolean): ChallengeVerdict {
  return resolveVerdict(committeeCorrect, modelCorrect);
}

export function ChallengeResult({
  verdict,
  committeeCorrect,
  modelCorrect,
  case_,
  committeeChoiceSrc,
  committeeChoiceLabel,
}: ChallengeResultProps) {
  const top1 = case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0];
  const rank = case_.metrics.rank;
  const cosine = case_.metrics.cosine;
  const csls = case_.metrics.csls;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="glass-panel border border-brain-border/50 p-5"
    >
      <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">Reveal</p>
      <motion.p
        className="mt-3 text-sm font-medium leading-relaxed text-slate-100 md:text-base"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.12, duration: 0.5 }}
      >
        {VERDICT_COPY[verdict]}
      </motion.p>

      <dl className="mt-4 grid gap-2 text-xs text-slate-400 sm:grid-cols-2">
        <div className="flex justify-between gap-2 border-b border-brain-border/30 pb-2">
          <dt className="text-slate-500">Committee</dt>
          <dd className={committeeCorrect ? 'font-medium text-emerald-400' : 'font-medium text-slate-300'}>
            {committeeCorrect ? 'Correct' : 'Incorrect'}
          </dd>
        </div>
        <div className="flex justify-between gap-2 border-b border-brain-border/30 pb-2">
          <dt className="text-slate-500">Model top-1</dt>
          <dd className={modelCorrect ? 'font-medium text-emerald-400' : 'font-medium text-slate-300'}>
            {modelCorrect ? 'Match (R@1)' : 'Different image'}
          </dd>
        </div>
        <div className="flex justify-between gap-2 border-b border-brain-border/30 pb-2 sm:col-span-2">
          <dt className="text-slate-500">Rank of ground truth in gallery</dt>
          <dd className="font-mono text-slate-200">{Number.isFinite(rank) ? rank : '—'}</dd>
        </div>
        <div className="flex justify-between gap-2 pb-2 sm:col-span-2">
          <dt className="text-slate-500">Similarity · cosine / CSLS</dt>
          <dd className="font-mono text-slate-200">
            {cosine != null ? cosine.toFixed(3) : '—'} · {csls != null ? csls.toFixed(3) : '—'}
          </dd>
        </div>
      </dl>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">Model top-1 retrieval</p>
          <div className="overflow-hidden rounded-lg border border-violet-500/30 bg-black/30">
            {top1?.image ? (
              <ImagePlaceholder
                src={top1.image}
                alt="Model top-1 retrieved"
                className="aspect-square border-0"
                type="retrieved"
                label={top1.score != null ? `score ${top1.score.toFixed(3)}` : undefined}
              />
            ) : (
              <div className="flex aspect-square items-center justify-center text-xs text-slate-600">—</div>
            )}
          </div>
        </div>
        <div>
          <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">{committeeChoiceLabel}</p>
          <div className="overflow-hidden rounded-lg border border-fuchsia-500/25 bg-black/30">
            {committeeChoiceSrc ? (
              <ImagePlaceholder
                src={committeeChoiceSrc}
                alt="Committee selection"
                className="aspect-square border-0"
                type="target"
              />
            ) : (
              <div className="flex aspect-square items-center justify-center text-xs text-slate-600">—</div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
