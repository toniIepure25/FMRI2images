import { motion } from 'framer-motion';
import { GlassCard } from '@/components/GlassCard';

export interface ScoreBoardProps {
  roundsPlayed: number;
  committeeCorrect: number;
  modelCorrect: number;
  draws: number;
  gameStarted: boolean;
  onPlayAgain: () => void;
}

export function ScoreBoard({
  roundsPlayed,
  committeeCorrect,
  modelCorrect,
  draws,
  gameStarted,
  onPlayAgain,
}: ScoreBoardProps) {
  const committeeLabel = roundsPlayed > 0 ? `${committeeCorrect}/${roundsPlayed}` : '—';
  const modelLabel = roundsPlayed > 0 ? `${modelCorrect}/${roundsPlayed}` : '—';

  return (
    <GlassCard className="h-fit p-5" glow>
      <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">Running score</p>
      <h2 className="mt-2 text-lg font-semibold text-white">Committee vs decoder</h2>
      <p className="mt-1 text-xs leading-relaxed text-slate-500">
        Draws count rounds where both the committee and the model’s top-1 retrieval were correct.
      </p>

      <dl className="mt-6 space-y-4">
        <motion.div
          key={`r-${roundsPlayed}`}
          initial={{ opacity: 0.6, x: 4 }}
          animate={{ opacity: 1, x: 0 }}
          className="rounded-lg border border-brain-border/40 bg-brain-navy/50 px-4 py-3"
        >
          <dt className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Rounds played</dt>
          <dd className="mt-1 font-mono text-2xl text-slate-100">{gameStarted ? roundsPlayed : '—'}</dd>
        </motion.div>

        <div className="rounded-lg border border-brain-border/40 bg-brain-navy/50 px-4 py-3">
          <dt className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Committee</dt>
          <dd className="mt-1 font-mono text-xl text-brain-accent">{committeeLabel}</dd>
          <p className="mt-1 text-[10px] text-slate-500">Correct selections over total rounds</p>
        </div>

        <div className="rounded-lg border border-brain-border/40 bg-brain-navy/50 px-4 py-3">
          <dt className="text-[10px] font-medium uppercase tracking-wider text-slate-500">Model (R@1)</dt>
          <dd className="mt-1 font-mono text-xl text-fuchsia-300">{modelLabel}</dd>
          <p className="mt-1 text-[10px] text-slate-500">Top-1 retrieval hits over total rounds</p>
        </div>

        <div className="rounded-lg border border-emerald-500/20 bg-emerald-950/20 px-4 py-3">
          <dt className="text-[10px] font-medium uppercase tracking-wider text-emerald-400/80">Agreement (both correct)</dt>
          <dd className="mt-1 font-mono text-xl text-emerald-300">{gameStarted ? draws : '—'}</dd>
        </div>
      </dl>

      <button
        type="button"
        onClick={onPlayAgain}
        className="mt-6 w-full rounded-lg border border-brain-border/60 bg-brain-panel/80 py-2.5 text-xs font-semibold text-slate-200 transition hover:border-brain-accent/40 hover:bg-brain-accent/10 hover:text-white"
      >
        Play again
      </button>
    </GlassCard>
  );
}
