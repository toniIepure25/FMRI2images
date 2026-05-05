import { useCallback, useMemo, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { useChallengePool } from '@/lib/hooks';
import { LoadingState } from '@/components/LoadingState';
import { ErrorState } from '@/components/ErrorState';
import { ImagePlaceholder } from '@/components/ImagePlaceholder';
import { GlassCard } from '@/components/GlassCard';
import { ChallengeCard } from '@/components/challenge/ChallengeCard';
import { ChallengeResult, deriveVerdict } from '@/components/challenge/ChallengeResult';
import { ScoreBoard } from '@/components/challenge/ScoreBoard';

export interface RoundOption {
  src: string;
  isCorrect: boolean;
}

function shuffle<T>(items: T[]): T[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function buildRoundOptions(case_: DemoCase, allCases: DemoCase[]): RoundOption[] {
  const target = case_.targetImage;
  const rawDistractors = [...(case_.challengeDistractors ?? [])].filter((p) => p && p !== target);
  const uniqueDistractors = [...new Set(rawDistractors)];

  const poolOthers = shuffle(
    allCases.filter((c) => c.id !== case_.id).map((c) => c.targetImage).filter((src) => src && src !== target),
  );

  const distractorPaths: string[] = [];
  for (const d of uniqueDistractors) {
    if (distractorPaths.length >= 3) break;
    if (!distractorPaths.includes(d)) distractorPaths.push(d);
  }
  for (const o of poolOthers) {
    if (distractorPaths.length >= 3) break;
    if (!distractorPaths.includes(o)) distractorPaths.push(o);
  }
  while (distractorPaths.length < 3) {
    distractorPaths.push('');
  }

  const options: RoundOption[] = [
    { src: targetingPath(target), isCorrect: true },
    ...distractorPaths.slice(0, 3).map((src) => ({ src: targetingPath(src), isCorrect: false })),
  ];
  return shuffle(options);
}

function targetingPath(src: string): string {
  return src?.trim() ?? '';
}

function pickCase(pool: DemoCase[], allCases: DemoCase[], excludeId?: string | null): DemoCase | null {
  const source = pool.length ? pool : allCases;
  if (!source.length) return null;
  let candidates = excludeId ? source.filter((c) => c.id !== excludeId) : source;
  if (excludeId && candidates.length === 0) {
    candidates = allCases.filter((c) => c.id !== excludeId);
  }
  if (candidates.length === 0) {
    return source[Math.floor(Math.random() * source.length)] ?? null;
  }
  return candidates[Math.floor(Math.random() * candidates.length)] ?? null;
}

function modelDisplaySrc(case_: DemoCase): string {
  if (case_.reconstructionImage?.trim()) return case_.reconstructionImage;
  const top1 = case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0];
  return top1?.image?.trim() ?? '';
}

export function ChallengeGame() {
  const { cases, pool, loading, error } = useChallengePool();

  const [gameStarted, setGameStarted] = useState(false);
  const [roundNumber, setRoundNumber] = useState(1);
  const [currentCase, setCurrentCase] = useState<DemoCase | null>(null);
  const [options, setOptions] = useState<RoundOption[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [stats, setStats] = useState({
    roundsPlayed: 0,
    committee: 0,
    model: 0,
    draws: 0,
  });

  const applyRoundCase = useCallback((nextCase: DemoCase) => {
    setCurrentCase(nextCase);
    setOptions(buildRoundOptions(nextCase, cases));
    setSelectedIndex(null);
    setRevealed(false);
  }, [cases]);

  const handleStartNewGame = useCallback(() => {
    if (!cases.length) return;
    const first = pickCase(pool.length ? pool : cases, cases) ?? cases[0];
    setStats({ roundsPlayed: 0, committee: 0, model: 0, draws: 0 });
    setRoundNumber(1);
    setGameStarted(true);
    applyRoundCase(first);
  }, [cases, pool, applyRoundCase]);

  const handlePlayAgain = useCallback(() => {
    handleStartNewGame();
  }, [handleStartNewGame]);

  const handleNextRound = useCallback(() => {
    if (!cases.length || !currentCase) return;
    const next = pickCase(pool.length ? pool : cases, cases, currentCase.id) ?? currentCase;
    setRoundNumber((r) => r + 1);
    applyRoundCase(next);
  }, [cases, pool, currentCase, applyRoundCase]);

  const handleSelect = useCallback(
    (index: number) => {
      if (revealed || !currentCase || selectedIndex !== null) return;
      setSelectedIndex(index);
      setRevealed(true);
      const committeeWon = options[index]?.isCorrect ?? false;
      const modelWon = currentCase.metrics.r1Correct;
      setStats((s) => ({
        roundsPlayed: s.roundsPlayed + 1,
        committee: s.committee + (committeeWon ? 1 : 0),
        model: s.model + (modelWon ? 1 : 0),
        draws: s.draws + (committeeWon && modelWon ? 1 : 0),
      }));
    },
    [revealed, currentCase, selectedIndex, options],
  );

  const revealContext = useMemo(() => {
    if (!currentCase || selectedIndex === null || !options.length) return null;
    const committeeCorrect = options[selectedIndex]?.isCorrect ?? false;
    const modelCorrect = currentCase.metrics.r1Correct;
    return {
      verdict: deriveVerdict(committeeCorrect, modelCorrect),
      committeeCorrect,
      modelCorrect,
      committeeChoiceSrc: options[selectedIndex]?.src ?? null,
    };
  }, [currentCase, selectedIndex, options]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  if (!cases.length) {
    return (
      <GlassCard className="p-8">
        <p className="text-sm text-slate-400">No demo cases loaded. Run data preparation and ensure `/data/demo_cases.json` is available.</p>
      </GlassCard>
    );
  }

  const reconSrc = currentCase ? modelDisplaySrc(currentCase) : '';
  const cardsDisabled = revealed || selectedIndex !== null;
  const nextDisabled = !revealed || !gameStarted;

  return (
    <div className="grid gap-8 lg:grid-cols-[1fr_min(100%,320px)] lg:items-start">
      <div className="space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">Round</p>
            <p className="font-mono text-2xl text-white">{gameStarted ? roundNumber : '—'}</p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={handleStartNewGame}
              className="rounded-lg border border-brain-accent/40 bg-brain-accent/10 px-4 py-2 text-xs font-semibold text-brain-accent transition hover:bg-brain-accent/20"
            >
              Start new game
            </button>
            <button
              type="button"
              onClick={handleNextRound}
              disabled={nextDisabled}
              className="rounded-lg border border-brain-border/60 bg-brain-panel/80 px-4 py-2 text-xs font-semibold text-slate-200 transition enabled:hover:border-fuchsia-400/50 enabled:hover:text-white disabled:cursor-not-allowed disabled:opacity-35"
            >
              Next round
            </button>
          </div>
        </div>

        {!gameStarted ? (
          <GlassCard className="p-8">
            <p className="text-sm leading-relaxed text-slate-400">
              You will see a single reconstruction decoded from fMRI-derived embeddings. Four candidate natural images appear
              below—only one was on screen during scanning. Select the image you believe matched the brain activity, then compare
              your judgment to the model’s top-1 retrieval and quantitative metrics.
            </p>
            <button
              type="button"
              onClick={handleStartNewGame}
              className="mt-6 rounded-lg bg-gradient-to-r from-brain-purple to-fuchsia-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-fuchsia-900/30 transition hover:brightness-110"
            >
              Begin challenge
            </button>
          </GlassCard>
        ) : (
          <>
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-panel overflow-hidden border border-brain-border/50"
            >
              <div className="border-b border-brain-border/40 px-5 py-4">
                <p className="text-xs font-medium text-slate-300">The model decoded this from brain activity</p>
                <p className="mt-1 text-[11px] text-slate-500">
                  Reconstruction conditioned on the predicted CLIP direction (diffusion). Candidate labels stay hidden until you choose.
                </p>
              </div>
              <div className="bg-[#050914] p-5">
                <div className="mx-auto max-w-md overflow-hidden rounded-xl border-2 border-fuchsia-500/50 shadow-[0_0_40px_rgba(192,38,211,0.18)] ring-1 ring-purple-500/30">
                  {reconSrc ? (
                    <ImagePlaceholder
                      src={reconSrc}
                      alt="Model reconstruction"
                      className="aspect-square border-0"
                      type="reconstruction"
                    />
                  ) : (
                    <div className="flex aspect-square items-center justify-center bg-gradient-to-br from-purple-950/50 to-brain-panel text-xs text-slate-500">
                      Reconstruction unavailable
                    </div>
                  )}
                </div>
              </div>
            </motion.div>

            <div>
              <p className="text-xs font-semibold tracking-wide text-slate-300">Which image was the subject actually viewing?</p>
              <p className="mt-1 text-[11px] text-slate-500">Click a card. Selection locks until you advance to the next round.</p>
              <div className="mt-4 grid grid-cols-2 gap-4">
                {options.map((opt, index) => (
                  <ChallengeCard
                    key={`${roundNumber}-${index}-${opt.src ? opt.src.slice(-32) : 'empty'}`}
                    src={opt.src}
                    alt={opt.isCorrect ? 'Ground truth candidate' : `Distractor ${index + 1}`}
                    index={index}
                    disabled={cardsDisabled}
                    isSelected={selectedIndex === index}
                    revealed={revealed}
                    isCorrect={opt.isCorrect}
                    onSelect={handleSelect}
                  />
                ))}
              </div>
            </div>

            <AnimatePresence mode="wait">
              {revealed && revealContext && currentCase && (
                <ChallengeResult
                  key={roundNumber}
                  verdict={revealContext.verdict}
                  committeeCorrect={revealContext.committeeCorrect}
                  modelCorrect={revealContext.modelCorrect}
                  case_={currentCase}
                  committeeChoiceSrc={revealContext.committeeChoiceSrc}
                  committeeChoiceLabel="Committee selection"
                />
              )}
            </AnimatePresence>
          </>
        )}
      </div>

      <ScoreBoard
        roundsPlayed={stats.roundsPlayed}
        committeeCorrect={stats.committee}
        modelCorrect={stats.model}
        draws={stats.draws}
        gameStarted={gameStarted}
        onPlayAgain={handlePlayAgain}
      />
    </div>
  );
}
