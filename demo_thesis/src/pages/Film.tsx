import { useCallback, useEffect, useMemo, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { useCases } from '@/lib/hooks';
import { GlassCard } from '@/components/GlassCard';
import { LoadingState } from '@/components/LoadingState';
import { ErrorState } from '@/components/ErrorState';
import { FilmTimeline, type FilmTimelineStepMeta } from '@/components/film/FilmTimeline';
import { FilmPipelineStrip } from '@/components/film/FilmPipelineStrip';
import { StepSelectTrial } from '@/components/film/steps/StepSelectTrial';
import { StepBrainActivation } from '@/components/film/steps/StepBrainActivation';
import { StepEmbedding } from '@/components/film/steps/StepEmbedding';
import { StepClipSpace } from '@/components/film/steps/StepClipSpace';
import { StepRetrieval } from '@/components/film/steps/StepRetrieval';
import { StepReconstruction } from '@/components/film/steps/StepReconstruction';
import { StepUncertainty } from '@/components/film/steps/StepUncertainty';
import { StepConclusion } from '@/components/film/steps/StepConclusion';

const STEP_COUNT = 8;
const SPEED_OPTIONS = [3000, 5000, 8000] as const;

const PLACEHOLDER_HINTS = ['placeholder', 'data:image/svg', '/1x1', 'empty.png'];

function targetLooksRealPath(c: DemoCase): boolean {
  const u = (c.targetImage ?? '').toLowerCase();
  if (u.length < 12) return false;
  return !PLACEHOLDER_HINTS.some((h) => u.includes(h));
}

function baseFilmScore(c: DemoCase): number {
  let s = 0;
  if (c.id === 'query_66976') s += 10_000;
  if (c.diffusionPrior && c.diffusionFinal) s += 3_000;
  const tier = { best: 400, medium: 300, hard: 200, mystery: 100 } as const;
  s += tier[c.difficulty] ?? 0;
  if (c.metrics.r1Correct) s += 80;
  if (c.retrievedImages?.length) s += 20;
  s += Math.round(c.metrics.cosine * 120);
  return s;
}

async function probeTargetBytes(url: string): Promise<number | null> {
  try {
    const r = await fetch(url, { method: 'HEAD' });
    const cl = r.headers.get('content-length');
    if (cl) return parseInt(cl, 10);
  } catch {
    /* static host may omit HEAD or block CORS — ignore */
  }
  return null;
}

/** Prefer true-diffusion cases, real (non-placeholder) targets &gt; ~15KB when HEAD works. */
async function pickDefaultCaseAsync(cases: DemoCase[]): Promise<DemoCase | null> {
  if (cases.length === 0) return null;
  const ranked = [...cases].sort((a, b) => baseFilmScore(b) - baseFilmScore(a));
  const headCandidates = ranked.slice(0, Math.min(12, ranked.length));
  const sizes = await Promise.all(headCandidates.map((c) => probeTargetBytes(c.targetImage)));

  for (let i = 0; i < headCandidates.length; i += 1) {
    const c = headCandidates[i];
    const bytes = sizes[i];
    if (!targetLooksRealPath(c)) continue;
    if (bytes !== null && bytes > 0 && bytes < 16_000) continue;
    return c;
  }
  return ranked[0];
}

/** Sync fallback when async probe has not run (e.g. tests). */
function pickDefaultCaseSync(cases: DemoCase[]): DemoCase | null {
  if (cases.length === 0) return null;
  const real = cases.filter(targetLooksRealPath);
  const pool = real.length > 0 ? real : cases;
  return [...pool].sort((a, b) => baseFilmScore(b) - baseFilmScore(a))[0] ?? null;
}

const TIMELINE_STEPS: FilmTimelineStepMeta[] = [
  { title: 'Incoming fMRI scan', shortLabel: 'fMRI' },
  { title: 'Visual cortex activation', shortLabel: 'Brain' },
  { title: 'Neural embedding decoding', shortLabel: 'Encode' },
  { title: 'CLIP vector space', shortLabel: 'CLIP' },
  { title: 'Retrieval candidates', shortLabel: 'Top-K' },
  { title: 'Diffusion reconstruction', shortLabel: 'Diffuse' },
  { title: 'Uncertainty-aware generation', shortLabel: 'UA' },
  { title: 'Decoding complete', shortLabel: 'Result' },
];

const stepVariants = {
  initial: { opacity: 0, x: 48, filter: 'blur(8px)' },
  animate: { opacity: 1, x: 0, filter: 'blur(0px)' },
  exit: { opacity: 0, x: -36, filter: 'blur(6px)' },
};

export function Film() {
  const navigate = useNavigate();
  const { cases, loading, error } = useCases();
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(5000);
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const syncDefaultIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (cases.length === 0) return;
    const sync = pickDefaultCaseSync(cases);
    syncDefaultIdRef.current = sync?.id ?? null;
    setSelectedCase((prev) => (prev && cases.some((c) => c.id === prev.id) ? prev : sync));
  }, [cases]);

  useEffect(() => {
    if (cases.length === 0) return;
    let cancelled = false;
    const lockedSyncId = syncDefaultIdRef.current;
    (async () => {
      const picked = await pickDefaultCaseAsync(cases);
      if (cancelled || !picked) return;
      setSelectedCase((prev) => {
        if (!prev || prev.id !== lockedSyncId) return prev;
        return picked;
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [cases]);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const id = window.setInterval(() => {
      setCurrentStep((s) => (s >= STEP_COUNT - 1 ? 0 : s + 1));
    }, speed);
    return () => window.clearInterval(id);
  }, [isPlaying, speed]);

  const togglePlay = useCallback(() => setIsPlaying((p) => !p), []);

  const goNext = useCallback(() => {
    setCurrentStep((s) => Math.min(STEP_COUNT - 1, s + 1));
  }, []);

  const goPrev = useCallback(() => {
    setCurrentStep((s) => Math.max(0, s - 1));
  }, []);

  const resetFilm = useCallback(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, []);

  const tryAnotherCase = useCallback(() => {
    if (cases.length < 2 || !selectedCase) return;
    const pool = cases.filter((c) => c.id !== selectedCase.id);
    const sorted = [...pool].sort((a, b) => baseFilmScore(b) - baseFilmScore(a));
    const pick = sorted[Math.floor(Math.random() * Math.min(4, sorted.length))] ?? sorted[0];
    if (pick) {
      setSelectedCase(pick);
      setCurrentStep(0);
      setIsPlaying(false);
    }
  }, [cases, selectedCase]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target;
      if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement) return;
      if (t instanceof HTMLSelectElement) return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        goNext();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        goPrev();
      } else if (e.key === 'r' || e.key === 'R') {
        if (!(t as HTMLElement).isContentEditable) resetFilm();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [togglePlay, goNext, goPrev, resetFilm]);

  const stepContent = useMemo(() => {
    if (!selectedCase) return null;
    switch (currentStep) {
      case 0:
        return <StepSelectTrial key="s0" case_={selectedCase} />;
      case 1:
        return <StepBrainActivation key="s1" case_={selectedCase} />;
      case 2:
        return <StepEmbedding key="s2" case_={selectedCase} />;
      case 3:
        return <StepClipSpace key="s3" case_={selectedCase} />;
      case 4:
        return <StepRetrieval key="s4" case_={selectedCase} />;
      case 5:
        return <StepReconstruction key="s5" case_={selectedCase} />;
      case 6:
        return <StepUncertainty key="s6" case_={selectedCase} />;
      case 7:
        return (
          <StepConclusion
            key="s7"
            case_={selectedCase}
            onTryAnotherCase={cases.length > 1 ? tryAnotherCase : undefined}
          />
        );
      default:
        return null;
    }
  }, [currentStep, selectedCase, tryAnotherCase]);

  if (loading) return <LoadingState message="Loading film data…" />;
  if (error) return <ErrorState message={error} />;
  if (!selectedCase) return <ErrorState message="No demo cases available." />;

  return (
    <div
      className="relative flex min-h-[calc(100vh-3.5rem)] flex-col overflow-hidden bg-[radial-gradient(ellipse_120%_80%_at_50%_-20%,rgba(34,211,238,0.12),transparent),radial-gradient(ellipse_80%_60%_at_100%_50%,rgba(139,92,246,0.08),transparent),linear-gradient(180deg,#030712_0%,#0a0f1a_45%,#04080f_100%)]"
    >
      <div className="flex flex-shrink-0 items-center justify-between gap-3 border-b border-white/5 px-4 py-3 sm:px-6">
        <div>
          <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-cyan-400/80">Cortex2Canvas</p>
          <h1 className="text-lg font-semibold text-white sm:text-xl">Film Mode</h1>
        </div>
        <button
          type="button"
          onClick={() => navigate('/')}
          className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 transition hover:border-cyan-500/30 hover:text-white"
        >
          Home
        </button>
      </div>

      <FilmPipelineStrip currentStep={currentStep} />

      <div className="relative flex min-h-0 flex-1 flex-col pb-36 pt-3">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            role="tabpanel"
            aria-label={TIMELINE_STEPS[currentStep]?.title}
            className="min-h-0 flex-1 overflow-y-auto px-2 sm:px-4"
            variants={stepVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
          >
            {stepContent}
          </motion.div>
        </AnimatePresence>
      </div>

      <div className="fixed bottom-0 left-0 right-0 z-20 border-t border-white/10 bg-slate-950/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl flex-col items-center gap-2 px-2 pb-3 pt-2">
          <p className="text-[9px] uppercase tracking-wider text-slate-600">Step detail</p>
          <FilmTimeline steps={TIMELINE_STEPS} currentStep={currentStep} onStepChange={setCurrentStep} />
        </div>
      </div>

      <motion.div
        className="fixed bottom-24 right-4 z-30 sm:bottom-28 sm:right-6"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <GlassCard className="flex flex-col gap-3 p-3 shadow-[0_8px_40px_rgba(0,0,0,0.45)]">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={goPrev}
              disabled={currentStep === 0}
              className="rounded-lg border border-white/15 bg-slate-900/80 px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-cyan-400/40 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Prev
            </button>
            <button
              type="button"
              onClick={togglePlay}
              className="min-w-[72px] rounded-lg border border-cyan-500/40 bg-cyan-500/15 px-3 py-2 text-xs font-semibold text-cyan-100 transition hover:bg-cyan-500/25"
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              type="button"
              onClick={goNext}
              disabled={currentStep === STEP_COUNT - 1}
              className="rounded-lg border border-white/15 bg-slate-900/80 px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-cyan-400/40 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Next
            </button>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">Auto-advance</span>
            <select
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="rounded-lg border border-white/10 bg-slate-900/90 px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-500/50"
            >
              {SPEED_OPTIONS.map((ms) => (
                <option key={ms} value={ms}>
                  {ms / 1000}s / step
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">Case</span>
            <select
              value={selectedCase.id}
              onChange={(e) => {
                const next = cases.find((c) => c.id === e.target.value);
                if (next) setSelectedCase(next);
                resetFilm();
              }}
              className="max-w-[200px] rounded-lg border border-white/10 bg-slate-900/90 px-2 py-1.5 text-xs text-slate-200 outline-none focus:border-cyan-500/50"
            >
              {cases.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.title} ({c.difficulty})
                </option>
              ))}
            </select>
          </div>
          <p className="text-[9px] leading-snug text-slate-600">Space play · ← → step · R reset</p>
        </GlassCard>
      </motion.div>
    </div>
  );
}
