import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePipelineCases } from '@/lib/hooks';
import type { DemoCase } from '@/types';
import { PipelineProgressBar } from '@/components/pipeline/PipelineProgressBar';
import { PhaseTrialSelection } from '@/components/pipeline/PhaseTrialSelection';
import { PhaseEncodingRetrieval } from '@/components/pipeline/PhaseEncodingRetrieval';
import { PhaseReconstruction } from '@/components/pipeline/PhaseReconstruction';
import { LoadingState } from '@/components/LoadingState';
import { ErrorState } from '@/components/ErrorState';
import { checkBackendHealth, isBackendAvailable, type BackendHealth } from '@/lib/api';

type PipelinePhase = 'selection' | 'retrieval' | 'reconstruction';

export function Pipeline() {
  const { cases, loading, error } = usePipelineCases();
  const [phase, setPhase] = useState<PipelinePhase>('selection');
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [backendHealth, setBackendHealth] = useState<BackendHealth | null>(null);
  const [backendChecked, setBackendChecked] = useState(false);

  useEffect(() => {
    checkBackendHealth().then((h) => {
      setBackendHealth(h);
      setBackendChecked(true);
    });
  }, []);

  const progressBarPhase: 1 | 2 | 3 =
    phase === 'selection' ? 1 : phase === 'retrieval' ? 2 : 3;

  const backToSelection = useCallback(() => {
    setPhase('selection');
    setSelectedCase(null);
  }, []);

  const handleTrialSelected = useCallback((c: DemoCase) => {
    setSelectedCase(c);
    setPhase('retrieval');
  }, []);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;

  const liveMode = isBackendAvailable();

  return (
    <div className="min-h-screen px-4 py-6 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-7xl space-y-5">
        {/* Title */}
        <motion.div
          className="rounded-2xl border border-white/[0.06] bg-gradient-to-r from-slate-900/90 via-brain-navy/50 to-slate-900/90 px-6 py-5 backdrop-blur-xl"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
            <div>
              <h1 className="text-base font-bold uppercase tracking-[0.22em] text-white/95 sm:text-lg">
                Cortex2Canvas{' '}
                <span className="text-brain-accent">Live Pipeline</span>
              </h1>
              <p className="mt-1 text-xs text-slate-500">
                fMRI → CLIP retrieval → uncertainty-aware diffusion
              </p>
            </div>
            <div className="flex items-center gap-3">
              {backendChecked && (
                <span
                  className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-[10px] font-bold uppercase tracking-wider ${
                    liveMode
                      ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
                      : 'border-amber-500/30 bg-amber-500/8 text-amber-400'
                  }`}
                >
                  <motion.span
                    className={`inline-block h-1.5 w-1.5 rounded-full ${liveMode ? 'bg-emerald-400' : 'bg-amber-500'}`}
                    animate={liveMode ? { opacity: [1, 0.5, 1] } : {}}
                    transition={{ duration: 1.5, repeat: Infinity }}
                  />
                  {liveMode ? `Live — ${backendHealth?.device}` : 'Precomputed replay'}
                </span>
              )}
            </div>
          </div>
        </motion.div>

        <PipelineProgressBar currentPhase={progressBarPhase} />

        {/* Back button */}
        <AnimatePresence>
          {phase !== 'selection' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <button
                type="button"
                className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2 text-xs font-semibold text-slate-400 transition hover:border-brain-accent/30 hover:text-brain-accent"
                onClick={backToSelection}
              >
                ← Back to selection
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Phases */}
        <AnimatePresence mode="wait">
          {phase === 'selection' && (
            <motion.div key="p1" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 12 }}>
              <PhaseTrialSelection cases={cases} onSelect={handleTrialSelected} />
            </motion.div>
          )}
          {phase === 'retrieval' && selectedCase && (
            <motion.div key="p2" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 12 }}>
              <PhaseEncodingRetrieval case_={selectedCase} onComplete={() => setPhase('reconstruction')} />
            </motion.div>
          )}
          {phase === 'reconstruction' && selectedCase && (
            <motion.div key="p3" initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 12 }}>
              <PhaseReconstruction case_={selectedCase} onReset={backToSelection} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
