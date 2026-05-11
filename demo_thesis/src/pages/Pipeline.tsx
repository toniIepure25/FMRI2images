import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePipelineCases } from '@/lib/hooks';
import type { DemoCase } from '@/types';
import type { PipelineRunMode } from '@/types/pipeline';
import { PipelineProgressBar } from '@/components/pipeline/PipelineProgressBar';
import { PipelineStatusHeader } from '@/components/pipeline/PipelineStatusHeader';
import { PhaseTrialSelection } from '@/components/pipeline/PhaseTrialSelection';
import { PhaseEncodingRetrieval } from '@/components/pipeline/PhaseEncodingRetrieval';
import { PhaseReconstruction } from '@/components/pipeline/PhaseReconstruction';
import { EvidenceIntegrityPanel } from '@/components/pipeline/EvidenceIntegrityPanel';
import { LoadingState } from '@/components/LoadingState';
import { ErrorState } from '@/components/ErrorState';
import {
  checkBackendHealth,
  isBackendAvailable,
  type BackendHealth,
} from '@/lib/api';

type PipelinePhase = 'selection' | 'retrieval' | 'reconstruction';

function resolveRunMode(h: BackendHealth | null): PipelineRunMode {
  if (!h) return 'offline-replay';
  if (h.live_retrieval_available || h.status === 'inference_ready') return 'hybrid';
  if (h.inference_available) return 'hybrid';
  if (h.server_online || h.status === 'data_ready' || h.status === 'server_online') return 'replay';
  return 'offline-replay';
}

export function Pipeline() {
  const { cases, loading, error } = usePipelineCases();
  const [phase, setPhase] = useState<PipelinePhase>('selection');
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [backendHealth, setBackendHealth] = useState<BackendHealth | null>(null);
  const [runMode, setRunMode] = useState<PipelineRunMode>('checking');

  useEffect(() => {
    checkBackendHealth()
      .then((h) => {
        setBackendHealth(h);
        setRunMode(resolveRunMode(h));
      })
      .catch(() => {
        setRunMode('offline-replay');
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
    <div className="pip-page-bg min-h-screen px-4 py-8 sm:px-6 lg:px-10">
      <div className="mx-auto max-w-[1280px] space-y-4">
        <PipelineStatusHeader
          runMode={runMode}
          backendHealth={backendHealth}
          selectedCase={selectedCase}
          phase={phase}
        />

        <PipelineProgressBar currentPhase={progressBarPhase} />

        <EvidenceIntegrityPanel selectedCase={selectedCase} liveMode={liveMode} backendHealth={backendHealth} />

        {/* Back button */}
        <AnimatePresence>
          {phase !== 'selection' && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <button
                type="button"
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-medium text-slate-500 transition hover:text-white"
                onClick={backToSelection}
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                </svg>
                Back to selection
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Phase content */}
        <AnimatePresence mode="wait">
          {phase === 'selection' && (
            <motion.div
              key="p1"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
              <PhaseTrialSelection
                cases={cases}
                onSelect={handleTrialSelected}
              />
            </motion.div>
          )}
          {phase === 'retrieval' && selectedCase && (
            <motion.div
              key="p2"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
              <PhaseEncodingRetrieval
                case_={selectedCase}
                liveMode={liveMode}
                onComplete={() => setPhase('reconstruction')}
                modelMeta={backendHealth?.model_meta ?? null}
              />
            </motion.div>
          )}
          {phase === 'reconstruction' && selectedCase && (
            <motion.div
              key="p3"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
            >
              <PhaseReconstruction
                case_={selectedCase}
                liveMode={liveMode}
                onReset={backToSelection}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
