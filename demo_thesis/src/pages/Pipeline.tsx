import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useSearchParams } from 'react-router-dom';
import { usePipelineCases } from '@/lib/hooks';
import type { DemoCase } from '@/types';
import type { PipelineRunMode } from '@/types/pipeline';
import { PipelineStatusHeader } from '@/components/pipeline/PipelineStatusHeader';
import { PhaseTrialSelection } from '@/components/pipeline/PhaseTrialSelection';
import { PhaseEncodingRetrieval } from '@/components/pipeline/PhaseEncodingRetrieval';
import { PhaseReconstruction } from '@/components/pipeline/PhaseReconstruction';
import { LoadingState } from '@/components/LoadingState';
import { ErrorState } from '@/components/ErrorState';
import {
  checkBackendHealth,
  isBackendAvailable,
  type BackendHealth,
} from '@/lib/api';

type PipelinePhase = 'selection' | 'retrieval' | 'reconstruction';
type PhaseParam = 'select' | 'decode' | 'evidence';

const PHASE_TO_PARAM: Record<PipelinePhase, PhaseParam> = {
  selection: 'select',
  retrieval: 'decode',
  reconstruction: 'evidence',
};
const PARAM_TO_PHASE: Record<PhaseParam, PipelinePhase> = {
  select: 'selection',
  decode: 'retrieval',
  evidence: 'reconstruction',
};

function resolveRunMode(_h: BackendHealth | null): PipelineRunMode {
  return 'live';
}

export function Pipeline() {
  const { cases, loading, error } = usePipelineCases();
  const [searchParams, setSearchParams] = useSearchParams();
  const [phase, setPhase] = useState<PipelinePhase>(() => {
    const p = searchParams.get('phase') as PhaseParam | null;
    return p && PARAM_TO_PHASE[p] ? PARAM_TO_PHASE[p] : 'selection';
  });
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [backendHealth, setBackendHealth] = useState<BackendHealth | null>(null);
  const [runMode, setRunMode] = useState<PipelineRunMode>('live');

  useEffect(() => {
    checkBackendHealth()
      .then((h) => {
        setBackendHealth(h);
        setRunMode('live');
      })
      .catch(() => {
        setRunMode('live');
      });
  }, []);

  // Restore selection from URL once cases load, so the deep-link
  // /pipeline?case=<id>&phase=<decode|evidence> survives a reload.
  useEffect(() => {
    if (cases.length === 0) return;
    const id = searchParams.get('case');
    if (!id) return;
    const found = cases.find((c) => c.id === id);
    if (found) {
      setSelectedCase(found);
      if (phase === 'selection') setPhase('retrieval');
    } else {
      // Stale id — clear it to keep the URL honest.
      const next = new URLSearchParams(searchParams);
      next.delete('case');
      next.delete('phase');
      setSearchParams(next, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cases]);

  const syncUrl = useCallback(
    (next: { case?: string | null; phase?: PipelinePhase | null }) => {
      const params = new URLSearchParams(searchParams);
      if (next.case === null) params.delete('case');
      else if (next.case) params.set('case', next.case);
      if (next.phase === null) params.delete('phase');
      else if (next.phase) params.set('phase', PHASE_TO_PARAM[next.phase]);
      setSearchParams(params, { replace: true });
    },
    [searchParams, setSearchParams],
  );

  const progressBarPhase: 1 | 2 | 3 =
    phase === 'selection' ? 1 : phase === 'retrieval' ? 2 : 3;

  const backToSelection = useCallback(() => {
    setPhase('selection');
    setSelectedCase(null);
    syncUrl({ case: null, phase: null });
  }, [syncUrl]);

  const handleTrialSelected = useCallback(
    (c: DemoCase) => {
      setSelectedCase(c);
      setPhase('retrieval');
      syncUrl({ case: c.id, phase: 'retrieval' });
    },
    [syncUrl],
  );

  const handleAdvanceToEvidence = useCallback(() => {
    setPhase('reconstruction');
    syncUrl({ phase: 'reconstruction' });
  }, [syncUrl]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;

  const liveMode = isBackendAvailable();

  return (
    <div className="pip-page-bg min-h-screen px-4 pb-14 pt-8 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-[1360px] space-y-3">
        <PipelineStatusHeader
          runMode={runMode}
          backendHealth={backendHealth}
          selectedCase={selectedCase}
          phase={phase}
          currentPhase={progressBarPhase}
          onBack={phase !== 'selection' ? backToSelection : undefined}
        />

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
                onComplete={handleAdvanceToEvidence}
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
