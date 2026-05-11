import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase, RetrievedImage } from '@/types';
import {
  resolveNsdIdToTrial,
  streamInference,
  type InferenceStepEvent,
} from '@/lib/api';
import { ProvenanceBadge } from './ProvenanceBadge';
import {
  REPLAY_PROV,
  LIVE_PROV,
  DERIVED_PROV,
  UNKNOWN_PROV,
  type Provenance,
} from '@/lib/provenance';
import {
  hasFmriPreview,
  hasClipPreview,
  fmriPreviewProvenance,
  clipPreviewProvenance,
} from '@/lib/pipelineNormalize';

export interface PhaseEncodingRetrievalProps {
  case_: DemoCase;
  liveMode?: boolean;
  onComplete: () => void;
  modelMeta?: { encoder_type?: string; encoder_hidden?: number[]; embedding_dim?: number; model_type?: string } | null;
}

type Subphase =
  | 'load_betas'
  | 'zscore'
  | 'roi_mask'
  | 'roi_encode'
  | 'vmf_decode'
  | 'gallery_search'
  | 'results'
  | 'done';

interface StepDef {
  phase: Subphase;
  label: string;
  detail: string;
  duration: number;
}

function buildSteps(isLive: boolean, encoderType?: string): StepDef[] {
  const isMlp = !encoderType || encoderType === 'mlp' || encoderType === 'unknown';
  return [
    { phase: 'load_betas', label: 'Load fMRI betas', detail: 'ROI-masked beta vector for selected trial', duration: 1200 },
    { phase: 'zscore', label: 'Feature preprocessing', detail: isLive ? 'Pre-extracted features (z-score N/A)' : 'Per-session z-score normalization', duration: 900 },
    { phase: 'roi_mask', label: 'ROI masking', detail: isMlp ? 'Retain nsdgeneral visual cortex voxels' : 'Retain visual cortex voxels as ROI tokens', duration: 700 },
    { phase: 'roi_encode', label: isMlp ? 'MLP encoder' : 'ROI transformer', detail: isMlp ? '15,724 → [8192, 8192, 4096, 2048] residual MLP' : '17 ROI tokens → transformer encoder → [CLS]', duration: 1600 },
    { phase: 'vmf_decode', label: 'vMF projection', detail: 'Directional embedding on the unit hypersphere', duration: 1200 },
    { phase: 'gallery_search', label: 'CSLS gallery search', detail: 'Rank 10,000 CLIP embeddings with CSLS', duration: 1600 },
    { phase: 'results', label: 'Top-K hypotheses', detail: 'Select top visual hypotheses from gallery', duration: 1400 },
  ];
}

const DEFAULT_STEPS = buildSteps(false);

const ROI_TOKENS = [
  { name: 'V1v', c: '#22d3ee' }, { name: 'V1d', c: '#38bdf8' },
  { name: 'V2v', c: '#818cf8' }, { name: 'V2d', c: '#a78bfa' },
  { name: 'V3v', c: '#c084fc' }, { name: 'V3d', c: '#e879f9' },
  { name: 'V3A', c: '#f472b6' }, { name: 'V3B', c: '#34d399' },
  { name: 'V4', c: '#4ade80' }, { name: 'FFA1', c: '#fbbf24' },
  { name: 'FFA2', c: '#fb923c' }, { name: 'PPA', c: '#f87171' },
  { name: 'EBA', c: '#94a3b8' }, { name: 'OFA', c: '#64748b' },
  { name: 'OPA', c: '#2dd4bf' }, { name: 'RSC', c: '#60a5fa' },
  { name: 'other', c: '#f59e0b' },
] as const;

function normalizeArr(vals: number[]): number[] {
  let mn = Infinity, mx = -Infinity;
  for (const v of vals) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const s = mx - mn || 1;
  return vals.map((v) => ((v - mn) / s) * 100);
}

const STEP_PHASES: Subphase[] = ['load_betas', 'zscore', 'roi_mask', 'roi_encode', 'vmf_decode', 'gallery_search', 'results'];
function stepIdx(s: Subphase): number {
  return STEP_PHASES.indexOf(s);
}

function SafeImg({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  useEffect(() => { setOk(true); }, [src]);
  if (!src || !ok) return (
    <div className={`flex items-center justify-center bg-slate-900/80 ${className ?? ''}`}>
      <span className="text-[10px] text-slate-700">—</span>
    </div>
  );
  return <img src={src} alt={alt} className={className} onError={() => setOk(false)} loading="lazy" />;
}

function LabelBadge({ label }: { label: RetrievedImage['label'] }) {
  const m = label === 'correct'
    ? { t: 'Match', cls: 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300' }
    : label === 'semantic_neighbor'
      ? { t: 'Neighbor', cls: 'bg-amber-500/10 border-amber-500/25 text-amber-300' }
      : { t: 'Distractor', cls: 'bg-slate-600/15 border-slate-500/25 text-slate-400' };
  return (
    <span className={`rounded-md border px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider ${m.cls}`}>
      {m.t}
    </span>
  );
}

type StepStatus = 'pending' | 'active' | 'done';

export function PhaseEncodingRetrieval({
  case_,
  liveMode = false,
  onComplete,
  modelMeta,
}: PhaseEncodingRetrievalProps) {
  const encoderType = modelMeta?.encoder_type;
  const isMlpEncoder = !encoderType || encoderType === 'mlp' || encoderType === 'unknown';
  const STEPS = useMemo(() => buildSteps(liveMode, encoderType), [liveMode, encoderType]);
  const skipRef = useRef(false);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const [isLiveInference, setIsLiveInference] = useState(false);
  const [sub, setSub] = useState<Subphase>('load_betas');
  const [progress, setProgress] = useState(0);
  const [clipReveal, setClipReveal] = useState(false);
  const [galCount, setGalCount] = useState(0);
  const [visRanks, setVisRanks] = useState<number[]>([]);
  const [showProceed, setShowProceed] = useState(false);
  const [kappaStr, setKappaStr] = useState<string | null>(null);

  const hasRealFmri = hasFmriPreview(case_);
  const hasRealClip = hasClipPreview(case_);
  const fmriProv = fmriPreviewProvenance(case_);
  const clipProv = clipPreviewProvenance(case_);

  const fmriHeights = useMemo(() => {
    if (!hasRealFmri || !case_.fmriPreview?.length) return null;
    return normalizeArr(case_.fmriPreview);
  }, [case_.fmriPreview, hasRealFmri]);

  const clipHeights = useMemo(() => {
    if (!hasRealClip || !case_.clipPreview?.length) return null;
    return normalizeArr(case_.clipPreview);
  }, [case_.clipPreview, hasRealClip]);

  const topK = useMemo(
    () => [...case_.retrievedImages].sort((a, b) => a.rank - b.rank).slice(0, 5),
    [case_.retrievedImages],
  );

  const pushT = useCallback((fn: () => void, ms: number) => {
    const id = setTimeout(() => { if (!skipRef.current) fn(); }, ms);
    timersRef.current.push(id);
  }, []);
  const clearT = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  const skip = useCallback(() => {
    skipRef.current = true;
    clearT();
    setSub('done');
    setProgress(1);
    setClipReveal(true);
    setGalCount(10000);
    setVisRanks(topK.map((r) => r.rank));
    setKappaStr(case_.uncertainty.kappa.toFixed(1));
    setShowProceed(true);
  }, [clearT, topK, case_.uncertainty.kappa]);

  useEffect(() => {
    skipRef.current = false;
    clearT();
    setIsLiveInference(false);
    setSub('load_betas');
    setProgress(0);
    setClipReveal(false);
    setGalCount(0);
    setVisRanks([]);
    setKappaStr(null);
    setShowProceed(false);

    if (liveMode) {
      let cancelled = false;
      const STEP_MAP: Record<string, Subphase> = {
        load_fmri: 'load_betas', load_betas: 'load_betas',
        zscore: 'zscore', roi_mask: 'roi_mask',
        encode: 'roi_encode', roi_encode: 'roi_encode',
        vmf_decode: 'vmf_decode',
        retrieval: 'gallery_search', gallery_search: 'gallery_search',
        results: 'results', done: 'done',
      };
      (async () => {
        try {
          const trialIdx = await resolveNsdIdToTrial(case_.nsdId);
          if (cancelled || trialIdx == null) { if (!cancelled) runSimulation(); return; }
          setIsLiveInference(true);
          await streamInference(trialIdx, (ev: InferenceStepEvent) => {
            if (cancelled || skipRef.current) return;
            const mapped = STEP_MAP[ev.step] ?? 'load_betas';
            setSub(mapped);
            setProgress(ev.status === 'done' ? 1 : 0.5);
            if (ev.kappa != null) { setKappaStr(ev.kappa.toFixed(1)); setClipReveal(true); }
            if (ev.gallery_size) setGalCount(ev.gallery_size);
            if (ev.top_k?.length) { setVisRanks(ev.top_k.map((t) => t.rank)); setClipReveal(true); }
            if (ev.step === 'done' || ev.step === 'results') {
              if (ev.step === 'results' || ev.step === 'done') { setSub('done'); setProgress(1); setShowProceed(true); }
            }
          });
          if (!cancelled && !skipRef.current) { setSub('done'); setProgress(1); setShowProceed(true); }
        } catch {
          if (!cancelled) { setIsLiveInference(false); runSimulation(); }
        }
      })();
      return () => { cancelled = true; clearT(); };
    }

    runSimulation();
    return () => clearT();

    function runSimulation() {
      let t = 0;
      for (let i = 0; i <= 12; i++) pushT(() => setProgress(i / 12), t + i * 90);
      t += DEFAULT_STEPS[0].duration;
      pushT(() => { setSub('zscore'); setProgress(0); }, t);
      for (let i = 0; i <= 9; i++) pushT(() => setProgress(i / 9), t + i * 90);
      t += DEFAULT_STEPS[1].duration;
      pushT(() => { setSub('roi_mask'); setProgress(0); }, t);
      for (let i = 0; i <= 7; i++) pushT(() => setProgress(i / 7), t + i * 90);
      t += DEFAULT_STEPS[2].duration;
      pushT(() => { setSub('roi_encode'); setProgress(0); }, t);
      for (let i = 0; i <= 16; i++) pushT(() => setProgress(i / 16), t + i * 90);
      t += DEFAULT_STEPS[3].duration;
      pushT(() => { setSub('vmf_decode'); setProgress(0); setClipReveal(true); }, t);
      for (let i = 0; i <= 12; i++) pushT(() => setProgress(i / 12), t + i * 90);
      pushT(() => setKappaStr(case_.uncertainty.kappa.toFixed(1)), t + 800);
      t += DEFAULT_STEPS[4].duration;
      pushT(() => { setSub('gallery_search'); setProgress(0); }, t);
      for (let s = 0; s <= 40; s++) {
        pushT(() => { setGalCount(Math.round((1 - (1 - s / 40) ** 2) * 10000)); setProgress(s / 40); },
          t + (DEFAULT_STEPS[5].duration * s) / 40);
      }
      t += DEFAULT_STEPS[5].duration;
      pushT(() => { setSub('results'); setProgress(0); }, t);
      topK.forEach((_, i) => {
        pushT(() => {
          setVisRanks((prev) => { const r = topK[i].rank; return prev.includes(r) ? prev : [...prev, r]; });
          setProgress((i + 1) / topK.length);
        }, t + 250 * (i + 1));
      });
      t += DEFAULT_STEPS[6].duration;
      pushT(() => { setSub('done'); setShowProceed(true); }, t);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [case_.id, case_.nsdId, liveMode]);

  const ci = stepIdx(sub);
  const getStatus = (p: Subphase): StepStatus => {
    if (sub === 'done') return 'done';
    if (sub === p) return 'active';
    return ci > stepIdx(p) ? 'done' : 'pending';
  };

  const stepProv: Provenance = isLiveInference ? LIVE_PROV : REPLAY_PROV;
  const fmriStats = case_.fmriPreviewMeta?.stats;

  return (
    <div className="relative space-y-4">
      {/* ── Header ── */}
      <motion.header
        className="flex items-center justify-between rounded-2xl border border-white/[0.06] bg-gradient-to-r from-[#0c1222]/90 via-[#111a33]/80 to-[#0c1222]/90 px-6 py-4 backdrop-blur-xl"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-semibold tracking-tight text-white sm:text-lg">
              Encoding & retrieval
            </h2>
            <ProvenanceBadge provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Replayed computation trace' }} />
          </div>
          <p className="font-mono text-[11px] text-slate-500">
            {case_.subject} · nsdId {case_.nsdId} · session {case_.session}
          </p>
        </div>
        <button type="button" onClick={skip}
          className="rounded-lg border border-white/[0.08] bg-white/[0.03] px-4 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500 transition hover:border-cyan-500/30 hover:text-cyan-400">
          Skip
        </button>
      </motion.header>

      {/* ── Main layout: Rail + Canvas ── */}
      <div className="flex flex-col gap-4 lg:flex-row">

        {/* ── Zone A: Execution Rail ── */}
        <div className="w-full shrink-0 lg:w-64 xl:w-72">
          <div className="sticky top-4 rounded-2xl border border-white/[0.05] bg-[#0a0f1e]/80 px-3 py-3.5">
            <p className="mb-3 px-1 text-[9px] font-semibold uppercase tracking-[0.18em] text-slate-600">
              {isLiveInference ? 'Backend inference trace' : 'Computation replay'}
            </p>

            <div className="relative space-y-px">
              <div className="absolute left-[15px] top-3 bottom-3 w-px bg-gradient-to-b from-slate-800 via-slate-700/40 to-slate-800" />

              {STEPS.map((step, si) => {
                const s = getStatus(step.phase);
                return (
                  <motion.div
                    key={step.phase}
                    className={`relative flex items-start gap-2.5 rounded-lg px-2 py-[7px] transition-colors duration-200 ${
                      s === 'active' ? 'bg-cyan-500/[0.06]' : ''
                    }`}
                    animate={{ opacity: s === 'pending' ? 0.3 : 1 }}
                  >
                    <div className="relative z-10 mt-[3px] flex h-[16px] w-[16px] shrink-0 items-center justify-center">
                      {s === 'done' ? (
                        <motion.div
                          className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500/20"
                          initial={{ scale: 0.5 }} animate={{ scale: 1 }}
                        >
                          <svg className="h-2.5 w-2.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        </motion.div>
                      ) : s === 'active' ? (
                        <motion.div
                          className="h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.35)]"
                          animate={{ opacity: [1, 0.4, 1] }}
                          transition={{ duration: 1.2, repeat: Infinity }}
                        />
                      ) : (
                        <div className="h-1.5 w-1.5 rounded-full bg-slate-700" />
                      )}
                    </div>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className={`font-mono text-[9px] tabular-nums ${s === 'pending' ? 'text-slate-700' : 'text-slate-600'}`}>
                          {String(si + 1).padStart(2, '0')}
                        </span>
                        <span className={`text-[11px] font-medium leading-snug ${
                          s === 'active' ? 'text-cyan-300' : s === 'done' ? 'text-slate-300' : 'text-slate-600'
                        }`}>
                          {step.label}
                        </span>
                      </div>
                      {s !== 'pending' && (
                        <p className="mt-0.5 text-[9px] leading-snug text-slate-600">{step.detail}</p>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {sub !== 'done' && (
              <div className="mx-2 mt-2.5 h-[3px] overflow-hidden rounded-full bg-slate-800/50">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-cyan-600/80 to-cyan-400/60"
                  animate={{ width: `${Math.min(100, progress * 100)}%` }}
                  transition={{ duration: 0.08 }}
                />
              </div>
            )}
          </div>
        </div>

        {/* ── Zone B: Evidence Canvas ── */}
        <div className="min-h-[340px] flex-1 rounded-[1.375rem] border border-white/[0.05] bg-white/[0.018] px-5 py-5 sm:px-6">
          <AnimatePresence mode="wait">

            {/* fMRI preprocessing steps */}
            {(sub === 'load_betas' || sub === 'zscore' || sub === 'roi_mask') && (
              <motion.div key="fmri" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-[13px] font-semibold text-white">fMRI signal</h3>
                  <ProvenanceBadge provenance={hasRealFmri ? fmriProv : UNKNOWN_PROV} />
                </div>

                {/* Real fMRI preview */}
                {hasRealFmri && fmriHeights ? (
                  <div className="space-y-3">
                    <div className="overflow-hidden rounded-2xl bg-black/25 p-4">
                      <div className="flex h-20 items-end gap-[2px]">
                        {fmriHeights.map((h, i) => (
                          <motion.div
                            key={i}
                            className="min-w-0 flex-1 rounded-t-sm bg-gradient-to-t from-cyan-600/60 to-cyan-400/35"
                            initial={{ height: 0 }}
                            animate={{ height: `${Math.max(4, h)}%` }}
                            transition={{ delay: i * 0.003, duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                            style={{ minHeight: 2 }}
                          />
                        ))}
                      </div>
                      <div className="mt-3 flex items-center justify-between">
                        <p className="text-[9px] text-slate-500">
                          ROI feature vector · {case_.fmriPreview.length} bins
                        </p>
                        <p className="font-mono text-[9px] text-slate-600">
                          {fmriStats?.n_voxels?.toLocaleString() ?? '~15,724'} voxels
                        </p>
                      </div>
                    </div>

                    {/* Stats grid */}
                    {fmriStats && (
                      <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                        {([
                          ['Voxels', fmriStats.n_voxels?.toLocaleString()],
                          ['Mean', fmriStats.mean?.toFixed(1)],
                          ['Std', fmriStats.std?.toFixed(1)],
                          ['Min', fmriStats.min?.toFixed(1)],
                          ['Max', fmriStats.max?.toFixed(1)],
                          ['|Mean|', fmriStats.abs_mean?.toFixed(1)],
                        ] as [string, string | undefined][]).map(([label, val]) => (
                          <div key={label} className="rounded-xl bg-white/[0.03] px-3 py-2">
                            <p className="text-[8px] font-medium text-slate-600">{label}</p>
                            <p className="mt-0.5 font-mono text-[12px] font-semibold tabular-nums text-slate-200">{val ?? '—'}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex h-28 flex-col items-center justify-center gap-2 rounded-2xl bg-black/15">
                    <svg className="h-6 w-6 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
                    </svg>
                    <p className="text-[10px] text-slate-600">fMRI preview not exported for this trial</p>
                  </div>
                )}

                {/* Trial metadata */}
                <div className="flex flex-wrap gap-3 text-[10px]">
                  {([
                    ['Subject', case_.subject],
                    ['Session', String(case_.session)],
                    ['nsdId', String(case_.nsdId)],
                    ['ROI', 'nsdgeneral'],
                  ] as [string, string][]).map(([k, v]) => (
                    <span key={k} className="text-slate-600">{k} <span className="font-mono font-medium text-slate-400">{v}</span></span>
                  ))}
                </div>

                {/* Step-specific annotations */}
                {sub === 'zscore' && (
                  <div className="rounded-xl bg-white/[0.02] px-3 py-2 text-[10px] text-slate-500">
                    {isLiveInference ? 'Pre-extracted ROI features used directly — z-score not applied.' : 'Per-session z-scoring reduces scale differences before decoding.'}
                  </div>
                )}
                {sub === 'roi_mask' && (
                  <div className="space-y-2">
                    <p className="text-[10px] text-slate-500">
                      {fmriStats?.n_voxels ? `${fmriStats.n_voxels.toLocaleString()} voxels` : '~15,724 voxels'} retained from nsdgeneral ROI mask{!isMlpEncoder ? ' → 17 ROI tokens' : ''}
                    </p>
                    {!isMlpEncoder && (
                      <div className="flex flex-wrap gap-1">
                        {ROI_TOKENS.slice(0, 9).map((r) => (
                          <span key={r.name} className="rounded border border-white/[0.06] px-1.5 py-0.5 text-[8px] font-semibold" style={{ color: r.c }}>
                            {r.name}
                          </span>
                        ))}
                        <span className="rounded border border-white/[0.06] px-1.5 py-0.5 text-[8px] text-slate-600">
                          +{ROI_TOKENS.length - 9} more
                        </span>
                      </div>
                    )}
                  </div>
                )}
              </motion.div>
            )}

            {/* Encoder step */}
            {sub === 'roi_encode' && (
              <motion.div key="roi" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-[13px] font-semibold text-white">{isMlpEncoder ? 'MLP encoder' : 'ROI transformer'}</h3>
                  <ProvenanceBadge provenance={stepProv} />
                </div>

                <div className="overflow-hidden rounded-2xl bg-black/20 p-5">
                  <p className="mb-4 text-[10px] font-medium text-slate-500">Architecture</p>

                  {isMlpEncoder ? (
                    <>
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="rounded-lg border border-cyan-500/20 bg-cyan-950/20 px-3 py-1.5">
                          <p className="text-[10px] font-semibold text-cyan-300">15,724 voxels</p>
                        </div>
                        <svg className="h-3 w-5 shrink-0 text-slate-600" viewBox="0 0 20 12" fill="none">
                          <path d="M0 6h16m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="1.5" />
                        </svg>
                        {(modelMeta?.encoder_hidden ?? [8192, 8192, 4096, 2048]).map((d, i) => (
                          <div key={i} className="flex items-center gap-2">
                            <div className="rounded-lg border border-violet-500/20 bg-violet-950/20 px-2.5 py-1.5">
                              <p className="text-[10px] font-semibold text-violet-300">{d.toLocaleString()}</p>
                              <p className="text-[8px] text-slate-600">GELU + residual</p>
                            </div>
                            {i < (modelMeta?.encoder_hidden ?? [8192, 8192, 4096, 2048]).length - 1 && (
                              <svg className="h-3 w-4 shrink-0 text-slate-700" viewBox="0 0 16 12" fill="none">
                                <path d="M0 6h12m0 0l-3-3m3 3l-3 3" stroke="currentColor" strokeWidth="1" />
                              </svg>
                            )}
                          </div>
                        ))}
                        <svg className="h-3 w-5 shrink-0 text-slate-600" viewBox="0 0 20 12" fill="none">
                          <path d="M0 6h16m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="1.5" />
                        </svg>
                        <div className="rounded-lg border border-emerald-500/20 bg-emerald-950/20 px-3 py-1.5">
                          <p className="text-[10px] font-semibold text-emerald-300">{modelMeta?.embedding_dim ?? 768}-D</p>
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="mb-4 flex flex-wrap gap-1.5">
                        {ROI_TOKENS.map((r, i) => (
                          <motion.span key={r.name} className="rounded-md border border-white/[0.06] px-2 py-0.5 text-[9px] font-semibold"
                            style={{ color: r.c, borderColor: `${r.c}20` }}
                            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: i * 0.025, duration: 0.2 }}>
                            {r.name}
                          </motion.span>
                        ))}
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="rounded-lg border border-cyan-500/20 bg-cyan-950/20 px-3 py-1.5">
                          <p className="text-[10px] font-semibold text-cyan-300">17 ROI tokens</p>
                        </div>
                        <svg className="h-3 w-5 shrink-0 text-slate-600" viewBox="0 0 20 12" fill="none"><path d="M0 6h16m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="1.5" /></svg>
                        <div className="rounded-lg border border-violet-500/20 bg-violet-950/20 px-3 py-1.5">
                          <p className="text-[10px] font-semibold text-violet-300">Transformer ×6</p>
                          <p className="text-[8px] text-slate-600">d=768, 12 heads</p>
                        </div>
                        <svg className="h-3 w-5 shrink-0 text-slate-600" viewBox="0 0 20 12" fill="none"><path d="M0 6h16m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="1.5" /></svg>
                        <div className="rounded-lg border border-emerald-500/20 bg-emerald-950/20 px-3 py-1.5">
                          <p className="text-[10px] font-semibold text-emerald-300">[CLS] 768-D</p>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                <p className="text-[10px] italic text-slate-600">
                  Architecture schematic — not data. {isMlpEncoder ? 'The MLP encoder maps the full voxel vector to a 768-D representation via residual blocks.' : 'The ROI transformer processes brain-topology-aware tokens.'}
                </p>
              </motion.div>
            )}

            {/* vMF decode / CLIP embedding */}
            {sub === 'vmf_decode' && (
              <motion.div key="vmf" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-[13px] font-semibold text-white">vMF projection</h3>
                  <ProvenanceBadge provenance={DERIVED_PROV} />
                </div>

                {/* Predicted embedding status */}
                {clipReveal && !isLiveInference && (
                  <div className="rounded-xl bg-white/[0.025] px-3.5 py-2.5">
                    <p className="text-[10px] text-slate-500">
                      <span className="font-medium text-slate-400">Predicted embedding</span> — not exported in this replay. Requires live backend.
                    </p>
                  </div>
                )}

                {/* Reference CLIP — only if we have real data, shown with clear labeling */}
                {clipReveal && hasRealClip && clipHeights ? (
                  <div className="space-y-2">
                    <p className="text-[10px] font-medium text-slate-400">Reference CLIP embedding (target image)</p>
                    <div className="overflow-hidden rounded-2xl bg-black/25 p-4">
                      <div className="flex h-16 items-end gap-[2px]">
                        {clipHeights.map((u, i) => (
                          <motion.div
                            key={i}
                            className="min-w-0 flex-1 rounded-t-sm bg-gradient-to-t from-violet-600/55 to-fuchsia-400/40"
                            initial={{ height: 0 }}
                            animate={{ height: `${Math.max(4, u)}%` }}
                            transition={{ delay: i * 0.005, duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                            style={{ minHeight: 2 }}
                          />
                        ))}
                      </div>
                      <div className="mt-3 flex items-center justify-between">
                        <p className="text-[9px] text-slate-500">
                          CLIP ViT-L/14 · {case_.clipPreview?.length ?? 96} bins from 768-D
                        </p>
                        <p className="text-[8px] text-slate-600">gallery target, not prediction</p>
                      </div>
                    </div>
                    <ProvenanceBadge provenance={clipProv} />
                  </div>
                ) : null}

                {kappaStr && (
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-lg border border-cyan-500/20 bg-cyan-950/20 px-3 py-1 font-mono text-[11px] font-bold text-cyan-200">
                      κ = {kappaStr}
                    </span>
                    <span className="rounded-lg border border-amber-500/20 bg-amber-950/20 px-3 py-1 font-mono text-[11px] font-bold text-amber-200">
                      δ = {case_.uncertainty.delta.toFixed(3)}
                    </span>
                    <ProvenanceBadge provenance={DERIVED_PROV} />
                  </div>
                )}

                <div className="rounded-lg bg-violet-950/10 px-3 py-2 text-[10px] text-violet-300/60">
                  The vMF head maps the encoder representation to a direction on the CLIP unit hypersphere. The reference embedding shows the retrieval target for comparison.
                </div>
              </motion.div>
            )}

            {/* Gallery search */}
            {sub === 'gallery_search' && (
              <motion.div key="gallery" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-[13px] font-semibold text-white">CSLS gallery search</h3>
                  <ProvenanceBadge provenance={stepProv} />
                </div>

                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-[42px] font-bold tabular-nums leading-none text-white">
                    {galCount.toLocaleString()}
                  </span>
                  <span className="text-[12px] text-slate-600">/ 10,000 embeddings scored</span>
                </div>

                {/* Progress visualization */}
                <div className="h-1.5 overflow-hidden rounded-full bg-white/[0.04]">
                  <motion.div
                    className="h-full rounded-full bg-cyan-400/50"
                    animate={{ width: `${(galCount / 10000) * 100}%` }}
                    transition={{ duration: 0.1 }}
                  />
                </div>

                <div className="rounded-xl bg-white/[0.02] px-3.5 py-2.5 text-[10px] text-slate-500">
                  CSLS corrects for hubness — the tendency of certain gallery points to be nearest neighbors to too many queries.
                </div>
              </motion.div>
            )}

            {/* Results / done — rich summary */}
            {(sub === 'results' || sub === 'done') && (
              <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
                {/* ── Verdict header ── */}
                <div className="flex items-center justify-between">
                  <h3 className="text-[14px] font-semibold text-white">Retrieval complete</h3>
                  <ProvenanceBadge provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Cached retrieval ranking' }} />
                </div>

                {/* ── Hero rank verdict ── */}
                <motion.div
                  className={`rounded-2xl p-5 ${
                    case_.metrics.rank === 1
                      ? 'bg-emerald-500/[0.06] ring-1 ring-emerald-500/15'
                      : case_.metrics.rank != null && case_.metrics.rank <= 5
                        ? 'bg-amber-500/[0.04] ring-1 ring-amber-500/10'
                        : 'bg-white/[0.02] ring-1 ring-white/[0.04]'
                  }`}
                  initial={{ scale: 0.97, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 24 }}
                >
                  <div className="flex items-start gap-5">
                    {/* Top-1 thumbnail */}
                    {topK[0]?.image && (
                      <div className="shrink-0 overflow-hidden rounded-xl">
                        <SafeImg
                          src={topK[0].image}
                          alt="Top-1 retrieval"
                          className="h-20 w-20 object-cover"
                        />
                      </div>
                    )}
                    <div className="flex-1 space-y-2">
                      <div className="flex items-baseline gap-3">
                        <span className={`font-mono text-[36px] font-bold tabular-nums leading-none ${
                          case_.metrics.rank === 1 ? 'text-emerald-400' : case_.metrics.rank != null && case_.metrics.rank <= 5 ? 'text-amber-300' : 'text-white'
                        }`}>
                          #{case_.metrics.rank ?? '?'}
                        </span>
                        <div>
                          <p className={`text-[13px] font-semibold ${
                            case_.metrics.rank === 1 ? 'text-emerald-300' : case_.metrics.rank != null && case_.metrics.rank <= 5 ? 'text-amber-300' : 'text-slate-300'
                          }`}>
                            {case_.metrics.rank === 1 ? 'Exact match' : case_.metrics.rank != null && case_.metrics.rank <= 5 ? 'Near match' : 'Candidate identified'}
                          </p>
                          <p className="text-[10px] text-slate-500">
                            out of 10,000 gallery images via CSLS ranking
                          </p>
                        </div>
                      </div>
                      {/* R@1 indicator */}
                      {case_.metrics.r1Correct && (
                        <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 px-2 py-0.5 text-[9px] font-semibold text-emerald-400">
                          <svg className="h-3 w-3" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
                          Recall@1 correct
                        </span>
                      )}
                    </div>
                  </div>
                </motion.div>

                {/* ── Retrieval statistics grid ── */}
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div className="rounded-xl bg-white/[0.025] px-3.5 py-3">
                    <p className="text-[9px] font-medium text-slate-600">Gallery size</p>
                    <p className="mt-0.5 font-mono text-[16px] font-semibold text-white">10,000</p>
                  </div>
                  <div className="rounded-xl bg-white/[0.025] px-3.5 py-3">
                    <p className="text-[9px] font-medium text-slate-600">Scoring method</p>
                    <p className="mt-0.5 font-mono text-[13px] font-semibold text-white">CSLS</p>
                    <p className="text-[8px] text-slate-600">hubness-corrected</p>
                  </div>
                  <div className="rounded-xl bg-white/[0.025] px-3.5 py-3">
                    <p className="text-[9px] font-medium text-slate-600">Top-1 CSLS</p>
                    <p className="mt-0.5 font-mono text-[16px] font-semibold text-white">
                      {topK[0]?.csls != null ? topK[0].csls.toFixed(3) : '—'}
                    </p>
                  </div>
                  <div className="rounded-xl bg-white/[0.025] px-3.5 py-3">
                    <p className="text-[9px] font-medium text-slate-600">Candidates shown</p>
                    <p className="mt-0.5 font-mono text-[16px] font-semibold text-white">{topK.length}</p>
                  </div>
                </div>

                {/* ── Uncertainty parameters ── */}
                {kappaStr && (
                  <div className="rounded-2xl bg-white/[0.02] p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <p className="text-[10px] font-medium text-slate-500">Uncertainty parameters</p>
                      <ProvenanceBadge provenance={DERIVED_PROV} />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      {/* κ */}
                      <div>
                        <div className="flex items-baseline gap-2">
                          <span className="font-mono text-[11px] text-slate-500">κ</span>
                          <span className="font-mono text-xl font-semibold text-cyan-300">{kappaStr}</span>
                        </div>
                        <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-white/[0.04]">
                          <motion.div
                            className="h-full rounded-full bg-cyan-400/45"
                            initial={{ width: '0%' }}
                            animate={{ width: `${Math.min(100, (parseFloat(kappaStr) / 400) * 100)}%` }}
                            transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                          />
                        </div>
                        <p className="mt-1 text-[9px] text-slate-600">
                          {parseFloat(kappaStr) > 200 ? 'High' : parseFloat(kappaStr) > 80 ? 'Moderate' : 'Low'} directional certainty
                        </p>
                      </div>
                      {/* δ */}
                      <div>
                        <div className="flex items-baseline gap-2">
                          <span className="font-mono text-[11px] text-slate-500">δ</span>
                          {isLiveInference ? (
                            <span className="font-mono text-xl font-semibold text-slate-600">N/A</span>
                          ) : (
                            <span className="font-mono text-xl font-semibold text-amber-300/80">{case_.uncertainty.delta.toFixed(3)}</span>
                          )}
                        </div>
                        {!isLiveInference && (
                          <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-white/[0.04]">
                            <motion.div
                              className="h-full rounded-full bg-amber-400/40"
                              initial={{ width: '0%' }}
                              animate={{ width: `${Math.min(100, case_.uncertainty.delta * 500)}%` }}
                              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
                            />
                          </div>
                        )}
                        <p className="mt-1 text-[9px] text-slate-600">
                          {isLiveInference ? 'V62a MLP has no per-ROI δ' : case_.uncertainty.delta < 0.1 ? 'Low ROI disagreement' : 'Notable ROI tension'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* ── Pipeline trace ── */}
                <div className="flex flex-wrap items-center gap-1.5 text-[9px]">
                  {['fMRI betas', 'z-score', 'ROI mask', isMlpEncoder ? 'MLP encoder' : 'ROI transformer', 'vMF projection', 'CSLS search', 'Top-K'].map((step, i, arr) => (
                    <span key={step} className="flex items-center gap-1.5">
                      <span className="rounded-md bg-white/[0.04] px-1.5 py-0.5 font-medium text-slate-400">{step}</span>
                      {i < arr.length - 1 && (
                        <svg className="h-2.5 w-2.5 text-slate-700" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                        </svg>
                      )}
                    </span>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Zone C: Retrieval Filmstrip ── */}
      <motion.section
        className={`overflow-hidden rounded-2xl border transition-colors duration-300 ${
          sub === 'done' || getStatus('results') === 'done'
            ? 'border-white/[0.06] bg-gradient-to-b from-[#0c1222]/70 to-[#0a0f1e]/50'
            : getStatus('results') === 'active'
              ? 'border-cyan-500/10 bg-gradient-to-b from-cyan-950/10 to-[#0a0f1e]/40'
              : 'border-white/[0.03] bg-[#0a0f1e]/30'
        } p-5`}
        animate={{ opacity: getStatus('results') === 'pending' ? 0.25 : 1 }}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className={`text-[10px] font-semibold uppercase tracking-[0.16em] ${getStatus('results') !== 'pending' ? 'text-slate-300' : 'text-slate-700'}`}>
            Top-{topK.length} retrieved candidates
          </h3>
          {getStatus('results') === 'done' && (
            <ProvenanceBadge provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Precomputed gallery ranking' }} />
          )}
        </div>

        <div className={`grid gap-3 ${topK.length <= 3 ? 'grid-cols-1 sm:grid-cols-3' : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5'}`}>
          {topK.map((item) => {
            const vis = visRanks.includes(item.rank);
            return (
              <motion.div
                key={item.rank}
                className="group overflow-hidden rounded-xl border border-white/[0.05] bg-[#060b18]/80"
                initial={{ opacity: 0, scale: 0.92 }}
                animate={{ opacity: vis ? 1 : 0, scale: vis ? 1 : 0.92 }}
                transition={{ type: 'spring', stiffness: 300, damping: 25 }}
              >
                <div className="relative aspect-square w-full overflow-hidden">
                  <SafeImg src={item.image} alt={`Rank ${item.rank}`} className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]" />
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60 to-transparent pt-4 pb-1.5 px-2">
                    <div className="flex items-center justify-between">
                      <span className={`rounded px-1.5 py-0.5 text-[9px] font-bold text-white ${item.rank === 1 ? 'bg-cyan-600/80' : 'bg-slate-700/70'}`}>
                        #{item.rank}
                      </span>
                      <LabelBadge label={item.label} />
                    </div>
                  </div>
                </div>
                <div className="px-2.5 py-2">
                  <p className="font-mono text-[10px] leading-relaxed text-slate-500">
                    {item.score <= 1.0 ? `cos ${item.score.toFixed(3)} · ` : ''}csls {item.csls.toFixed(3)}
                  </p>
                </div>
              </motion.div>
            );
          })}
        </div>
      </motion.section>

      {/* ── Computation details (collapsed) ── */}
      <details className="group rounded-xl border border-white/[0.04] bg-[#0a0f1e]/40">
        <summary className="flex cursor-pointer items-center justify-between px-4 py-2.5 text-[10px] font-semibold uppercase tracking-wider text-slate-600 select-none">
          <span>Computation details</span>
          <span className="text-[8px] transition-transform group-open:rotate-180">▼</span>
        </summary>
        <div className="space-y-1.5 border-t border-white/[0.04] px-4 py-3 text-[10px] text-slate-500">
          <p><span className="font-medium text-slate-400">Mode:</span> {isLiveInference ? 'Live backend inference via /api/infer-stream' : 'Replay of real experiment outputs from cached JSON'}</p>
          <p><span className="font-medium text-slate-400">Data source:</span> {isLiveInference ? 'Backend returned embedding, top-K, κ during this session' : 'Cached experiment results loaded at page init (pipeline_cases.json)'}</p>
          <p><span className="font-medium text-slate-400">Encoder:</span> {isLiveInference ? (isMlpEncoder ? 'MLP encoder (V62a)' : 'ROI Transformer') : 'Architecture from replay case'}</p>
          <p><span className="font-medium text-slate-400">fMRI preview:</span> {hasRealFmri ? `Real ROI feature vector — binned to ${case_.fmriPreview.length} mean |activation| values` : 'Not available in this export'}</p>
          <p><span className="font-medium text-slate-400">Reference CLIP:</span> {hasRealClip ? `Real CLIP ViT-L/14 image embedding (gallery target) — compressed to ${case_.clipPreview?.length ?? 0} bins from 768-D` : 'Not available in this export'}</p>
          <p><span className="font-medium text-slate-400">Predicted CLIP:</span> {isLiveInference ? 'Returned by backend vMF head during this session' : 'Not exported — requires live backend inference'}</p>
          <p><span className="font-medium text-slate-400">κ:</span> {isLiveInference ? 'Live — returned by vMF decoder' : 'Derived from retrieval rank in build script'}</p>
          <p><span className="font-medium text-slate-400">δ (delta):</span> {isLiveInference ? 'Not available — V62a MLP model does not produce δ' : 'Derived from retrieval rank in build script'}</p>
          <p><span className="font-medium text-slate-400">Retrieval:</span> {isLiveInference ? 'Live CSLS search over 10,000 gallery embeddings' : 'Cached ranking from experiment output'}</p>
        </div>
      </details>

      {/* ── Proceed CTA ── */}
      <AnimatePresence>
        {showProceed && (
          <motion.div
            className="sticky bottom-4 z-10 mt-3 flex justify-center"
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 12 }}
          >
            <button type="button" onClick={onComplete}
              className="flex items-center gap-3 rounded-xl border border-emerald-400/30 bg-emerald-950/30 px-7 py-3.5 text-sm font-semibold text-emerald-200 shadow-[0_6px_30px_rgba(16,185,129,0.1)] backdrop-blur-xl transition hover:border-emerald-400/50 hover:shadow-[0_6px_40px_rgba(16,185,129,0.18)]">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/20 text-xs">→</span>
              Proceed to reconstruction
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
