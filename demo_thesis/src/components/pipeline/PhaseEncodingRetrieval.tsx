import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase, RetrievedImage } from '@/types';
import { resolveNsdIdToTrial, streamInference, type InferenceStepEvent } from '@/lib/api';
import { ProvenanceBadge } from './ProvenanceBadge';
import { REPLAY_PROV, LIVE_PROV, DERIVED_PROV, UNKNOWN_PROV, type Provenance } from '@/lib/provenance';
import { hasFmriPreview, hasClipPreview, fmriPreviewProvenance, clipPreviewProvenance } from '@/lib/pipelineNormalize';

export interface PhaseEncodingRetrievalProps {
  case_: DemoCase;
  liveMode?: boolean;
  onComplete: () => void;
  modelMeta?: { encoder_type?: string; encoder_hidden?: number[]; embedding_dim?: number; model_type?: string } | null;
}

type Subphase = 'load_betas' | 'zscore' | 'roi_mask' | 'roi_encode' | 'vmf_decode' | 'gallery_search' | 'results' | 'done';

interface StepDef { phase: Subphase; label: string; detail: string; duration: number; }

function buildSteps(isLive: boolean, encoderType?: string): StepDef[] {
  const isMlp = !encoderType || encoderType === 'mlp' || encoderType === 'unknown';
  return [
    { phase: 'load_betas',    label: 'Load fMRI betas',  detail: 'ROI-masked beta vector for selected trial', duration: 1200 },
    { phase: 'zscore',        label: 'Preprocessing',     detail: isLive ? 'Pre-extracted features (z-score N/A)' : 'Per-session z-score normalization', duration: 900 },
    { phase: 'roi_mask',      label: 'ROI masking',       detail: isMlp ? 'Retain nsdgeneral visual cortex voxels' : 'Retain visual cortex voxels as ROI tokens', duration: 700 },
    { phase: 'roi_encode',    label: isMlp ? 'MLP encoder' : 'ROI transformer', detail: isMlp ? '15,724 → [8192, 8192, 4096, 2048] residual MLP' : '17 ROI tokens → transformer encoder → [CLS]', duration: 1600 },
    { phase: 'vmf_decode',    label: 'vMF projection',    detail: 'Directional embedding on the unit hypersphere', duration: 1200 },
    { phase: 'gallery_search',label: 'CSLS gallery search', detail: 'Rank 10,000 CLIP embeddings with CSLS', duration: 1600 },
    { phase: 'results',       label: 'Top-K hypotheses',  detail: 'Select top visual hypotheses from gallery', duration: 1400 },
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
function stepIdx(s: Subphase): number { return STEP_PHASES.indexOf(s); }

function SafeImg({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  useEffect(() => { setOk(true); }, [src]);
  if (!src || !ok) return (
    <div className={`flex items-center justify-center bg-surface-raised ${className ?? ''}`}>
      <span className="text-[10px] text-text-muted">&mdash;</span>
    </div>
  );
  return <img src={src} alt={alt} className={className} onError={() => setOk(false)} loading="lazy" />;
}

function LabelBadge({ label }: { label: RetrievedImage['label'] }) {
  const m = label === 'correct'
    ? { t: 'Match', cls: 'bg-status-success/10 text-status-success ring-1 ring-status-success/20' }
    : label === 'semantic_neighbor'
      ? { t: 'Neighbor', cls: 'bg-status-warning/8 text-status-warning ring-1 ring-status-warning/15' }
      : { t: 'Distractor', cls: 'bg-surface-active text-text-muted' };
  return (
    <span className={`rounded px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider ${m.cls}`}>{m.t}</span>
  );
}

type StepStatus = 'pending' | 'active' | 'done';

/* ─────────────────────────── Helper sub-components ─────────────────────── */

function SectionHeader({ title, provenance }: { title: string; provenance: Provenance }) {
  return (
    <div className="flex items-center justify-between">
      <h3 className="text-[15px] font-semibold text-text-primary">{title}</h3>
      <ProvenanceBadge provenance={provenance} />
    </div>
  );
}

function Explanation({ children }: { children: string }) {
  return (
    <div className="rounded-lg bg-surface-raised px-4 py-2.5 text-[12px] leading-relaxed text-text-muted">
      {children}
    </div>
  );
}

/* ─────────────────────────── Main component ───────────────────────────── */

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
    skipRef.current = true; clearT();
    setSub('done'); setProgress(1); setClipReveal(true); setGalCount(10000);
    setVisRanks(topK.map((r) => r.rank)); setKappaStr(case_.uncertainty.kappa.toFixed(1)); setShowProceed(true);
  }, [clearT, topK, case_.uncertainty.kappa]);

  useEffect(() => {
    skipRef.current = false; clearT();
    setIsLiveInference(false); setSub('load_betas'); setProgress(0);
    setClipReveal(false); setGalCount(0); setVisRanks([]); setKappaStr(null); setShowProceed(false);

    if (liveMode) {
      let cancelled = false;
      const STEP_MAP: Record<string, Subphase> = {
        load_fmri: 'load_betas', load_betas: 'load_betas', zscore: 'zscore', roi_mask: 'roi_mask',
        encode: 'roi_encode', roi_encode: 'roi_encode', vmf_decode: 'vmf_decode',
        retrieval: 'gallery_search', gallery_search: 'gallery_search', results: 'results', done: 'done',
      };
      (async () => {
        try {
          const trialIdx = await resolveNsdIdToTrial(case_.nsdId);
          if (cancelled || trialIdx == null) { if (!cancelled) runSimulation(); return; }
          setIsLiveInference(true);
          await streamInference(trialIdx, (ev: InferenceStepEvent) => {
            if (cancelled || skipRef.current) return;
            const mapped = STEP_MAP[ev.step] ?? 'load_betas';
            setSub(mapped); setProgress(ev.status === 'done' ? 1 : 0.5);
            if (ev.kappa != null) { setKappaStr(ev.kappa.toFixed(1)); setClipReveal(true); }
            if (ev.gallery_size) setGalCount(ev.gallery_size);
            if (ev.top_k?.length) { setVisRanks(ev.top_k.map((t) => t.rank)); setClipReveal(true); }
            if (ev.step === 'done' || ev.step === 'results') { setSub('done'); setProgress(1); setShowProceed(true); }
          });
          if (!cancelled && !skipRef.current) { setSub('done'); setProgress(1); setShowProceed(true); }
        } catch { if (!cancelled) { setIsLiveInference(false); runSimulation(); } }
      })();
      return () => { cancelled = true; clearT(); };
    }

    runSimulation();
    return () => clearT();

    function runSimulation() {
      let t = 0;
      for (let i = 0; i <= 12; i++) pushT(() => setProgress(i / 12), t + i * 90); t += DEFAULT_STEPS[0].duration;
      pushT(() => { setSub('zscore'); setProgress(0); }, t); for (let i = 0; i <= 9; i++) pushT(() => setProgress(i / 9), t + i * 90); t += DEFAULT_STEPS[1].duration;
      pushT(() => { setSub('roi_mask'); setProgress(0); }, t); for (let i = 0; i <= 7; i++) pushT(() => setProgress(i / 7), t + i * 90); t += DEFAULT_STEPS[2].duration;
      pushT(() => { setSub('roi_encode'); setProgress(0); }, t); for (let i = 0; i <= 16; i++) pushT(() => setProgress(i / 16), t + i * 90); t += DEFAULT_STEPS[3].duration;
      pushT(() => { setSub('vmf_decode'); setProgress(0); setClipReveal(true); }, t); for (let i = 0; i <= 12; i++) pushT(() => setProgress(i / 12), t + i * 90);
      pushT(() => setKappaStr(case_.uncertainty.kappa.toFixed(1)), t + 800); t += DEFAULT_STEPS[4].duration;
      pushT(() => { setSub('gallery_search'); setProgress(0); }, t);
      for (let s = 0; s <= 40; s++) { pushT(() => { setGalCount(Math.round((1 - (1 - s / 40) ** 2) * 10000)); setProgress(s / 40); }, t + (DEFAULT_STEPS[5].duration * s) / 40); }
      t += DEFAULT_STEPS[5].duration;
      pushT(() => { setSub('results'); setProgress(0); }, t);
      topK.forEach((_, i) => { pushT(() => { setVisRanks((prev) => { const r = topK[i].rank; return prev.includes(r) ? prev : [...prev, r]; }); setProgress((i + 1) / topK.length); }, t + 250 * (i + 1)); });
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
    <div className="relative space-y-5">
      {/* ── Phase header ── */}
      <motion.header
        className="flex items-center justify-between surface-card px-6 py-4"
        initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
      >
        <div className="space-y-0.5">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-semibold tracking-tight text-text-primary">Encoding &amp; retrieval</h2>
            <ProvenanceBadge provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Replayed computation trace' }} />
          </div>
          <p className="font-mono text-[12px] text-text-muted">{case_.subject} &middot; nsdId {case_.nsdId} &middot; session {case_.session}</p>
        </div>
        <button type="button" onClick={skip}
          className="rounded-lg border border-border-subtle bg-surface-raised px-4 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-text-muted transition hover:border-accent/25 hover:text-accent">
          Skip animation
        </button>
      </motion.header>

      {/* ── Rail + Canvas ── */}
      <div className="flex flex-col gap-5 lg:flex-row">

        {/* Zone A: Execution rail */}
        <div className="w-full shrink-0 lg:w-64 xl:w-72">
          <div className="sticky top-4 rounded-xl border border-border-subtle bg-surface-raised px-3 py-4">
            <p className="mb-3 px-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-text-muted">
              {isLiveInference ? 'Backend inference' : 'Computation replay'}
            </p>
            <div className="relative space-y-px">
              <div className="absolute left-[15px] top-3 bottom-3 w-px bg-border-subtle" />
              {STEPS.map((step, si) => {
                const s = getStatus(step.phase);
                return (
                  <motion.div
                    key={step.phase}
                    className={`relative flex items-start gap-2.5 rounded-lg px-2 py-[7px] transition-colors duration-200 ${
                      s === 'active' ? 'bg-accent/[0.06]' : ''
                    }`}
                    animate={{ opacity: s === 'pending' ? 0.35 : 1 }}
                  >
                    <div className="relative z-10 mt-[3px] flex h-[16px] w-[16px] shrink-0 items-center justify-center">
                      {s === 'done' ? (
                        <motion.div className="flex h-4 w-4 items-center justify-center rounded-full bg-accent/15" initial={{ scale: 0.5 }} animate={{ scale: 1 }}>
                          <svg className="h-2.5 w-2.5 text-accent" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                        </motion.div>
                      ) : s === 'active' ? (
                        <motion.div className="h-2 w-2 rounded-full bg-accent" animate={{ opacity: [1, 0.4, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
                      ) : (
                        <div className="h-1.5 w-1.5 rounded-full bg-border-emphasis" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className={`font-mono text-[9px] tabular-nums ${s === 'pending' ? 'text-text-muted/50' : 'text-text-muted'}`}>
                          {String(si + 1).padStart(2, '0')}
                        </span>
                        <span className={`text-[11px] font-medium leading-snug ${
                          s === 'active' ? 'text-text-primary' : s === 'done' ? 'text-text-secondary' : 'text-text-muted'
                        }`}>{step.label}</span>
                      </div>
                      {s !== 'pending' && <p className="mt-0.5 text-[9px] leading-snug text-text-muted">{step.detail}</p>}
                    </div>
                  </motion.div>
                );
              })}
            </div>
            {sub !== 'done' && (
              <div className="mx-2 mt-3 h-[3px] overflow-hidden rounded-full bg-surface-active">
                <motion.div className="h-full rounded-full bg-accent/50" animate={{ width: `${Math.min(100, progress * 100)}%` }} transition={{ duration: 0.08 }} />
              </div>
            )}
          </div>
        </div>

        {/* Zone B: Evidence canvas */}
        <div className="min-h-[400px] flex-1 rounded-xl border border-border-subtle bg-surface-elevated px-6 py-6">
          <AnimatePresence mode="wait">

            {/* ── 01 / 02 / 03: fMRI signal + prep ── */}
            {(sub === 'load_betas' || sub === 'zscore' || sub === 'roi_mask') && (
              <motion.div key="fmri" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-6">
                {/* Title for current active substep */}
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold text-text-primary">
                      {sub === 'load_betas' ? 'fMRI signal acquisition' : sub === 'zscore' ? 'Feature preprocessing' : 'ROI masking'}
                    </h3>
                    <p className="mt-1 text-[13px] text-text-muted">
                      {sub === 'load_betas'
                        ? 'Trial-specific ROI beta vector loaded from cached experiment assets.'
                        : sub === 'zscore'
                          ? (isLiveInference ? 'Pre-extracted features used directly — z-score not applied.' : 'Per-session z-score normalization stabilizes feature scale before decoding.')
                          : 'Retain nsdgeneral visual cortex voxels for the model input.'}
                    </p>
                  </div>
                  <ProvenanceBadge provenance={hasRealFmri ? fmriProv : UNKNOWN_PROV} />
                </div>

                {hasRealFmri && fmriHeights ? (
                  <div className="space-y-4">
                    {/* ── fMRI waveform ── */}
                    <div className="overflow-hidden rounded-xl border border-border-subtle bg-surface-base p-5">
                      <div className="flex items-center justify-between mb-3">
                        <p className="text-[12px] font-semibold text-text-primary">ROI feature vector</p>
                        <p className="font-mono text-[12px] text-text-secondary">{fmriStats?.n_voxels?.toLocaleString() ?? '~15,724'} voxels</p>
                      </div>
                      <div className="flex h-36 items-end gap-[2px]">
                        {fmriHeights.map((h, i) => (
                          <motion.div
                            key={i} className="min-w-0 flex-1 rounded-t-sm bg-accent/55"
                            initial={{ height: 0 }} animate={{ height: `${Math.max(4, h)}%` }}
                            transition={{ delay: i * 0.0015, duration: 0.35, ease: [0.22, 1, 0.36, 1] }} style={{ minHeight: 2 }}
                          />
                        ))}
                      </div>
                      <div className="mt-3 flex items-center justify-between">
                        <p className="text-[11px] text-text-secondary">{case_.fmriPreview.length} bins from ROI-masked feature vector</p>
                        <p className="text-[11px] text-accent/70">&#9650; = higher activation</p>
                      </div>
                    </div>

                    {/* Stats grid */}
                    {fmriStats && (
                      <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
                        {(['Voxels', 'Mean', 'Std', 'Min', 'Max', '|Mean|'] as const).map((label) => {
                          const key = label === '|Mean|' ? 'abs_mean' : label.toLowerCase();
                          const val = fmriStats[key as keyof typeof fmriStats];
                          return (
                            <div key={label} className="rounded-lg bg-surface-raised px-3 py-2.5">
                              <p className="text-[9px] font-medium text-text-muted">{label}</p>
                              <p className="mt-0.5 font-mono text-[14px] font-semibold text-text-primary">
                                {val != null ? (typeof val === 'number' ? val.toFixed(1) : String(val)) : '—'}
                              </p>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Metadata bar */}
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px]">
                      {(['Subject', 'Session', 'nsdId', 'ROI'] as const).map((k) => {
                        const vals: Record<string, string> = { Subject: case_.subject, Session: String(case_.session), nsdId: String(case_.nsdId), ROI: 'nsdgeneral' };
                        return (
                          <span key={k} className="text-text-secondary">{k} <span className="font-mono font-medium text-text-primary">{vals[k]}</span></span>
                        );
                      })}
                    </div>

                    {/* Preprocessing visual */}
                    {sub === 'zscore' && (
                      <div className="rounded-xl border border-border-subtle bg-surface-raised p-5">
                        <p className="text-[12px] font-semibold text-text-primary mb-4">z-score normalization</p>
                        <div className="flex flex-wrap items-center gap-4 text-[11px]">
                          <div className="flex flex-col items-center gap-1 rounded-lg bg-surface-elevated px-4 py-3">
                            <span className="font-mono text-text-secondary">Raw features</span>
                            <span className="text-[9px] text-text-muted">variable scale</span>
                          </div>
                          <span className="text-border-emphasis text-lg font-bold">&rarr;</span>
                          <div className="flex flex-col items-center gap-1 rounded-lg border border-accent/15 bg-accent/[0.04] px-4 py-3">
                            <span className="font-mono text-accent">(x − μ) / σ</span>
                            <span className="text-[9px] text-text-muted">per-session</span>
                          </div>
                          <span className="text-border-emphasis text-lg font-bold">&rarr;</span>
                          <div className="flex flex-col items-center gap-1 rounded-lg bg-surface-elevated px-4 py-3">
                            <span className="font-mono text-text-secondary">Normalized</span>
                            <span className="text-[9px] text-text-muted">μ=0, σ=1</span>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* ROI masking visual */}
                    {sub === 'roi_mask' && (
                      <div className="rounded-xl border border-border-subtle bg-surface-raised p-5">
                        <p className="text-[12px] font-semibold text-text-primary mb-4">nsdgeneral ROI mask</p>
                        <div className="flex flex-wrap items-center gap-3">
                          <div className="flex flex-col items-center gap-1 rounded-lg bg-surface-elevated px-4 py-3">
                            <span className="font-mono text-text-secondary">Full brain</span>
                            <span className="text-[9px] text-text-muted">~150,000+ voxels</span>
                          </div>
                          <span className="text-border-emphasis text-lg font-bold">&rarr;</span>
                          <div className="flex flex-col items-center gap-1 rounded-lg border border-accent/15 bg-accent/[0.04] px-4 py-3">
                            <span className="font-mono text-accent">nsdgeneral ROI</span>
                            <span className="text-[9px] text-text-muted">visual cortex mask</span>
                          </div>
                          <span className="text-border-emphasis text-lg font-bold">&rarr;</span>
                          <div className="flex flex-col items-center gap-1 rounded-lg bg-surface-elevated px-4 py-3">
                            <span className="font-mono text-text-secondary font-bold">
                              {fmriStats?.n_voxels ? `${fmriStats.n_voxels.toLocaleString()}` : '15,724'}
                            </span>
                            <span className="text-[9px] text-text-muted">retained voxels</span>
                          </div>
                        </div>
                        <p className="mt-3 text-[11px] text-text-muted">The model uses only the nsdgeneral visual cortex ROI rather than all brain voxels.</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex h-40 flex-col items-center justify-center gap-3 rounded-xl bg-surface-base border border-border-subtle">
                    <svg className="h-9 w-9 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
                    </svg>
                    <p className="text-[13px] font-medium text-text-muted">fMRI preview not exported for this trial</p>
                    <p className="text-[11px] text-text-muted">The beta vector for NSD trial {case_.nsdId} was not included in the replay assets.</p>
                  </div>
                )}
              </motion.div>
            )}

            {/* ── 04: MLP ENCODER ── */}
            {sub === 'roi_encode' && (
              <motion.div key="roi" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-6">
                <SectionHeader title={isMlpEncoder ? 'MLP encoder' : 'ROI transformer'} provenance={stepProv} />

                {isMlpEncoder ? (
                  <div className="space-y-4">
                    <p className="text-[12px] font-medium text-text-muted">Architecture — residual MLP</p>

                    {/* ── Single continuous horizontal flow ── */}
                    <div className="flex flex-wrap items-stretch gap-2 xl:flex-nowrap">
                      {/* Input */}
                      <div className="flex min-w-[120px] flex-shrink-0 flex-col items-center justify-center rounded-xl border border-accent/20 bg-accent/[0.05] px-4 py-5">
                        <span className="text-[10px] font-semibold text-accent">Input</span>
                        <span className="mt-1 font-mono text-2xl font-bold text-text-primary">15,724</span>
                        <span className="text-[9px] text-text-muted">ROI voxels</span>
                      </div>

                      {/* Arrow */}
                      <div className="flex items-center justify-center px-1">
                        <svg className="h-6 w-6 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
                      </div>

                      {/* Residual MLP block — all hidden layers inside one container */}
                      <div className="flex flex-1 flex-col gap-1 rounded-xl border border-border-subtle bg-surface-base p-3">
                        <p className="mb-1 text-center text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">Residual MLP</p>
                        <div className="flex flex-1 items-stretch gap-1">
                          {(modelMeta?.encoder_hidden ?? [8192, 8192, 4096, 2048]).map((d, i) => (
                            <div key={i} className="flex flex-1 items-center gap-1">
                              <div className="flex flex-1 flex-col items-center justify-center rounded-lg bg-surface-raised px-2 py-3">
                                <span className="font-mono text-sm font-bold tabular-nums text-text-primary">{d.toLocaleString()}</span>
                                <span className="mt-0.5 text-[9px] text-text-muted">Layer {i + 1}</span>
                                <span className="text-[8px] text-text-muted">GELU+res</span>
                              </div>
                              {i < (modelMeta?.encoder_hidden ?? [8192, 8192, 4096, 2048]).length - 1 && (
                                <div className="flex w-5 shrink-0 items-center justify-center">
                                  <svg className="h-3.5 w-3.5 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Arrow */}
                      <div className="flex items-center justify-center px-1">
                        <svg className="h-6 w-6 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
                      </div>

                      {/* Output */}
                      <div className="flex min-w-[100px] flex-shrink-0 flex-col items-center justify-center rounded-xl border border-accent/20 bg-accent/[0.05] px-4 py-5">
                        <span className="text-[10px] font-semibold text-accent">CLIP latent</span>
                        <span className="mt-1 font-mono text-2xl font-bold text-text-primary">{modelMeta?.embedding_dim ?? 768}-D</span>
                      </div>
                    </div>

                    {/* vMF head note */}
                    <div className="flex items-center gap-3 rounded-lg bg-surface-raised px-4 py-2.5">
                      <span className="font-mono text-[10px] text-text-muted">&rarr; vMF projection</span>
                      <span className="text-[11px] text-text-muted">Unit hypersphere mapping with learnable concentration κ</span>
                    </div>

                    <p className="text-[11px] text-text-muted">Residual MLP maps the full voxel vector into the CLIP embedding space via four learned layers.</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="flex flex-wrap gap-1.5">
                      {ROI_TOKENS.map((r, i) => (
                        <motion.span key={r.name} className="rounded border px-2 py-1 text-[9px] font-semibold"
                          style={{ color: r.c, borderColor: `${r.c}20` }}
                          initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.02, duration: 0.2 }}>
                          {r.name}
                        </motion.span>
                      ))}
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="rounded-lg border border-accent/20 bg-accent/[0.05] px-3 py-2"><p className="text-[10px] font-semibold text-accent">17 ROI tokens</p></div>
                      <svg className="h-3 w-5 shrink-0 text-border-emphasis" viewBox="0 0 20 12" fill="none"><path d="M0 6h16m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="1.5" /></svg>
                      <div className="rounded-lg border border-accent/15 bg-accent/[0.04] px-3 py-2"><p className="text-[10px] font-semibold text-text-secondary">Transformer ×6</p><p className="text-[8px] text-text-muted">d=768, 12 heads</p></div>
                      <svg className="h-3 w-5 shrink-0 text-border-emphasis" viewBox="0 0 20 12" fill="none"><path d="M0 6h16m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="1.5" /></svg>
                      <div className="rounded-lg border border-accent/20 bg-accent/[0.05] px-3 py-2"><p className="text-[10px] font-semibold text-accent">[CLS] 768-D</p></div>
                    </div>
                    <p className="text-[11px] text-text-muted">ROI transformer processes brain-topology-aware tokens through 6 transformer layers.</p>
                  </div>
                )}
              </motion.div>
            )}

            {/* ── 05: vMF projection ── */}
            {sub === 'vmf_decode' && (
              <motion.div key="vmf" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-6">
                <SectionHeader title="vMF projection" provenance={DERIVED_PROV} />

                {/* ── Hypersphere schematic + κ/δ ── */}
                <div className="flex flex-col gap-5 sm:flex-row sm:items-start">
                  {/* SVG unit hypersphere — larger */}
                  <div className="flex shrink-0 items-center justify-center">
                    <svg viewBox="0 0 160 160" className="h-44 w-44">
                      {/* Sphere */}
                      <defs>
                        <radialGradient id="sphereGrad2" cx="35%" cy="35%">
                          <stop offset="0%" stopColor="rgb(77,124,255)" stopOpacity="0.22" />
                          <stop offset="70%" stopColor="rgb(77,124,255)" stopOpacity="0.06" />
                          <stop offset="100%" stopColor="rgb(77,124,255)" stopOpacity="0" />
                        </radialGradient>
                      </defs>
                      <circle cx="80" cy="80" r="70" fill="url(#sphereGrad2)" stroke="rgb(77,124,255)" strokeWidth="1.5" strokeOpacity="0.3" />
                      <circle cx="80" cy="80" r="70" fill="none" stroke="rgb(77,124,255)" strokeWidth="0.5" strokeOpacity="0.1" strokeDasharray="4 6" />
                      {/* Equator hint */}
                      <ellipse cx="80" cy="80" rx="70" ry="25" fill="none" stroke="rgb(77,124,255)" strokeWidth="0.5" strokeOpacity="0.08" />
                      {/* Direction μ */}
                      <line x1="80" y1="80" x2="120" y2="32" stroke="rgb(77,124,255)" strokeWidth="2.5" strokeOpacity="0.7" strokeLinecap="round" />
                      <circle cx="120" cy="32" r="5" fill="rgb(77,124,255)" fillOpacity="0.9" />
                      <text x="128" y="30" className="fill-accent text-[11px] font-mono font-bold" style={{ fontFamily: 'JetBrains Mono' }}>μ</text>
                      {/* κ ring */}
                      <circle cx="120" cy="32" r="16" fill="none" stroke="rgb(77,124,255)" strokeWidth="1.2" strokeOpacity="0.25" strokeDasharray="3 2" />
                      {/* Origin */}
                      <circle cx="80" cy="80" r="3" fill="rgb(148,163,184)" />
                      <text x="80" y="100" textAnchor="middle" className="fill-text-secondary text-[9px] font-mono" style={{ fontFamily: 'JetBrains Mono' }}>CLIP S²</text>
                    </svg>
                  </div>

                  {/* Explanation + values */}
                  <div className="flex-1 space-y-4">
                    <div>
                      <p className="text-[14px] font-semibold text-text-primary">Unit hypersphere projection</p>
                      <p className="mt-1 text-[12px] leading-relaxed text-text-muted">
                        The encoder output is mapped to a direction μ on the CLIP unit hypersphere. The vMF concentration κ measures how sharply the distribution peaks around μ.
                      </p>
                    </div>

                    {clipReveal && !isLiveInference && (
                      <div className="rounded-lg bg-surface-raised px-4 py-3">
                        <p className="text-[12px] text-text-muted">
                          <span className="font-medium text-text-secondary">Predicted embedding</span> — not exported in this replay. Requires live backend.
                        </p>
                      </div>
                    )}

                    {kappaStr && (
                      <div className="flex flex-wrap items-center gap-3">
                        <span className="rounded-lg border border-accent/15 bg-accent/[0.05] px-4 py-2 font-mono text-[14px] font-semibold text-accent">κ = {kappaStr}</span>
                        <span className="rounded-lg border border-status-warning/15 bg-status-warning/[0.05] px-4 py-2 font-mono text-[14px] font-semibold text-status-warning">δ = {case_.uncertainty.delta.toFixed(3)}</span>
                        <ProvenanceBadge provenance={DERIVED_PROV} />
                      </div>
                    )}

                    <div className="flex flex-wrap gap-2 text-[10px]">
                      <span className="rounded bg-surface-raised px-2 py-1 text-text-muted">768-D latent</span>
                      <span className="text-border-emphasis">&rarr;</span>
                      <span className="rounded bg-surface-raised px-2 py-1 text-accent">μ direction</span>
                      <span className="text-border-emphasis">+</span>
                      <span className="rounded bg-surface-raised px-2 py-1 text-accent">κ concentration</span>
                    </div>
                  </div>
                </div>

                {/* Reference CLIP bar chart (clearly labeled as distinct) */}
                {clipReveal && hasRealClip && clipHeights ? (
                  <div className="space-y-2 rounded-xl border border-border-subtle bg-surface-base p-4">
                    <div className="flex items-center justify-between">
                      <p className="text-[11px] font-medium text-text-muted">Reference CLIP embedding (gallery target — NOT the predicted μ)</p>
                      <ProvenanceBadge provenance={clipProv} />
                    </div>
                    <div className="flex h-16 items-end gap-[2px]">
                      {clipHeights.map((u, i) => (
                        <motion.div key={i} className="min-w-0 flex-1 rounded-t-sm bg-accent/30"
                          initial={{ height: 0 }} animate={{ height: `${Math.max(4, u)}%` }}
                          transition={{ delay: i * 0.004, duration: 0.3, ease: [0.22, 1, 0.36, 1] }} style={{ minHeight: 2 }} />
                      ))}
                    </div>
                    <div className="flex items-center justify-between">
                      <p className="text-[10px] text-text-muted">CLIP ViT-L/14 &middot; {case_.clipPreview?.length ?? 96} bins from 768-D</p>
                    </div>
                  </div>
                ) : null}
              </motion.div>
            )}

            {/* ── 06: CSLS gallery search ── */}
            {sub === 'gallery_search' && (
              <motion.div key="gallery" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-5">
                <SectionHeader title="CSLS gallery search" provenance={stepProv} />

                <div className="flex flex-col gap-5 sm:flex-row sm:items-end">
                  <div className="space-y-3">
                    <div className="flex items-baseline gap-3">
                      <span className="font-mono text-[56px] font-bold tabular-nums leading-none text-text-primary">{galCount.toLocaleString()}</span>
                      <span className="text-[14px] text-text-muted">/ 10,000 embeddings scored</span>
                    </div>

                    <div className="h-2.5 overflow-hidden rounded-full bg-surface-active">
                      <motion.div className="h-full rounded-full bg-accent/45"
                        animate={{ width: `${(galCount / 10000) * 100}%` }} transition={{ duration: 0.1 }} />
                    </div>
                  </div>

                  {/* Mini gallery mosaic — brighter */}
                  <div className="flex shrink-0 items-end">
                    <svg viewBox="0 0 120 80" className="h-24 w-[120px]">
                      {Array.from({ length: 100 }).map((_, i) => {
                        const col = i % 10;
                        const row = Math.floor(i / 10);
                        const scanned = (col + row * 10) / 100 <= galCount / 10000;
                        return (
                          <rect
                            key={i}
                            x={12 + col * 10}
                            y={4 + row * 7.5}
                            width={8}
                            height={5.5}
                            rx={1}
                            fill={scanned ? 'rgb(77,124,255)' : 'rgb(55,55,72)'}
                            fillOpacity={scanned ? 0.55 : 0.35}
                          />
                        );
                      })}
                    </svg>
                  </div>
                </div>

                <div className="flex items-center gap-4 rounded-lg bg-surface-raised px-4 py-3">
                  <span className="rounded bg-surface-elevated px-2 py-1 font-mono text-[11px] text-text-secondary">Query</span>
                  <span className="text-border-emphasis">&rarr;</span>
                  <span className="rounded bg-surface-elevated px-2 py-1 font-mono text-[11px] text-text-secondary">CSLS scan</span>
                  <span className="text-border-emphasis">&rarr;</span>
                  <span className="rounded bg-surface-elevated px-2 py-1 font-mono text-[11px] text-text-secondary">Ranked</span>
                </div>

                <Explanation>CSLS corrects for hubness — certain gallery points tend to dominate nearest-neighbor queries. The query scans all 10,000 images to produce a ranked list.</Explanation>
              </motion.div>
            )}

            {/* ── 07: Top-K / Done ── */}
            {(sub === 'results' || sub === 'done') && (
              <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="space-y-6">
                <SectionHeader title="Retrieval complete" provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Cached retrieval ranking' }} />

                {/* ── HERO RANK VERDICT ── */}
                <motion.div
                  className={`rounded-xl p-7 ${
                    case_.metrics.rank === 1
                      ? 'border border-accent/25 bg-accent/[0.05]'
                      : case_.metrics.rank != null && case_.metrics.rank <= 5
                        ? 'border border-status-warning/20 bg-status-warning/[0.04]'
                        : 'border border-border-subtle bg-surface-raised'
                  }`}
                  initial={{ scale: 0.97, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 24 }}
                >
                  <div className="flex items-start gap-6">
                    {topK[0]?.image && (
                      <div className="shrink-0 overflow-hidden rounded-xl border-2 border-border-subtle">
                        <SafeImg src={topK[0].image} alt="Top-1 retrieval" className="h-36 w-36 object-cover" />
                      </div>
                    )}
                    <div className="flex-1 space-y-3">
                      <div className="flex items-baseline gap-4">
                        <span className={`font-mono text-[64px] font-bold tabular-nums leading-none ${
                          case_.metrics.rank === 1 ? 'text-accent' : case_.metrics.rank != null && case_.metrics.rank <= 5 ? 'text-status-warning' : 'text-text-primary'
                        }`}>
                          #{case_.metrics.rank ?? '?'}
                        </span>
                        <div>
                          <p className={`text-2xl font-semibold ${
                            case_.metrics.rank === 1 ? 'text-accent' : case_.metrics.rank != null && case_.metrics.rank <= 5 ? 'text-status-warning' : 'text-text-secondary'
                          }`}>
                            {case_.metrics.rank === 1 ? 'Exact match' : case_.metrics.rank != null && case_.metrics.rank <= 5 ? 'Near match' : 'Candidate identified'}
                          </p>
                          <p className="text-sm text-text-secondary">10,000 gallery images via CSLS ranking</p>
                        </div>
                      </div>
                      {case_.metrics.r1Correct && (
                        <span className="inline-flex items-center gap-1.5 rounded bg-accent/10 px-2.5 py-1 text-xs font-semibold text-accent">
                          <svg className="h-3.5 w-3.5" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
                          Recall@1 correct
                        </span>
                      )}
                    </div>
                  </div>
                </motion.div>

                {/* Stats grid */}
                <div className="grid gap-3 sm:grid-cols-4">
                  {([
                    ['Gallery', '10,000'],
                    ['Method', 'CSLS'],
                    ['Top-1 CSLS', topK[0]?.csls != null ? topK[0].csls.toFixed(3) : '—'],
                    ['Candidates', String(topK.length)],
                  ] as [string, string][]).map(([label, val]) => (
                    <div key={label} className="rounded-lg bg-surface-raised px-4 py-3">
                      <p className="text-[11px] font-medium text-text-secondary">{label}</p>
                      <p className="mt-0.5 font-mono text-xl font-semibold text-text-primary">{val}</p>
                    </div>
                  ))}
                </div>

                {/* κ + δ strip */}
                {kappaStr && (
                  <div className="flex items-center gap-3 rounded-lg bg-surface-raised px-4 py-2.5">
                    <span className="text-[12px] font-semibold text-text-secondary">Uncertainty</span>
                    <span className="font-mono text-[14px] font-semibold text-accent">κ = {kappaStr}</span>
                    <span className="text-border-emphasis">&middot;</span>
                    <span className="font-mono text-[14px] font-semibold text-status-warning">δ = {isLiveInference ? 'N/A' : case_.uncertainty.delta.toFixed(3)}</span>
                    <ProvenanceBadge provenance={DERIVED_PROV} />
                  </div>
                )}

                {/* Pipeline trace */}
                <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
                  {['fMRI betas', 'z-score', 'ROI mask', isMlpEncoder ? 'MLP encoder' : 'ROI transformer', 'vMF projection', 'CSLS search', 'Top-K'].map((step, i, arr) => (
                    <span key={step} className="flex items-center gap-1.5">
                      <span className="rounded bg-surface-raised px-1.5 py-0.5 font-medium text-text-secondary">{step}</span>
                      {i < arr.length - 1 && (
                        <svg className="h-2.5 w-2.5 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" /></svg>
                      )}
                    </span>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Top-K Filmstrip ── */}
      <motion.section
        className={`overflow-hidden rounded-xl border transition-colors duration-300 ${
          sub === 'done' || getStatus('results') === 'done'
            ? 'border-border-emphasis bg-surface-elevated'
            : getStatus('results') === 'active'
              ? 'border-accent/15 bg-surface-elevated'
              : 'border-border-subtle bg-surface-raised'
        } p-5`}
        animate={{ opacity: getStatus('results') === 'pending' ? 0.3 : 1 }}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className={`text-[12px] font-semibold uppercase tracking-[0.12em] ${
            getStatus('results') !== 'pending' ? 'text-text-secondary' : 'text-text-muted'
          }`}>
            Top-{topK.length} retrieved candidates
          </h3>
          {getStatus('results') === 'done' && (
            <ProvenanceBadge provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Precomputed gallery ranking' }} />
          )}
        </div>

        {/* ── PREMIUM AWAITING STATE (ghost cards with visible labels) ── */}
        {!visRanks.length && getStatus('results') !== 'done' && (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {[1, 2, 3, 4, 5].map((rank) => (
                <div key={rank} className="overflow-hidden rounded-lg border border-border-subtle bg-surface-base">
                  <div className="relative aspect-square w-full">
                    <div className="absolute inset-0 bg-surface-raised shimmer-bg" />
                    <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-surface-base/90 to-transparent pt-5 pb-2 px-2">
                      <span className="rounded bg-surface-overlay/50 px-2 py-0.5 text-[10px] font-bold text-text-muted">#{rank}</span>
                    </div>
                  </div>
                  <div className="px-3 py-2.5 space-y-1.5">
                    <div className="h-2.5 w-10 rounded bg-surface-active shimmer-bg" />
                    <div className="h-2.5 w-16 rounded bg-surface-active shimmer-bg" />
                  </div>
                </div>
              ))}
            </div>
            <div className="flex flex-col items-center gap-2 text-center">
              <p className="text-sm font-medium text-text-secondary">Awaiting retrieval candidates</p>
              <p className="text-[12px] text-text-muted">
                Top-K hypotheses will appear after CSLS gallery search completes.
                <br />
                <span className="text-accent/70">Current step: {STEPS.find((s) => s.phase === sub)?.label ?? 'Processing...'}</span>
              </p>
            </div>
          </div>
        )}

        {/* ── CANDIDATE GRID ── */}
        {visRanks.length > 0 && (
          <div className={`grid gap-3 ${topK.length <= 3 ? 'grid-cols-1 sm:grid-cols-3' : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5'}`}>
            {topK.map((item) => {
              const vis = visRanks.includes(item.rank);
              return (
                <motion.div
                  key={item.rank}
                  className={`group overflow-hidden rounded-lg border bg-surface-base transition-all duration-200 ${
                    item.rank === 1 ? 'border-accent/30 ring-1 ring-accent/10' : 'border-border-subtle'
                  }`}
                  initial={{ opacity: 0, scale: 0.92 }}
                  animate={{ opacity: vis ? 1 : 0, scale: vis ? 1 : 0.92 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                >
                  <div className="relative aspect-square w-full overflow-hidden">
                    <SafeImg src={item.image} alt={`Rank ${item.rank}`} className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.04]" />
                    <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent pt-5 pb-2 px-2">
                      <div className="flex items-center justify-between">
                        <span className={`rounded px-2 py-0.5 text-[10px] font-bold text-white ${item.rank === 1 ? 'bg-accent' : 'bg-surface-overlay/60'}`}>#{item.rank}</span>
                        <LabelBadge label={item.label} />
                      </div>
                    </div>
                  </div>
                  <div className="px-3 py-2.5">
                    <p className="font-mono text-[11px] leading-relaxed text-text-muted">
                      {item.score <= 1.0 ? `cos ${item.score.toFixed(3)} · ` : ''}csls {item.csls.toFixed(3)}
                    </p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </motion.section>

      {/* Computation details (collapsed) */}
      <details className="group rounded-xl border border-border-subtle bg-surface-raised">
        <summary className="flex cursor-pointer items-center justify-between px-4 py-2.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted select-none">
          <span>Computation details</span>
          <span className="text-[8px] transition-transform group-open:rotate-180">&#9660;</span>
        </summary>
        <div className="space-y-1.5 border-t border-border-subtle px-4 py-3 text-[10px] text-text-muted">
          <p><span className="font-medium text-text-secondary">Mode:</span> {isLiveInference ? 'Live backend inference via /api/infer-stream' : 'Replay of real experiment outputs from cached JSON'}</p>
          <p><span className="font-medium text-text-secondary">Encoder:</span> {isLiveInference ? (isMlpEncoder ? 'MLP encoder (V62a)' : 'ROI Transformer') : 'Architecture from replay case'}</p>
          <p><span className="font-medium text-text-secondary">Retrieval:</span> {isLiveInference ? 'Live CSLS search over 10,000 gallery embeddings' : 'Cached ranking from experiment output'}</p>
        </div>
      </details>

      {/* Proceed CTA */}
      <AnimatePresence>
        {showProceed && (
          <motion.div className="sticky bottom-4 z-10 mt-3 flex justify-center"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}>
            <button type="button" onClick={onComplete}
              className="flex items-center gap-3 rounded-xl border border-accent/25 bg-accent/10 px-8 py-4 text-sm font-semibold text-accent transition hover:border-accent/40 hover:bg-accent/15">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent/15 text-xs">&rarr;</span>
              Proceed to reconstruction
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
