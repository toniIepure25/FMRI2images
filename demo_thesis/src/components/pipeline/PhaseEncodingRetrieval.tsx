import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase, RetrievedImage } from '@/types';
import { resolveNsdIdToTrial, streamInference, fetchInference, isBackendAvailable, type InferenceStepEvent } from '@/lib/api';
import { ProvenanceBadge } from './ProvenanceBadge';
import { REPLAY_PROV, LIVE_PROV, DERIVED_PROV, UNKNOWN_PROV, type Provenance } from '@/lib/provenance';
import { hasFmriPreview, hasClipPreview, fmriPreviewProvenance, clipPreviewProvenance } from '@/lib/pipelineNormalize';
import { HeroResultPanel } from '@/components/premium/HeroResultPanel';
import { RetrievalCandidatesStrip } from '@/components/premium/RetrievalCandidatesStrip';
import { ComputationCard } from './ComputationCard';

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
    { phase: 'roi_encode',    label: isMlp ? 'MLP encoder' : 'ROI transformer', detail: isMlp ? '15,724 → 8192 → 8192 → 4096 → 2048 → 768' : '17 ROI tokens → transformer encoder → [CLS]', duration: 1600 },
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

type StepStatus = 'pending' | 'active' | 'done';

/* ─────────────────────────── Main component ───────────────────────────── */

export function PhaseEncodingRetrieval({
  case_,
  liveMode = false,
  onComplete,
  modelMeta,
}: PhaseEncodingRetrievalProps) {
  const encoderType = modelMeta?.encoder_type;
  const isMlpEncoder = !encoderType || encoderType === 'mlp' || encoderType === 'unknown';
  const hiddenDims = modelMeta?.encoder_hidden ?? [8192, 8192, 4096, 2048];
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

  const [liveReconAvailable, setLiveReconAvailable] = useState(false);

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

  // Fetch full live inference response when live retrieval completes
  useEffect(() => {
    if (!isLiveInference || !isBackendAvailable()) return;
    let cancelled = false;
    async function fetchLiveData() {
      try {
        const trialIdx = await resolveNsdIdToTrial(case_.nsdId);
        if (cancelled || trialIdx == null) return;
        const data = await fetchInference(trialIdx);
        if (cancelled || !data) return;
        setLiveReconAvailable(data.reconstruction?.ok ?? false);
      } catch { /* silently ignore — live data is best-effort */ }
    }
    if (sub === 'done' && isLiveInference) {
      fetchLiveData();
    }
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sub, isLiveInference, case_.nsdId]);

  const ci = stepIdx(sub);
  const getStatus = (p: Subphase): StepStatus => {
    if (sub === 'done') return 'done';
    if (sub === p) return 'active';
    return ci > stepIdx(p) ? 'done' : 'pending';
  };

  const stepProv: Provenance = isLiveInference ? LIVE_PROV : REPLAY_PROV;
  const fmriStats = case_.fmriPreviewMeta?.stats;
  const hasReconAsset = !!(liveReconAvailable || case_.diffusionFinal || case_.reconstructionImage);
  const candidateStripReady = visRanks.length > 0 || getStatus('results') === 'active' || getStatus('results') === 'done';

  return (
    <div className="relative space-y-3">
      {/* ── Rail + Canvas — two panels with matching elevation read as one workbench ── */}
      <div className="flex flex-col gap-3 lg:flex-row">

        {/* Zone A: Execution rail (sticky, flatter than the canvas so the
            active card dominates and the rail reads as a timeline). */}
        <aside className="w-full shrink-0 lg:w-[252px]">
          <div className="premium-panel-flat sticky top-[88px] px-3 py-4">
            <div className="mb-3 flex items-center justify-between gap-3 px-1.5">
              <p className="premium-kicker">
                {isLiveInference ? 'Backend trace' : 'Replay trace'}
              </p>
              {sub !== 'done' ? (
                <button
                  type="button"
                  onClick={skip}
                  className="rounded text-[10px] font-medium text-text-muted/60 transition hover:text-text-secondary"
                  title="Skip to final result"
                >
                  Skip
                </button>
              ) : null}
            </div>
            <ol className="relative space-y-0.5">
              <div className="absolute left-[18px] top-3 bottom-3 w-px bg-border-subtle/60" aria-hidden />
              {STEPS.map((step) => {
                const s = getStatus(step.phase);
                return (
                  <motion.li
                    key={step.phase}
                    className={`relative flex items-start gap-3 rounded-lg px-2 py-2 transition-colors duration-200 ${
                      s === 'active' ? 'bg-accent/[0.05]' : ''
                    }`}
                    animate={{ opacity: s === 'pending' ? 0.62 : 1 }}
                  >
                    <div className="relative z-10 mt-[5px] flex h-[14px] w-[14px] shrink-0 items-center justify-center">
                      {s === 'done' ? (
                        <div className="h-1.5 w-1.5 rounded-full bg-accent/70" />
                      ) : s === 'active' ? (
                        <motion.div
                          className="h-2 w-2 rounded-full bg-accent"
                          animate={{ opacity: [1, 0.45, 1] }}
                          transition={{ duration: 1.2, repeat: Infinity }}
                        />
                      ) : (
                        <div className="h-1.5 w-1.5 rounded-full bg-border-emphasis/60" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <span
                        className={`block text-[12px] font-semibold leading-snug ${
                          s === 'active'
                            ? 'text-accent'
                            : s === 'done'
                            ? 'text-text-primary'
                            : 'text-text-muted'
                        }`}
                      >
                        {step.label}
                      </span>
                      <p
                        className={`mt-0.5 text-[11px] leading-snug ${
                          s === 'pending' ? 'text-text-muted/55' : 'text-text-secondary'
                        }`}
                      >
                        {step.detail}
                      </p>
                    </div>
                  </motion.li>
                );
              })}
            </ol>
            {sub !== 'done' && (
              <div className="mx-1.5 mt-3 h-[3px] overflow-hidden rounded-full bg-surface-active">
                <motion.div
                  className="h-full rounded-full bg-accent/55"
                  animate={{ width: `${Math.min(100, progress * 100)}%` }}
                  transition={{ duration: 0.08 }}
                />
              </div>
            )}
          </div>
        </aside>

        {/* Zone B: Evidence canvas */}
        <div className="premium-panel flex-1 px-5 py-5 sm:px-6 sm:py-6">
          <AnimatePresence mode="wait">

            {/* ── 01 / 02 / 03: fMRI signal + prep ── */}
            {(sub === 'load_betas' || sub === 'zscore' || sub === 'roi_mask') && (
              <motion.div key="fmri" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ComputationCard
                  step={sub === 'load_betas' ? 'Step 01 · fMRI' : sub === 'zscore' ? 'Step 02 · Preprocessing' : 'Step 03 · ROI'}
                  title={sub === 'load_betas' ? 'fMRI signal acquisition' : sub === 'zscore' ? 'Feature preprocessing' : 'ROI masking'}
                  subtitle={sub === 'load_betas' ? 'Trial-specific ROI beta vector loaded from cached experiment assets.' : sub === 'zscore' ? (isLiveInference ? 'Pre-extracted features used directly — z-score not applied.' : 'Per-session z-score normalization stabilizes feature scale before decoding.') : 'Retain nsdgeneral visual cortex voxels for the model input.'}
                  provenance={hasRealFmri ? fmriProv : UNKNOWN_PROV}
                >

                {hasRealFmri && fmriHeights ? (
                  <div className="space-y-5">
                    {/* ── Activation profile — thin bars on a baseline,
                           reads as a scientific signal trace, not toy bars ── */}
                    <div className="workbench-inset">
                      <div className="mb-3 flex items-baseline justify-between">
                        <p className="premium-kicker">ROI feature vector</p>
                        <span className="font-mono text-[11px] tabular-nums text-text-secondary">
                          {fmriStats?.n_voxels?.toLocaleString() ?? '~15,724'} voxels
                          <span className="text-text-muted"> · {case_.fmriPreview.length} bins</span>
                        </span>
                      </div>

                      <div className="relative h-[112px]">
                        {/* Subtle gridlines for scale reference */}
                        <div className="pointer-events-none absolute inset-0 flex flex-col justify-between" aria-hidden>
                          <div className="h-px bg-white/[0.04]" />
                          <div className="h-px bg-white/[0.04]" />
                          <div className="h-px bg-white/[0.04]" />
                          <div className="h-px bg-white/[0.06]" />
                        </div>
                        {/* Bars: 1.5px effective width, fixed gap, lower opacity */}
                        <div className="absolute inset-0 flex items-end gap-px">
                          {fmriHeights.map((h, i) => (
                            <motion.div
                              key={i}
                              className="min-w-0 flex-1 bg-accent/45"
                              initial={{ height: 0 }}
                              animate={{ height: `${Math.max(3, h)}%` }}
                              transition={{ delay: i * 0.0012, duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
                              style={{ minHeight: 2 }}
                            />
                          ))}
                        </div>
                      </div>
                      <p className="mt-2.5 text-[10.5px] text-text-muted">
                        ROI-masked beta vector · higher bars = stronger activation
                      </p>
                    </div>

                    {/* ── Scientific stat readout — hairline-separated
                           columns, tabular numerals, no card chrome ── */}
                    {fmriStats && (() => {
                      const statDefs: { label: string; key: keyof typeof fmriStats; fmt: 'int' | 'dec' }[] = [
                        { label: 'Voxels', key: 'n_voxels', fmt: 'int' },
                        { label: 'Mean', key: 'mean', fmt: 'dec' },
                        { label: 'Std', key: 'std', fmt: 'dec' },
                        { label: 'Min', key: 'min', fmt: 'dec' },
                        { label: 'Max', key: 'max', fmt: 'dec' },
                        { label: '|Mean|', key: 'abs_mean', fmt: 'dec' },
                      ];
                      // Drop Voxels from the row if unavailable to avoid the N/A dominating.
                      const visible = statDefs.filter((s) => !(s.key === 'n_voxels' && fmriStats[s.key] == null));
                      return (
                        <div className="grid grid-cols-3 divide-x divide-white/[0.05] overflow-hidden rounded-xl border border-white/[0.04] bg-white/[0.012] sm:grid-cols-6">
                          {visible.map(({ label, key, fmt }) => {
                            const val = fmriStats[key];
                            const display =
                              val == null
                                ? 'Unavailable'
                                : fmt === 'int' && typeof val === 'number'
                                ? val.toLocaleString()
                                : typeof val === 'number'
                                ? val.toFixed(1)
                                : String(val);
                            const isUnavailable = val == null;
                            return (
                              <div key={label} className="px-3.5 py-2.5">
                                <p className="premium-kicker">{label}</p>
                                <p
                                  className={`mt-1 font-mono text-[14px] font-semibold tabular-nums leading-none ${
                                    isUnavailable ? 'text-text-muted/80' : 'text-text-primary'
                                  }`}
                                >
                                  {display}
                                </p>
                              </div>
                            );
                          })}
                        </div>
                      );
                    })()}

                    {/* ── Metadata rail — single line, kicker labels ── */}
                    <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1.5 text-[11.5px]">
                      {(['Subject', 'Session', 'nsdId', 'ROI'] as const).map((k) => {
                        const vals: Record<string, string> = {
                          Subject: case_.subject,
                          Session: String(case_.session),
                          nsdId: String(case_.nsdId),
                          ROI: 'nsdgeneral',
                        };
                        return (
                          <span key={k} className="inline-flex items-baseline gap-1.5">
                            <span className="premium-kicker">{k}</span>
                            <span className="font-mono text-text-primary tabular-nums">{vals[k]}</span>
                          </span>
                        );
                      })}
                    </div>

                    {/* Preprocessing visual */}
                    {sub === 'zscore' && (
                      <div className="workbench-inset">
                        <p className="premium-kicker mb-3">z-score normalization</p>
                        <div className="flex flex-wrap items-center gap-3 text-[11px]">
                          <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] px-3.5 py-2.5">
                            <div className="font-mono text-text-secondary">Raw features</div>
                            <div className="text-[9.5px] text-text-muted">variable scale</div>
                          </div>
                          <svg className="h-3 w-3 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                          </svg>
                          <div className="rounded-lg border border-accent/20 bg-accent/[0.06] px-3.5 py-2.5">
                            <div className="font-mono text-accent">(x − μ) / σ</div>
                            <div className="text-[9.5px] text-text-muted">per-session</div>
                          </div>
                          <svg className="h-3 w-3 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.6}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                          </svg>
                          <div className="rounded-lg border border-white/[0.05] bg-white/[0.02] px-3.5 py-2.5">
                            <div className="font-mono text-text-secondary">Normalized</div>
                            <div className="text-[9.5px] text-text-muted">μ=0, σ=1</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* ── ROI extraction diagram — refined three-stage flow,
                           emphasis carried by the final voxel count ── */}
                    {sub === 'roi_mask' && (
                      <div className="workbench-inset">
                        <div className="mb-4 flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <p className="premium-kicker">Visual cortex ROI extraction</p>
                            <p className="mt-1.5 text-[12px] leading-relaxed text-text-muted">
                              The full-volume beta map is reduced to the nsdgeneral visual cortex mask before the model sees it.
                            </p>
                          </div>
                          <span className="sci-chip sci-chip-mono sci-chip-muted">atlas: nsdgeneral</span>
                        </div>

                        <div className="grid items-stretch gap-3 md:grid-cols-[1fr_auto_1fr_auto_1.05fr]">
                          {/* Stage 1: full volume */}
                          <div className="flex flex-col justify-between rounded-xl border border-white/[0.045] bg-white/[0.018] px-4 py-3.5">
                            <p className="premium-kicker">Full beta volume</p>
                            <p className="mt-2 font-mono text-[20px] font-semibold tabular-nums leading-none text-text-secondary">
                              ~150k
                            </p>
                            <p className="mt-1.5 text-[10.5px] text-text-muted">source voxel field</p>
                          </div>
                          {/* Arrow */}
                          <div className="hidden items-center justify-center text-border-emphasis md:flex">
                            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                            </svg>
                          </div>
                          {/* Stage 2: ROI mask — refined label, no giant "visual" word */}
                          <div className="flex flex-col justify-between rounded-xl border border-accent/18 bg-accent/[0.05] px-4 py-3.5">
                            <p className="premium-kicker" style={{ color: 'rgb(123 156 255)' }}>nsdgeneral ROI</p>
                            <p className="mt-2 font-mono text-[13px] font-semibold leading-tight text-text-primary">
                              visual cortex mask
                            </p>
                            <p className="mt-1.5 text-[10.5px] text-text-muted">stimulus-responsive voxels</p>
                          </div>
                          {/* Arrow */}
                          <div className="hidden items-center justify-center text-border-emphasis md:flex">
                            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                            </svg>
                          </div>
                          {/* Stage 3: model input — emphasized numeric */}
                          <div className="flex flex-col justify-between rounded-xl border border-white/[0.06] bg-white/[0.03] px-4 py-3.5">
                            <p className="premium-kicker">Model input</p>
                            <p className="mt-2 font-mono text-[24px] font-semibold tabular-nums leading-none text-text-primary">
                              {fmriStats?.n_voxels ? fmriStats.n_voxels.toLocaleString() : '15,724'}
                            </p>
                            <p className="mt-1.5 text-[10.5px] text-text-muted">retained voxels → V62a</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="workbench-inset flex h-40 flex-col items-center justify-center gap-3 !p-5">
                    <svg className="h-9 w-9 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
                    </svg>
                    <p className="text-[13px] font-medium text-text-muted">fMRI preview not exported for this trial</p>
                    <p className="text-[11px] text-text-muted">The beta vector for NSD trial {case_.nsdId} was not included in the replay assets.</p>
                  </div>
                )}
                </ComputationCard>
              </motion.div>
            )}

            {/* ── 04: MLP ENCODER ── */}
            {sub === 'roi_encode' && (
              <motion.div key="roi" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ComputationCard
                  step={isMlpEncoder ? 'Step 04 · MLP encoder' : 'Step 04 · ROI transformer'}
                  title={isMlpEncoder ? 'MLP encoder' : 'ROI transformer'}
                  subtitle={isMlpEncoder ? 'Residual projection from ROI voxels into the CLIP latent direction.' : 'Transformer processes brain-topology-aware tokens through 6 layers.'}
                  provenance={stepProv}
                >

                {isMlpEncoder ? (
                  (() => {
                    const outDim = modelMeta?.embedding_dim ?? 768;
                    const inDim = 15724;
                    // ── Architecture figure rendered as a single SVG so the
                    //    layers, funnel trapezoids, and skip arc share one
                    //    coordinate system and align perfectly. Cube-root
                    //    dimension scaling exaggerates the visual compression
                    //    so 15,724 → 768 reads as a real funnel, not equal
                    //    rectangles. ──
                    const VB_W = 720;
                    const VB_H = 220;
                    const BASELINE_Y = 130; // vertical center for layer blocks
                    const maxScale = Math.cbrt(inDim);
                    const sc = (d: number) => Math.cbrt(d) / maxScale; // 0..1
                    const halfH = (d: number) => sc(d) * 56;            // px

                    // ── Layout — fixed x positions in viewBox coords. ──
                    type Stage = {
                      key: string;
                      cx: number;
                      w: number;
                      dim: number;
                      label: string;
                      kind: 'input' | 'layer' | 'output';
                      idx?: number;
                    };
                    const stages: Stage[] = [
                      { key: 'roi', cx: 56,  w: 64, dim: inDim,           label: 'ROI vector',     kind: 'input' },
                      { key: 'L1',  cx: 178, w: 52, dim: hiddenDims[0],   label: 'L1',             kind: 'layer', idx: 0 },
                      { key: 'L2',  cx: 286, w: 52, dim: hiddenDims[1],   label: 'L2',             kind: 'layer', idx: 1 },
                      { key: 'L3',  cx: 394, w: 52, dim: hiddenDims[2],   label: 'L3',             kind: 'layer', idx: 2 },
                      { key: 'L4',  cx: 502, w: 52, dim: hiddenDims[3],   label: 'L4',             kind: 'layer', idx: 3 },
                      { key: 'out', cx: 640, w: 76, dim: outDim,          label: 'CLIP / vMF',     kind: 'output' },
                    ];
                    const Y_TOP    = (d: number) => BASELINE_Y - halfH(d);
                    const Y_BOT    = (d: number) => BASELINE_Y + halfH(d);
                    const RIGHT    = (s: Stage) => s.cx + s.w / 2;
                    const LEFT     = (s: Stage) => s.cx - s.w / 2;

                    // Funnel trapezoids between adjacent stages — they
                    // physically *connect* the layers, so the compression
                    // reads as a continuous architecture, not 4 isolated
                    // towers.
                    const funnels = stages.slice(0, -1).map((s, i) => {
                      const next = stages[i + 1];
                      return { from: s, to: next, key: `${s.key}-${next.key}` };
                    });

                    return (
                      // ── Single architecture canvas — one SVG figure for
                      //    the entire encoder, sitting in a workbench inset.
                      //    No nested cards, no decorative chrome. ──
                      <div className="workbench-inset !p-6">
                        <div className="mb-4 flex items-baseline justify-between">
                          <p className="premium-kicker">Residual MLP encoder</p>
                          <p className="text-[9.5px] uppercase tracking-[0.18em] text-text-muted/70">
                            GELU · LayerNorm · residual skip
                          </p>
                        </div>

                        <svg
                          viewBox={`0 0 ${VB_W} ${VB_H}`}
                          preserveAspectRatio="xMidYMid meet"
                          className="block h-auto w-full"
                          role="img"
                          aria-label="MLP encoder architecture: 15,724 ROI voxels into a residual MLP with 8192, 8192, 4096, 2048 hidden units, projecting to a 768-dimensional CLIP direction."
                        >
                          <defs>
                            {/* Subtle vertical gradient inside layer blocks
                                so they read with a quiet sense of volume,
                                not flat fills. */}
                            <linearGradient id="mlp-layer-fill" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%"  stopColor="rgb(123,156,255)" stopOpacity="0.18" />
                              <stop offset="100%" stopColor="rgb(123,156,255)" stopOpacity="0.07" />
                            </linearGradient>
                            <linearGradient id="mlp-input-fill" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%"  stopColor="rgb(170,180,200)" stopOpacity="0.10" />
                              <stop offset="100%" stopColor="rgb(170,180,200)" stopOpacity="0.04" />
                            </linearGradient>
                            <linearGradient id="mlp-output-fill" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%"  stopColor="rgb(123,156,255)" stopOpacity="0.28" />
                              <stop offset="100%" stopColor="rgb(123,156,255)" stopOpacity="0.12" />
                            </linearGradient>
                            <linearGradient id="mlp-funnel-fill" x1="0" y1="0" x2="1" y2="0">
                              <stop offset="0%"  stopColor="rgb(123,156,255)" stopOpacity="0.06" />
                              <stop offset="100%" stopColor="rgb(123,156,255)" stopOpacity="0.06" />
                            </linearGradient>
                          </defs>

                          {/* ─── (1) Funnels between adjacent stages — drawn
                                  first so layer blocks sit cleanly on top. ─ */}
                          {funnels.map(({ from, to, key }) => {
                            const x1 = RIGHT(from);
                            const x2 = LEFT(to);
                            const yt1 = Y_TOP(from.dim);
                            const yb1 = Y_BOT(from.dim);
                            const yt2 = Y_TOP(to.dim);
                            const yb2 = Y_BOT(to.dim);
                            return (
                              <g key={key}>
                                <polygon
                                  points={`${x1},${yt1} ${x2},${yt2} ${x2},${yb2} ${x1},${yb1}`}
                                  fill="url(#mlp-funnel-fill)"
                                />
                                {/* Hairline edges along the top and bottom
                                    of the funnel — gives the compression a
                                    crisp silhouette. */}
                                <line x1={x1} y1={yt1} x2={x2} y2={yt2} stroke="rgb(123,156,255)" strokeOpacity="0.32" strokeWidth="0.7" />
                                <line x1={x1} y1={yb1} x2={x2} y2={yb2} stroke="rgb(123,156,255)" strokeOpacity="0.32" strokeWidth="0.7" />
                              </g>
                            );
                          })}

                          {/* ─── (2) Residual skip arc — L1 → L4. One clean
                                  curve, anchored to the layer tops, with a
                                  quiet "residual skip" label above it. ─── */}
                          {(() => {
                            const a = stages[1]; // L1
                            const b = stages[4]; // L4
                            const ax = a.cx;
                            const ay = Y_TOP(a.dim) - 4;
                            const bx = b.cx;
                            const by = Y_TOP(b.dim) - 4;
                            const apexY = Math.min(ay, by) - 36;
                            const midX = (ax + bx) / 2;
                            return (
                              <g>
                                <path
                                  d={`M ${ax} ${ay} Q ${midX} ${apexY} ${bx} ${by}`}
                                  stroke="rgb(123,156,255)"
                                  strokeOpacity="0.65"
                                  strokeWidth="1"
                                  fill="none"
                                  strokeLinecap="round"
                                />
                                {/* Start marker — open ring */}
                                <circle cx={ax} cy={ay} r="2" fill="none" stroke="rgb(123,156,255)" strokeOpacity="0.7" strokeWidth="0.9" />
                                {/* Arrival marker — filled dot (confluence) */}
                                <circle cx={bx} cy={by} r="2.4" fill="rgb(123,156,255)" fillOpacity="0.85" />
                                {/* Quiet label */}
                                <text
                                  x={midX}
                                  y={apexY - 4}
                                  textAnchor="middle"
                                  fill="rgb(123,156,255)"
                                  fillOpacity="0.7"
                                  style={{ fontFamily: 'Inter, system-ui, sans-serif', fontSize: 9.5, letterSpacing: '0.16em', textTransform: 'uppercase' }}
                                >
                                  residual skip
                                </text>
                              </g>
                            );
                          })()}

                          {/* ─── (3) Stage blocks ─── */}
                          {stages.map((s) => {
                            const x = LEFT(s);
                            const yT = Y_TOP(s.dim);
                            const yB = Y_BOT(s.dim);
                            const h = yB - yT;

                            if (s.kind === 'input') {
                              // ROI input — clean bar histogram inside the
                              // input rectangle. Consistent bar widths, a
                              // sparse accent pattern for "active voxels".
                              const barCount = 16;
                              const barW = (s.w - 8) / barCount;
                              return (
                                <g key={s.key}>
                                  <rect
                                    x={x}
                                    y={yT}
                                    width={s.w}
                                    height={h}
                                    rx={4}
                                    fill="url(#mlp-input-fill)"
                                    stroke="rgb(255,255,255)"
                                    strokeOpacity="0.10"
                                    strokeWidth="0.7"
                                  />
                                  {/* Hair-thin baseline */}
                                  <line
                                    x1={x + 4}
                                    y1={yB - 6}
                                    x2={x + s.w - 4}
                                    y2={yB - 6}
                                    stroke="rgb(255,255,255)"
                                    strokeOpacity="0.18"
                                    strokeWidth="0.6"
                                  />
                                  {/* Bars — deterministic but smooth heights;
                                      every 5th bar is "active" and tinted
                                      accent so it reads as a sparse fMRI
                                      activation. */}
                                  {Array.from({ length: barCount }).map((_, i) => {
                                    const bh = 6 + ((i * 41 + 11) % (h - 18));
                                    const bx = x + 4 + i * barW;
                                    const isActive = i % 5 === 2;
                                    return (
                                      <rect
                                        key={i}
                                        x={bx}
                                        y={yB - 6 - bh}
                                        width={Math.max(1, barW - 1)}
                                        height={bh}
                                        fill={isActive ? 'rgb(123,156,255)' : 'rgb(220,225,235)'}
                                        fillOpacity={isActive ? 0.72 : 0.32}
                                        rx={0.6}
                                      />
                                    );
                                  })}
                                </g>
                              );
                            }

                            if (s.kind === 'output') {
                              // Output token — refined unit-direction marker
                              // (filled hemisphere of a unit circle with a
                              // radial vector ending in a dot) sitting above
                              // a compact latent-channel strip. Together
                              // they read as "768-D direction · unit-norm".
                              const cx = s.cx;
                              const cy = (yT + yB) / 2 - 8;
                              const R  = Math.min(20, h / 3.2);
                              const stripY = cy + R + 8;
                              const stripH = Math.max(6, yB - stripY - 4);
                              return (
                                <g key={s.key}>
                                  <rect
                                    x={x}
                                    y={yT}
                                    width={s.w}
                                    height={h}
                                    rx={6}
                                    fill="url(#mlp-output-fill)"
                                    stroke="rgb(123,156,255)"
                                    strokeOpacity="0.40"
                                    strokeWidth="0.8"
                                  />
                                  {/* Unit sphere outline */}
                                  <circle cx={cx} cy={cy} r={R} fill="none" stroke="rgb(123,156,255)" strokeOpacity="0.45" strokeWidth="0.8" />
                                  {/* Horizon line */}
                                  <line x1={cx - R} y1={cy} x2={cx + R} y2={cy} stroke="rgb(123,156,255)" strokeOpacity="0.18" strokeWidth="0.5" />
                                  {/* μ vector ending in a bright dot */}
                                  <line x1={cx} y1={cy} x2={cx + R * 0.72} y2={cy - R * 0.66} stroke="rgb(123,156,255)" strokeWidth="1.4" strokeLinecap="round" />
                                  <circle cx={cx + R * 0.72} cy={cy - R * 0.66} r="2.2" fill="rgb(123,156,255)" />
                                  {/* Origin marker */}
                                  <circle cx={cx} cy={cy} r="1.2" fill="rgb(220,225,235)" fillOpacity="0.7" />
                                  {/* Compact latent strip — 8 cells */}
                                  {Array.from({ length: 8 }).map((_, i) => {
                                    const cellW = (s.w - 10) / 8;
                                    const cx2 = x + 5 + i * cellW;
                                    const intensity = 0.22 + ((i * 47 + 13) % 60) / 110;
                                    return (
                                      <rect
                                        key={i}
                                        x={cx2}
                                        y={stripY}
                                        width={cellW - 1.5}
                                        height={stripH}
                                        rx={0.8}
                                        fill="rgb(123,156,255)"
                                        fillOpacity={intensity}
                                      />
                                    );
                                  })}
                                </g>
                              );
                            }

                            // Hidden layer block. Internal vertical tick
                            // lines suggest a column of hidden units; the
                            // count scales lightly with the layer's
                            // dimensionality.
                            const units = 5 + Math.round(sc(s.dim) * 5); // 5..10
                            return (
                              <g key={s.key}>
                                <rect
                                  x={x}
                                  y={yT}
                                  width={s.w}
                                  height={h}
                                  rx={5}
                                  fill="url(#mlp-layer-fill)"
                                  stroke="rgb(123,156,255)"
                                  strokeOpacity="0.36"
                                  strokeWidth="0.8"
                                />
                                {/* Hair-thin top highlight — implies a head row */}
                                <line x1={x + 2} y1={yT + 1} x2={x + s.w - 2} y2={yT + 1} stroke="rgb(255,255,255)" strokeOpacity="0.12" strokeWidth="0.7" />
                                {/* Vertical unit ticks */}
                                {Array.from({ length: units }).map((_, i) => {
                                  const ux = x + 4 + ((s.w - 8) * (i + 0.5)) / units;
                                  return (
                                    <line
                                      key={i}
                                      x1={ux}
                                      y1={yT + 4}
                                      x2={ux}
                                      y2={yB - 4}
                                      stroke="rgb(123,156,255)"
                                      strokeOpacity={0.16 + ((i * 11) % 9) / 80}
                                      strokeWidth="0.55"
                                    />
                                  );
                                })}
                              </g>
                            );
                          })}

                          {/* ─── (4) Labels — section labels above, dim
                                  numbers and stage labels below the rail. ─ */}
                          {stages.map((s) => {
                            const sectionLabel =
                              s.kind === 'input' ? 'ROI vector'
                              : s.kind === 'output' ? 'CLIP / vMF'
                              : null;
                            return (
                              <g key={`label-${s.key}`}>
                                {/* Section label above */}
                                {sectionLabel ? (
                                  <text
                                    x={s.cx}
                                    y={Y_TOP(s.dim) - 12}
                                    textAnchor="middle"
                                    fill="rgb(156,163,175)"
                                    style={{ fontFamily: 'Inter, system-ui, sans-serif', fontSize: 9, fontWeight: 600, letterSpacing: '0.18em', textTransform: 'uppercase' }}
                                  >
                                    {sectionLabel}
                                  </text>
                                ) : null}

                                {/* Stage caption below (L1/L2/L3/L4) */}
                                {s.kind === 'layer' ? (
                                  <text
                                    x={s.cx}
                                    y={Y_BOT(s.dim) + 14}
                                    textAnchor="middle"
                                    fill="rgb(156,163,175)"
                                    style={{ fontFamily: 'Inter, system-ui, sans-serif', fontSize: 9, letterSpacing: '0.14em', textTransform: 'uppercase' }}
                                  >
                                    {s.label}
                                  </text>
                                ) : null}

                                {/* Numeric dim — primary readout */}
                                <text
                                  x={s.cx}
                                  y={s.kind === 'layer' ? Y_BOT(s.dim) + 28 : Y_BOT(s.dim) + 16}
                                  textAnchor="middle"
                                  fill={s.kind === 'output' ? 'rgb(123,156,255)' : 'rgb(229,231,235)'}
                                  style={{
                                    fontFamily: 'JetBrains Mono, ui-monospace, monospace',
                                    fontSize: s.kind === 'layer' ? 12 : 14,
                                    fontWeight: 600,
                                    fontVariantNumeric: 'tabular-nums',
                                  }}
                                >
                                  {s.dim.toLocaleString()}{s.kind === 'output' ? '-D' : ''}
                                </text>

                                {/* Sub-caption */}
                                {s.kind === 'input' ? (
                                  <text
                                    x={s.cx}
                                    y={Y_BOT(s.dim) + 30}
                                    textAnchor="middle"
                                    fill="rgb(125,130,140)"
                                    style={{ fontFamily: 'Inter, system-ui, sans-serif', fontSize: 9 }}
                                  >
                                    voxels · nsdgeneral
                                  </text>
                                ) : null}
                                {s.kind === 'output' ? (
                                  <text
                                    x={s.cx}
                                    y={Y_BOT(s.dim) + 30}
                                    textAnchor="middle"
                                    fill="rgb(125,130,140)"
                                    style={{ fontFamily: 'Inter, system-ui, sans-serif', fontSize: 9 }}
                                  >
                                    CLIP direction · unit-norm
                                  </text>
                                ) : null}
                              </g>
                            );
                          })}

                          {/* ─── (5) Inlet / outlet arrows — sit on the
                                  baseline so the eye traces ROI → encoder
                                  → CLIP. ─── */}
                          <g stroke="rgb(125,130,140)" strokeWidth="0.9" fill="none" strokeLinecap="round" strokeLinejoin="round" opacity="0.7">
                            <line x1={RIGHT(stages[0]) + 4} y1={BASELINE_Y} x2={LEFT(stages[1]) - 4} y2={BASELINE_Y} />
                            <path d={`M ${LEFT(stages[1]) - 4} ${BASELINE_Y} l -4 -3 m 4 3 l -4 3`} />
                            <line x1={RIGHT(stages[4]) + 4} y1={BASELINE_Y} x2={LEFT(stages[5]) - 4} y2={BASELINE_Y} />
                            <path d={`M ${LEFT(stages[5]) - 4} ${BASELINE_Y} l -4 -3 m 4 3 l -4 3`} />
                          </g>
                        </svg>

                        {/* ── Single formula line, integrated under the
                            figure. Only the final term is accented. ── */}
                        <p className="mt-5 border-t border-white/[0.04] pt-3.5 font-mono text-[11.5px] leading-relaxed text-text-secondary">
                          <span className="text-text-primary">{inDim.toLocaleString()} ROI voxels</span>
                          <span className="mx-1.5 text-border-emphasis">→</span>
                          {hiddenDims.map((d, i) => (
                            <span key={i}>
                              <span className="text-text-secondary">{d.toLocaleString()}</span>
                              {i < hiddenDims.length - 1 ? (
                                <span className="mx-1.5 text-border-emphasis">→</span>
                              ) : null}
                            </span>
                          ))}
                          <span className="mx-1.5 text-border-emphasis">→</span>
                          <span className="font-semibold text-accent">{outDim}-D CLIP direction</span>
                        </p>
                      </div>
                    );
                  })()
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
                        <div className="rounded-lg border border-accent/15 bg-accent/[0.04] px-3 py-2"><p className="text-[10px] font-semibold text-text-secondary">Transformer ×6</p><p className="text-[9px] text-text-muted">d=768, 12 heads</p></div>
                      <svg className="h-3 w-5 shrink-0 text-border-emphasis" viewBox="0 0 20 12" fill="none"><path d="M0 6h16m0 0l-4-4m4 4l-4 4" stroke="currentColor" strokeWidth="1.5" /></svg>
                      <div className="rounded-lg border border-accent/20 bg-accent/[0.05] px-3 py-2"><p className="text-[10px] font-semibold text-accent">[CLS] 768-D</p></div>
                    </div>
                    <p className="text-[11px] text-text-muted">ROI transformer processes brain-topology-aware tokens through 6 transformer layers.</p>
                  </div>
                )}
                </ComputationCard>
              </motion.div>
            )}

            {/* ── 05: vMF projection ── */}
            {sub === 'vmf_decode' && (
              <motion.div key="vmf" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ComputationCard
                  step="Step 05 · vMF projection"
                  title="vMF projection"
                  subtitle="The encoder output is mapped to a direction μ on the CLIP unit hypersphere. Higher κ means sharper directional confidence."
                  provenance={DERIVED_PROV}
                >

                {/* ── Directional embedding canvas: hypersphere on the left,
                       evidence readout on the right, with the cached CLIP
                       latent profile and honest provenance annotation folded
                       into the same inset. One scientific figure, no nested
                       cards. ── */}
                <div className="workbench-inset !p-0 overflow-hidden">
                  <div className="grid gap-0 sm:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)] sm:divide-x sm:divide-white/[0.05]">
                    {/* Visualization canvas */}
                    <div className="relative px-5 py-6 sm:py-7">
                      <p className="premium-kicker mb-3">Directional embedding</p>
                      <div className="flex items-center justify-center">
                        <svg viewBox="0 0 220 200" className="h-48 w-full max-w-[260px] sm:h-52" aria-label="vMF directional embedding visualization">
                          <defs>
                            {/* Subtle sphere fill — no glow */}
                            <radialGradient id="vmfSphere" cx="38%" cy="34%">
                              <stop offset="0%" stopColor="rgb(255,255,255)" stopOpacity="0.05" />
                              <stop offset="60%" stopColor="rgb(255,255,255)" stopOpacity="0.012" />
                              <stop offset="100%" stopColor="rgb(0,0,0)" stopOpacity="0" />
                            </radialGradient>
                            {/* Concentration cone fill — accent, low alpha */}
                            <linearGradient id="vmfCone" x1="0" y1="0" x2="1" y2="0">
                              <stop offset="0%" stopColor="rgb(77,124,255)" stopOpacity="0" />
                              <stop offset="100%" stopColor="rgb(77,124,255)" stopOpacity="0.18" />
                            </linearGradient>
                          </defs>

                          {/* Sphere */}
                          <circle cx="110" cy="100" r="78" fill="url(#vmfSphere)" stroke="rgb(255,255,255)" strokeWidth="1" strokeOpacity="0.14" />
                          {/* Latitude rings */}
                          <ellipse cx="110" cy="100" rx="78" ry="22" fill="none" stroke="rgb(255,255,255)" strokeWidth="0.5" strokeOpacity="0.06" />
                          <ellipse cx="110" cy="100" rx="78" ry="42" fill="none" stroke="rgb(255,255,255)" strokeWidth="0.5" strokeOpacity="0.05" />
                          <ellipse cx="110" cy="100" rx="78" ry="62" fill="none" stroke="rgb(255,255,255)" strokeWidth="0.5" strokeOpacity="0.04" />
                          {/* Longitude meridian */}
                          <ellipse cx="110" cy="100" rx="30" ry="78" fill="none" stroke="rgb(255,255,255)" strokeWidth="0.5" strokeOpacity="0.05" />

                          {/* Concentration cone — wedge around μ communicates κ.
                              κ is high → narrow cone; we render a fixed compact
                              wedge since absolute κ → angle mapping is heuristic. */}
                          <path
                            d="M 110 100 L 168 52 A 76 76 0 0 0 152 36 Z"
                            fill="url(#vmfCone)"
                            stroke="rgb(77,124,255)"
                            strokeOpacity="0.32"
                            strokeWidth="0.6"
                          />

                          {/* μ direction vector */}
                          <line x1="110" y1="100" x2="162" y2="46" stroke="rgb(77,124,255)" strokeWidth="2" strokeOpacity="0.85" strokeLinecap="round" />
                          {/* μ endpoint */}
                          <circle cx="162" cy="46" r="3.5" fill="rgb(77,124,255)" />
                          <circle cx="162" cy="46" r="6" fill="none" stroke="rgb(77,124,255)" strokeOpacity="0.35" strokeWidth="1" />

                          {/* μ label */}
                          <text x="170" y="42" className="fill-accent" style={{ fontFamily: 'JetBrains Mono', fontSize: 11, fontWeight: 700 }}>μ</text>

                          {/* Origin */}
                          <circle cx="110" cy="100" r="2" fill="rgb(160,160,165)" />

                          {/* Annotation lines (subtle leaders) */}
                          <line x1="32" y1="100" x2="110" y2="100" stroke="rgb(255,255,255)" strokeWidth="0.5" strokeOpacity="0.08" strokeDasharray="2 3" />
                          <text x="18" y="103" className="fill-text-muted" style={{ fontFamily: 'Inter', fontSize: 8.5 }}>origin</text>

                          {/* Cone annotation */}
                          <text x="172" y="64" className="fill-text-secondary" style={{ fontFamily: 'Inter', fontSize: 8.5 }}>κ cone</text>
                        </svg>
                      </div>
                      <div className="mt-2 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-center text-[10px] text-text-muted">
                        <span>unit hypersphere</span>
                        <span className="text-border-emphasis">·</span>
                        <span>μ direction</span>
                        <span className="text-border-emphasis">·</span>
                        <span>concentration cone (κ)</span>
                      </div>
                    </div>

                    {/* ── Evidence readout — single tabular column.
                           κ and δ get strong numeric hierarchy; the other
                           three rows are quiet metadata. No nested mini-cards. ── */}
                    <div className="flex flex-col justify-center px-5 py-6 sm:py-7">
                      <p className="premium-kicker mb-3">Directional evidence</p>

                      <dl className="divide-y divide-white/[0.04]">
                        <div className="flex items-baseline justify-between gap-3 py-2.5">
                          <dt className="text-[11.5px] text-text-muted">
                            κ <span className="text-text-muted/60">angular sharpness</span>
                          </dt>
                          <dd className="font-mono text-[18px] font-semibold tabular-nums leading-none text-accent">
                            {kappaStr ?? case_.uncertainty.kappa.toFixed(1)}
                          </dd>
                        </div>
                        <div className="flex items-baseline justify-between gap-3 py-2.5">
                          <dt className="text-[11.5px] text-text-muted">
                            δ <span className="text-text-muted/60">ROI disagreement</span>
                          </dt>
                          <dd className="font-mono text-[16px] font-semibold tabular-nums leading-none text-status-warning">
                            {case_.uncertainty.delta.toFixed(3)}
                          </dd>
                        </div>
                        <div className="flex items-baseline justify-between gap-3 py-2">
                          <dt className="text-[11.5px] text-text-muted">Latent</dt>
                          <dd className="font-mono text-[12px] tabular-nums text-text-primary">768-D</dd>
                        </div>
                        <div className="flex items-baseline justify-between gap-3 py-2">
                          <dt className="text-[11.5px] text-text-muted">Output</dt>
                          <dd className="font-mono text-[12px] text-text-primary">μ direction</dd>
                        </div>
                        <div className="flex items-baseline justify-between gap-3 py-2">
                          <dt className="text-[11.5px] text-text-muted">Concentration</dt>
                          <dd className="text-[12px] text-accent">high · sharp</dd>
                        </div>
                      </dl>

                      <div className="mt-3">
                        <ProvenanceBadge provenance={DERIVED_PROV} />
                      </div>
                    </div>
                  </div>

                  {/* ── Cached reference CLIP latent profile — inline strip
                         folded into the same inset via a hairline divider.
                         Honest distinction from predicted μ. ── */}
                  {clipReveal && hasRealClip && clipHeights ? (
                    <div className="border-t border-white/[0.05] px-5 py-4">
                      <div className="mb-2 flex items-baseline justify-between gap-3">
                        <div className="min-w-0">
                          <p className="premium-kicker">Reference CLIP latent profile</p>
                          <p className="mt-0.5 text-[10.5px] text-text-muted">
                            Cached gallery-target embedding · <span className="text-text-secondary">not</span> the predicted μ
                          </p>
                        </div>
                        <ProvenanceBadge provenance={clipProv} />
                      </div>
                      <div className="relative h-12">
                        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-white/[0.05]" aria-hidden />
                        <div className="absolute inset-0 flex items-end gap-px">
                          {clipHeights.map((u, i) => (
                            <motion.div
                              key={i}
                              className="min-w-0 flex-1 bg-accent/40"
                              initial={{ height: 0 }}
                              animate={{ height: `${Math.max(3, u)}%` }}
                              transition={{ delay: i * 0.004, duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                              style={{ minHeight: 2 }}
                            />
                          ))}
                        </div>
                      </div>
                      <p className="mt-2 text-[10px] text-text-muted">
                        CLIP ViT-L/14 · {case_.clipPreview?.length ?? 96} bins of the 768-D reference vector ·{' '}
                        {isLiveInference
                          ? 'predicted μ available from the backend stream.'
                          : "predicted μ is not exported in this replay — never confuse it with the model's μ."}
                      </p>
                    </div>
                  ) : null}
                </div>
                </ComputationCard>
              </motion.div>
            )}

            {/* ── 06: CSLS gallery search ── */}
            {sub === 'gallery_search' && (
              <motion.div key="gallery" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <ComputationCard
                  step="Step 06 · CSLS"
                  title="CSLS gallery search"
                  subtitle="Hubness-corrected re-ranking over 10,000 CLIP gallery embeddings."
                  provenance={stepProv}
                >

                {(() => {
                  // ── Deterministic gallery cloud — 4 concentric rings of dots
                  //    represent the 10k gallery field around the query at center.
                  //    Reveal in proportion to scanned count so the cloud truly
                  //    animates the CSLS scan. ──
                  const rings = [
                    { r: 22, n: 10 },
                    { r: 38, n: 16 },
                    { r: 56, n: 22 },
                    { r: 74, n: 28 },
                  ];
                  const dots: { x: number; y: number; ring: number }[] = [];
                  rings.forEach((ring, ri) => {
                    for (let i = 0; i < ring.n; i++) {
                      const angle = (i / ring.n) * Math.PI * 2 + ri * 0.27;
                      const jitter = ((i * 19 + ring.r) % 9) - 4;
                      dots.push({
                        x: 110 + Math.cos(angle) * (ring.r + jitter * 0.6),
                        y: 90 + Math.sin(angle) * (ring.r + jitter * 0.6),
                        ring: ri,
                      });
                    }
                  });
                  const scanFrac = Math.min(1, galCount / 10000);
                  const dotsScanned = Math.floor(scanFrac * dots.length);
                  const top3 = topK.slice(0, 3);
                  return (
                    <div className="workbench-inset !p-5">
                      <div className="grid items-stretch gap-5 lg:grid-cols-[minmax(120px,0.7fr)_minmax(0,2fr)_minmax(150px,1fr)]">

                        {/* A · QUERY EMBEDDING — latent vector as horizontal
                            channel strips. Reads as a 768-D direction, not a
                            game-board. */}
                        <div className="flex flex-col">
                          <p className="premium-kicker">Query embedding</p>
                          <div
                            className="relative mt-3 flex-1 overflow-hidden rounded-md px-2 py-2"
                            aria-hidden
                            style={{
                              minHeight: 170,
                              background: 'rgb(77 124 255 / 0.045)',
                              boxShadow: 'inset 0 0 0 1px rgb(77 124 255 / 0.32)',
                            }}
                          >
                            <div className="flex h-full w-full flex-col justify-between gap-[3px]">
                              {Array.from({ length: 10 }).map((_, i) => {
                                const intensity = 0.28 + ((i * 41 + 17) % 60) / 110;
                                const len = 48 + ((i * 31 + 19) % 52);
                                return (
                                  <div
                                    key={i}
                                    className="flex-1 rounded-[1.5px]"
                                    style={{
                                      background: `rgb(77 124 255 / ${intensity.toFixed(3)})`,
                                      width: `${len}%`,
                                    }}
                                  />
                                );
                              })}
                            </div>
                          </div>
                          <p className="mt-2.5 font-mono text-[13px] font-semibold tabular-nums leading-none text-accent">
                            768-D
                          </p>
                          <p className="mt-1 text-[10px] leading-tight text-text-muted">
                            {isLiveInference ? 'predicted CLIP direction · μ' : 'cached CLIP direction · μ'}
                          </p>
                        </div>

                        {/* B · GALLERY CLOUD with DENSITY RING — embedding-
                            space figure, faint coordinate plane, refined
                            μ marker, label as a corner callout. */}
                        <div className="flex flex-col">
                          <div className="mb-2.5 flex items-baseline justify-between gap-3">
                            <p className="premium-kicker">Density-corrected scan</p>
                            <span className="font-mono text-[10.5px] tabular-nums text-text-muted">
                              <span className="text-text-secondary">{galCount.toLocaleString()}</span>
                              <span className="text-text-muted/70"> / 10,000 scanned</span>
                            </span>
                          </div>
                          <div
                            className="relative flex-1 overflow-hidden rounded-md bg-white/[0.012]"
                            style={{ minHeight: 210, boxShadow: 'inset 0 0 0 1px rgb(255 255 255 / 0.04)' }}
                          >
                            <svg viewBox="0 0 220 180" className="h-full w-full" aria-hidden>
                              <defs>
                                {/* Faint dot grid to suggest a coordinate plane
                                    without drawing visible axes. */}
                                <pattern id="csls-grid" width="11" height="11" patternUnits="userSpaceOnUse">
                                  <circle cx="0.5" cy="0.5" r="0.4" fill="rgb(255,255,255)" fillOpacity="0.045" />
                                </pattern>
                              </defs>
                              <rect x="0" y="0" width="220" height="180" fill="url(#csls-grid)" />

                              {/* Center cross — minimal, almost imperceptible */}
                              <line x1="6" y1="90" x2="214" y2="90" stroke="rgb(255,255,255)" strokeOpacity="0.030" strokeWidth="0.4" />
                              <line x1="110" y1="6" x2="110" y2="174" stroke="rgb(255,255,255)" strokeOpacity="0.030" strokeWidth="0.4" />

                              {/* Density / local-neighborhood ring around the query */}
                              <circle
                                cx="110"
                                cy="90"
                                r="32"
                                fill="rgb(77,124,255)"
                                fillOpacity="0.045"
                                stroke="rgb(77,124,255)"
                                strokeOpacity="0.32"
                                strokeWidth="0.6"
                                strokeDasharray="2 2.5"
                              />

                              {/* Tiny callout leader from the ring's edge to a
                                  corner label — replaces the oversized
                                  "neighborhood" word in-canvas. */}
                              <line x1="140" y1="63" x2="172" y2="46" stroke="rgb(123,156,255)" strokeOpacity="0.45" strokeWidth="0.5" />
                              <text
                                x="173"
                                y="44"
                                className="fill-text-secondary"
                                style={{ fontFamily: 'Inter', fontSize: 7.5, letterSpacing: '0.06em', textTransform: 'uppercase' }}
                              >
                                local density
                              </text>

                              {/* Gallery dots — smaller, more precise. Scanned
                                  dots go accent; unscanned stay near-invisible. */}
                              {dots.map((p, i) => {
                                const scanned = i < dotsScanned;
                                return (
                                  <circle
                                    key={i}
                                    cx={p.x}
                                    cy={p.y}
                                    r={scanned ? 1.4 : 1.1}
                                    fill={scanned ? 'rgb(77,124,255)' : 'rgb(255,255,255)'}
                                    fillOpacity={scanned ? 0.72 : 0.14}
                                  />
                                );
                              })}

                              {/* Top-3 emerging connectors — drawn to ring-0
                                  dots after the scan crosses thresholds. */}
                              {top3.map((_, idx) => {
                                const target = dots[idx * 3];
                                if (!target) return null;
                                const visible = scanFrac > 0.55 + idx * 0.12;
                                if (!visible) return null;
                                return (
                                  <line
                                    key={idx}
                                    x1="110"
                                    y1="90"
                                    x2={target.x}
                                    y2={target.y}
                                    stroke="rgb(77,124,255)"
                                    strokeOpacity={idx === 0 ? 0.85 : 0.55 - idx * 0.12}
                                    strokeWidth={idx === 0 ? 1.1 : 0.8}
                                    strokeLinecap="round"
                                    strokeDasharray={idx === 0 ? '0' : '1 1.5'}
                                  />
                                );
                              })}

                              {/* Query at center — elegant double dot */}
                              <circle cx="110" cy="90" r="3.4" fill="rgb(77,124,255)" />
                              <circle cx="110" cy="90" r="5.6" fill="none" stroke="rgb(77,124,255)" strokeOpacity="0.40" strokeWidth="0.65" />
                              <text
                                x="116"
                                y="86"
                                className="fill-accent"
                                style={{ fontFamily: 'JetBrains Mono', fontSize: 8.5, fontWeight: 700 }}
                              >
                                μ
                              </text>
                            </svg>

                            {/* Hair-thin progress line at the canvas bottom —
                                no card, no rounded track. Reads as a precise
                                instrument readout. */}
                            <div className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-white/[0.04]" />
                            <motion.div
                              className="pointer-events-none absolute bottom-0 left-0 h-px bg-accent/70"
                              animate={{ width: `${scanFrac * 100}%` }}
                              transition={{ duration: 0.1 }}
                            />
                          </div>
                          <p className="mt-2.5 text-[10.5px] leading-relaxed text-text-muted">
                            CSLS re-ranks gallery embeddings by correcting local-neighborhood density, reducing hubness before top-k retrieval.
                          </p>
                        </div>

                        {/* C · RANKED OUTPUT — refined retrieval list. Thin
                            tracks, mono numeric values, #1 subtly emphasized. */}
                        <div className="flex flex-col">
                          <p className="premium-kicker">Top-K output</p>
                          <ol className="mt-3 space-y-2">
                            {top3.map((item, idx) => {
                              const revealed = scanFrac > 0.55 + idx * 0.12;
                              const maxCsls = top3[0]?.csls ?? 1;
                              const pct = Math.max(2, ((item.csls ?? 0) / Math.max(maxCsls, 0.001)) * 100);
                              const isTop = item.rank === 1;
                              return (
                                <li
                                  key={item.rank}
                                  className="space-y-1 transition-opacity duration-300"
                                  style={{ opacity: revealed ? 1 : 0.18 }}
                                >
                                  <div className="flex items-baseline justify-between gap-2">
                                    <span
                                      className={`font-mono text-[10.5px] font-semibold tabular-nums ${
                                        isTop ? 'text-accent' : 'text-text-muted'
                                      }`}
                                    >
                                      #{item.rank}
                                    </span>
                                    <span
                                      className={`font-mono text-[10.5px] tabular-nums ${
                                        isTop ? 'text-text-primary' : 'text-text-secondary'
                                      }`}
                                    >
                                      {item.csls != null ? item.csls.toFixed(3) : '—'}
                                    </span>
                                  </div>
                                  <div className="relative h-[3px] overflow-hidden rounded-full bg-white/[0.04]">
                                    <motion.div
                                      className={`absolute inset-y-0 left-0 rounded-full ${
                                        isTop ? 'bg-accent/75' : 'bg-text-secondary/40'
                                      }`}
                                      initial={{ width: 0 }}
                                      animate={{ width: revealed ? `${pct}%` : 0 }}
                                      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                                    />
                                  </div>
                                </li>
                              );
                            })}
                          </ol>
                          {/* Compact metadata — hair divider, aligned bottom */}
                          <div className="mt-auto border-t border-white/[0.04] pt-3">
                            <dl className="flex items-baseline justify-between gap-3 text-[10.5px]">
                              <div className="flex items-baseline gap-1.5">
                                <dt className="text-text-muted/85">Space</dt>
                                <dd className="font-mono tabular-nums text-text-secondary">ViT-L/14</dd>
                              </div>
                              <span className="h-3 w-px bg-white/[0.05]" aria-hidden />
                              <div className="flex items-baseline gap-1.5">
                                <dt className="text-text-muted/85">Method</dt>
                                <dd className="font-mono text-text-secondary">CSLS</dd>
                              </div>
                            </dl>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })()}

                </ComputationCard>
              </motion.div>
            )}

            {/* ── 07: Retrieval complete (the climax) ── */}
            {(sub === 'results' || sub === 'done') && (
              <motion.div
                key="results"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-5"
              >
                <HeroResultPanel
                  case_={case_}
                  top1={topK[0]}
                  provenance={
                    <ProvenanceBadge
                      provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Cached retrieval ranking' }}
                    />
                  }
                  liveLabel={
                    isLiveInference ? (
                      <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/20 bg-accent/10 px-2.5 py-1 text-[10px] font-semibold text-accent">
                        <span className="h-1.5 w-1.5 rounded-full bg-accent pulse-dot" />
                        LIVE CUDA
                      </span>
                    ) : null
                  }
                />

                {/* ── Slim CSLS distribution strip — no card chrome, just a
                       quiet evidence readout under the hero. Removes the prior
                       Neural-decoding-path card (already explained by the trace
                       rail) and the duplicate bordered scores panel. ── */}
                <div className="px-1">
                  <div className="mb-2 flex items-baseline justify-between gap-3">
                    <p className="premium-kicker">Top-5 CSLS scores</p>
                    <span className="text-[10.5px] text-text-muted">hubness-corrected · 10k gallery</span>
                  </div>
                  <ol className="space-y-1.5">
                    {topK.map((item) => {
                      const maxCsls = topK[0]?.csls ?? 1;
                      const pct = Math.max(2, ((item.csls ?? 0) / Math.max(maxCsls, 0.001)) * 100);
                      const isTop = item.rank === 1;
                      return (
                        <li key={item.rank} className="flex items-center gap-3">
                          <span className={`w-6 text-right font-mono text-[11px] font-semibold tabular-nums ${isTop ? 'text-accent' : 'text-text-muted'}`}>
                            #{item.rank}
                          </span>
                          <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.04]">
                            <motion.div
                              className={`absolute inset-y-0 left-0 rounded-full ${isTop ? 'bg-accent/70' : 'bg-text-secondary/45'}`}
                              initial={{ width: 0 }}
                              animate={{ width: `${pct}%` }}
                              transition={{ duration: 0.55, delay: item.rank * 0.04, ease: [0.22, 1, 0.36, 1] }}
                            />
                          </div>
                          <span className={`w-[68px] text-right font-mono text-[11.5px] font-semibold tabular-nums ${isTop ? 'text-text-primary' : 'text-text-secondary'}`}>
                            {item.csls?.toFixed(4)}
                          </span>
                        </li>
                      );
                    })}
                  </ol>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* ── Integrated workbench footer: candidates / next-output cue,
              computation details, and proceed CTA share one rhythm anchored
              to the active panel above. Replaces three floating siblings. ── */}
      <section className="premium-panel px-5 py-5 sm:px-6 sm:py-5" aria-label="Workbench footer">
        {/* Candidates strip or next-output cue */}
        <AnimatePresence mode="wait">
          {candidateStripReady ? (
            <motion.div
              key="strip"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-[14px] font-semibold text-text-primary">
                  Top-{topK.length} retrieved candidates
                </h3>
                {visRanks.length > 0 && getStatus('results') === 'done' && (
                  <ProvenanceBadge provenance={isLiveInference ? LIVE_PROV : { kind: 'replay', detail: 'Precomputed gallery ranking' }} />
                )}
              </div>
              {visRanks.length > 0 ? (
                <RetrievalCandidatesStrip candidates={topK} visibleRanks={visRanks} />
              ) : (
                <p className="text-[12px] text-text-muted">
                  Resolving ranks from CSLS scores…
                </p>
              )}
            </motion.div>
          ) : (
            <motion.div
              key="cue"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex items-center gap-3"
            >
              <span className="inline-flex h-2 w-2 rounded-full bg-accent/60" aria-hidden />
              <p className="text-[12px] text-text-secondary">
                Next: Top-K candidates appear here once CSLS ranking completes.
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Proceed CTA — alone, no preceding "Computation details" disclosure
            (the prior Mode/Encoder/Retrieval text was low-value duplication
            of evidence already carried by the provenance badges and rail). */}
        <AnimatePresence>
          {showProceed && (
            <motion.div
              className="mt-5 flex justify-end border-t border-border-subtle/70 pt-4"
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <button
                type="button"
                onClick={onComplete}
                className="premium-button-primary"
              >
                {hasReconAsset ? 'Open reconstruction evidence' : 'Open evidence review'}
                <svg className="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </section>
    </div>
  );
}
