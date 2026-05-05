import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase, RetrievedImage } from '@/types';
import { IconPlay } from '@/components/ui/Icon';

export interface PhaseEncodingRetrievalProps {
  case_: DemoCase;
  onComplete: () => void;
}

type Subphase = 'load_betas' | 'zscore' | 'roi_mask' | 'roi_encode' | 'vmf_decode' | 'gallery_search' | 'results' | 'done';

const STEPS: { phase: Subphase; label: string; detail: string; duration: number }[] = [
  { phase: 'load_betas', label: 'Load fMRI Betas', detail: 'Loading pre-extracted beta volume (.npy, float32)', duration: 1400 },
  { phase: 'zscore', label: 'Z-Score Normalize', detail: 'Per-session voxel z-score normalization', duration: 1100 },
  { phase: 'roi_mask', label: 'ROI Masking', detail: 'Applying nsdgeneral mask → 15,724 voxels', duration: 900 },
  { phase: 'roi_encode', label: 'ROI Transformer', detail: '17 ROI tokens → Transformer ×6 → [CLS] 768-D', duration: 1800 },
  { phase: 'vmf_decode', label: 'vMF Decode', detail: '(μ, κ) on CLIP L2 hypersphere (768-D)', duration: 1400 },
  { phase: 'gallery_search', label: 'CSLS Search', detail: 'Ranking 10,000 gallery embeddings', duration: 1800 },
  { phase: 'results', label: 'Top-K Results', detail: 'Retrieved candidates ranked', duration: 1600 },
];

const ROI_TOKENS = [
  { name: 'V1v', fullName: 'Ventral V1 — primary visual cortex, ventral division', c: '#22d3ee' },
  { name: 'V1d', fullName: 'Dorsal V1 — primary visual cortex, dorsal division', c: '#38bdf8' },
  { name: 'V2v', fullName: 'Ventral V2 — secondary visual cortex, ventral division', c: '#818cf8' },
  { name: 'V2d', fullName: 'Dorsal V2 — secondary visual cortex, dorsal division', c: '#a78bfa' },
  { name: 'V3v', fullName: 'Ventral V3 — tertiary visual cortex, ventral division', c: '#c084fc' },
  { name: 'V3d', fullName: 'Dorsal V3 — tertiary visual cortex, dorsal division', c: '#e879f9' },
  { name: 'V3A', fullName: 'Visual area V3A — dorsal stream association', c: '#f472b6' },
  { name: 'V3B', fullName: 'Visual area V3B — dorsal stream association', c: '#34d399' },
  { name: 'V4', fullName: 'Fourth visual area — intermediate shape/color processing', c: '#4ade80' },
  { name: 'FFA1', fullName: 'Fusiform face area 1 — category-selective cortex', c: '#fbbf24' },
  { name: 'FFA2', fullName: 'Fusiform face area 2 — category-selective cortex', c: '#fb923c' },
  { name: 'PPA', fullName: 'Parahippocampal place area — scene-selective cortex', c: '#f87171' },
  { name: 'EBA', fullName: 'Extrastriate body area — body-selective cortex', c: '#94a3b8' },
  { name: 'OFA', fullName: 'Occipital face area — face-selective cortex', c: '#64748b' },
  { name: 'OPA', fullName: 'Occipital place area — scene-selective cortex', c: '#2dd4bf' },
  { name: 'RSC', fullName: 'Retrosplenial cortex — navigation / spatial context', c: '#60a5fa' },
  { name: 'other', fullName: 'Residual masked cortex within nsdgeneral (non-atlas overlap)', c: '#f59e0b' },
] as const;

const CLIP_SEGMENT_GRADS = [
  'bg-gradient-to-t from-indigo-950/95 via-violet-700/90 to-fuchsia-400/70',
  'bg-gradient-to-t from-violet-950/95 via-purple-700/85 to-pink-400/65',
  'bg-gradient-to-t from-fuchsia-950/95 via-fuchsia-600/85 to-rose-400/65',
  'bg-gradient-to-t from-slate-900/95 via-indigo-600/85 to-violet-400/70',
  'bg-gradient-to-t from-violet-950/95 via-blue-700/80 to-cyan-400/65',
  'bg-gradient-to-t from-purple-950/95 via-violet-600/85 to-fuchsia-300/60',
  'bg-gradient-to-t from-indigo-950/95 via-sky-700/85 to-teal-400/65',
  'bg-gradient-to-t from-fuchsia-950/95 via-purple-700/85 to-indigo-400/65',
] as const;

function clipGroupedMeans(raw768: number[], segments = 64, chunk = 12): number[] {
  const out: number[] = [];
  for (let i = 0; i < segments; i++) {
    let s = 0;
    const base = i * chunk;
    for (let j = 0; j < chunk; j++) s += raw768[base + j] ?? 0;
    out.push(s / chunk);
  }
  return out;
}

function preview64(c: DemoCase): number[] {
  const p = c.fmriPreview;
  return p.length >= 64 ? p.slice(0, 64) : [...p, ...Array(64 - p.length).fill(0)];
}

function normalize(vals: number[]): number[] {
  let mn = Infinity, mx = -Infinity;
  for (const v of vals) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const s = mx - mn || 1;
  return vals.map((v) => ((v - mn) / s) * 100);
}

function seedRng(id: string): () => number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = ((h << 5) - h + id.charCodeAt(i)) | 0;
  let a = (Math.abs(h) >>> 0) || 1;
  return () => { let t = (a += 0x6d2b79f5); t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function makeClipBars(id: string, n: number): number[] {
  const r = seedRng(`${id}-clip`);
  return Array.from({ length: n }, () => 0.12 + r() * 0.88);
}

function stepIdx(s: Subphase): number { const i = STEPS.findIndex((x) => x.phase === s); return i >= 0 ? i : STEPS.length; }

function SafeImg({ src, alt, className }: { src?: string; alt: string; className?: string }) {
  const [ok, setOk] = useState(true);
  if (!src || !ok) return <div className={`flex items-center justify-center bg-slate-900/80 ${className ?? ''}`}><span className="text-[10px] text-slate-700">—</span></div>;
  return <img src={src} alt={alt} className={className} onError={() => setOk(false)} loading="lazy" />;
}

function LabelBadge({ label }: { label: RetrievedImage['label'] }) {
  const m = label === 'correct'
    ? { t: 'Match', bg: 'bg-emerald-500/20', border: 'border-emerald-500/40', text: 'text-emerald-200' }
    : label === 'semantic_neighbor'
      ? { t: 'Neighbor', bg: 'bg-amber-500/15', border: 'border-amber-500/35', text: 'text-amber-200' }
      : { t: 'Distractor', bg: 'bg-slate-600/20', border: 'border-slate-500/30', text: 'text-slate-300' };
  return <span className={`rounded-md border px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider ${m.bg} ${m.border} ${m.text}`}>{m.t}</span>;
}

export function PhaseEncodingRetrieval({ case_, onComplete }: PhaseEncodingRetrievalProps) {
  const skipRef = useRef(false);
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const [sub, setSub] = useState<Subphase>('load_betas');
  const [progress, setProgress] = useState(0);
  const [hlIdx, setHlIdx] = useState(-1);
  const [clipReveal, setClipReveal] = useState(false);
  const [galCount, setGalCount] = useState(0);
  const [visRanks, setVisRanks] = useState<number[]>([]);
  const [showProceed, setShowProceed] = useState(false);
  const [kappaStr, setKappaStr] = useState<string | null>(null);

  const heights = useMemo(() => normalize(preview64(case_)), [case_]);
  const clipH768 = useMemo(() => makeClipBars(case_.id, 768), [case_.id]);
  const clipGrouped64 = useMemo(() => clipGroupedMeans(clipH768), [clipH768]);
  const topK = useMemo(() => [...case_.retrievedImages].sort((a, b) => a.rank - b.rank).slice(0, 5), [case_.retrievedImages]);

  const pushT = useCallback((fn: () => void, ms: number) => {
    const id = setTimeout(() => { if (!skipRef.current) fn(); }, ms);
    timersRef.current.push(id);
  }, []);
  const clearT = useCallback(() => { timersRef.current.forEach(clearTimeout); timersRef.current = []; }, []);

  const skip = useCallback(() => {
    skipRef.current = true; clearT();
    setSub('done'); setProgress(1); setHlIdx(-1); setClipReveal(true);
    setGalCount(10000); setVisRanks(topK.map((r) => r.rank));
    setKappaStr(case_.uncertainty.kappa.toFixed(1)); setShowProceed(true);
  }, [clearT, topK, case_.uncertainty.kappa]);

  useEffect(() => {
    skipRef.current = false; clearT();
    setSub('load_betas'); setProgress(0); setHlIdx(-1);
    setClipReveal(false); setGalCount(0); setVisRanks([]); setKappaStr(null); setShowProceed(false);

    let t = 0;
    // 1: load betas
    for (let i = 0; i <= 14; i++) pushT(() => setProgress(i / 14), t + i * 90);
    t += STEPS[0].duration;
    // 2: zscore
    pushT(() => { setSub('zscore'); setProgress(0); }, t);
    let hi = 0;
    const hop = () => { setHlIdx(hi % 64); hi++; if (hi < 36) pushT(hop, 28); };
    pushT(hop, t + 40);
    for (let i = 0; i <= 11; i++) pushT(() => setProgress(i / 11), t + i * 90);
    t += STEPS[1].duration;
    // 3: roi mask
    pushT(() => { setSub('roi_mask'); setProgress(0); setHlIdx(-1); }, t);
    for (let i = 0; i <= 9; i++) pushT(() => setProgress(i / 9), t + i * 90);
    t += STEPS[2].duration;
    // 4: roi encode
    pushT(() => { setSub('roi_encode'); setProgress(0); }, t);
    for (let i = 0; i <= 18; i++) pushT(() => setProgress(i / 18), t + i * 90);
    t += STEPS[3].duration;
    // 5: vmf decode
    pushT(() => { setSub('vmf_decode'); setProgress(0); setClipReveal(true); }, t);
    for (let i = 0; i <= 14; i++) pushT(() => setProgress(i / 14), t + i * 90);
    pushT(() => setKappaStr(case_.uncertainty.kappa.toFixed(1)), t + 900);
    t += STEPS[4].duration;
    // 6: gallery
    pushT(() => { setSub('gallery_search'); setProgress(0); }, t);
    for (let s = 0; s <= 48; s++) {
      pushT(() => { const p = s / 48; const e = 1 - (1 - p) ** 2; setGalCount(Math.round(e * 10000)); setProgress(p); }, t + (STEPS[5].duration * s) / 48);
    }
    t += STEPS[5].duration;
    // 7: results
    pushT(() => { setSub('results'); setProgress(0); }, t);
    topK.forEach((_, i) => {
      pushT(() => { setVisRanks((prev) => { const r = topK[i].rank; return prev.includes(r) ? prev : [...prev, r]; }); setProgress((i + 1) / topK.length); }, t + 280 * (i + 1));
    });
    t += STEPS[6].duration;
    pushT(() => { setSub('done'); setShowProceed(true); }, t);

    return () => clearT();
  }, [case_.id, clearT, pushT, topK, case_.uncertainty.kappa]);

  const ci = stepIdx(sub);
  const isA = (p: Subphase) => sub === p;
  const isD = (p: Subphase) => ci > stepIdx(p) || sub === 'done';
  const st = (p: Subphase): 'w' | 'a' | 'd' => isA(p) ? 'a' : isD(p) ? 'd' : 'w';

  return (
    <div className="relative space-y-4">
      {/* Header */}
      <motion.div
        className="flex items-center justify-between rounded-2xl border border-white/[0.06] bg-gradient-to-r from-slate-900/80 via-brain-navy/60 to-slate-900/80 p-5 backdrop-blur-xl"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div>
          <h2 className="text-base font-semibold text-white sm:text-lg">Encoding & Retrieval</h2>
          <p className="mt-0.5 font-mono text-[11px] text-slate-500">
            {case_.subject} · nsdId {case_.nsdId} · session {case_.session}
          </p>
        </div>
        <button
          type="button"
          onClick={skip}
          className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2 text-[10px] font-bold uppercase tracking-wider text-slate-400 transition hover:border-brain-accent/30 hover:text-brain-accent"
        >
          Skip
        </button>
      </motion.div>

      {/* Pipeline progress chips */}
      <div className="flex flex-wrap gap-1.5 rounded-xl border border-white/[0.04] bg-black/20 px-2 py-2 sm:px-3 sm:py-2.5">
        {STEPS.map((step, i) => {
          const s = st(step.phase);
          return (
            <div
              key={step.phase}
              className={`flex min-w-0 max-w-[calc(100%-0.25rem)] items-center gap-1 rounded-lg px-2 py-1.5 text-[8px] font-bold uppercase leading-snug tracking-wide transition-all duration-300 sm:gap-1.5 sm:px-2.5 sm:text-[9px] sm:tracking-wider ${
                s === 'a' ? 'bg-brain-accent/12 text-brain-accent ring-1 ring-brain-accent/25'
                : s === 'd' ? 'bg-emerald-500/8 text-emerald-400'
                : 'text-slate-600'
              }`}
            >
              {s === 'd' ? <span className="shrink-0 text-emerald-400">✓</span>
                : s === 'a' ? <motion.span className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-brain-accent" animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 0.8, repeat: Infinity }} />
                : <span className="inline-block h-1 w-1 shrink-0 rounded-full bg-slate-700" />}
              <span className="min-w-0 break-words">{step.label}</span>
              <span className="shrink-0 text-[7px] font-mono opacity-60 sm:hidden">{String(i + 1).padStart(2, '0')}</span>
            </div>
          );
        })}
      </div>

      {/* Steps */}
      <div className="space-y-3">
        {/* 1: Load Betas */}
        <Step title="1 · Load fMRI Betas" s={st('load_betas')} p={isA('load_betas') ? progress : isD('load_betas') ? 1 : 0}>
          <p className="text-[13px] text-slate-300">
            {isD('load_betas') ? 'Beta volume loaded.' : isA('load_betas') ? 'Loading NIfTI beta volume from pre-extracted .npy...' : 'Waiting...'}
          </p>
          <div className="mt-1 font-mono text-[9px] text-slate-600">
            Shape: (~30,000 × ~15,724) · float32 · ROI-masked
          </div>
        </Step>

        {/* 2: Z-score */}
        <Step title="2 · Per-session Z-Score" s={st('zscore')} p={isA('zscore') ? progress : isD('zscore') ? 1 : 0}>
          <p className="text-[13px] text-slate-300">
            {isD('zscore') ? 'Z-score normalization complete.' : isA('zscore') ? 'Normalizing each voxel within session...' : 'Waiting...'}
          </p>
          {(isA('zscore') || isD('zscore')) && (
            <div className="mt-2 flex h-10 items-end gap-px rounded-lg bg-black/30 p-1 ring-1 ring-white/[0.04]">
              {heights.map((h, i) => (
                <motion.div key={i}
                  className={`min-w-0 flex-1 rounded-[1px] transition-colors duration-100 ${
                    isA('zscore') && i === hlIdx ? 'bg-white' : 'bg-gradient-to-t from-cyan-800/70 to-cyan-500/50'
                  }`}
                  animate={{ height: `${Math.max(8, h)}%` }}
                  style={{ minHeight: 2 }}
                />
              ))}
            </div>
          )}
        </Step>

        {/* 3: ROI Mask */}
        <Step title="3 · ROI Masking" s={st('roi_mask')} p={isA('roi_mask') ? progress : isD('roi_mask') ? 1 : 0}>
          <p className="text-[13px] text-slate-300">
            {isD('roi_mask') ? '15,724 visual cortex voxels retained.' : isA('roi_mask') ? 'Applying nsdgeneral.nii.gz mask...' : 'Waiting...'}
          </p>
          <div className="mt-1 font-mono text-[9px] text-slate-600">nsdgeneral.nii.gz · 1.8mm iso · 7T</div>
        </Step>

        {/* 4: ROI Transformer */}
        <Step title="4 · ROI Transformer" s={st('roi_encode')} p={isA('roi_encode') ? progress : isD('roi_encode') ? 1 : 0}>
          <p className="text-[13px] text-slate-300">
            {isD('roi_encode') ? '[CLS] embedding computed from 17 ROI tokens.' : isA('roi_encode') ? 'Tokenizing into 17 ROI regions → Transformer ×6...' : 'Waiting...'}
          </p>
          {(isA('roi_encode') || isD('roi_encode')) && (
            <div className="mt-3 flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
              <div className="flex flex-wrap justify-center gap-1">
                {ROI_TOKENS.map((r, i) => (
                  <motion.div key={r.name}
                    className="flex flex-col items-center"
                    initial={{ scale: 0 }} animate={{ scale: 1 }}
                    transition={{ delay: i * 0.03, type: 'spring', stiffness: 300, damping: 20 }}
                  >
                    <div
                      title={r.fullName}
                      className="h-6 w-6 rounded-full border border-white/15"
                      style={{ backgroundColor: `${r.c}77` }}
                    />
                    <span className="mt-0.5 text-[9px] font-bold leading-tight text-slate-600">{r.name}</span>
                  </motion.div>
                ))}
              </div>
              <div className="flex items-center gap-2 text-slate-600">
                <span>→</span>
                <div className="rounded-lg border border-violet-500/30 bg-violet-950/25 px-3 py-2 text-center">
                  <p className="text-[9px] font-bold uppercase tracking-widest text-violet-300">Transformer ×6</p>
                  <p className="text-[8px] text-slate-500">12 heads · 768-D</p>
                </div>
                <span>→</span>
                <div className="rounded-lg border border-emerald-500/30 bg-emerald-950/25 px-3 py-2">
                  <p className="text-[9px] font-bold uppercase text-emerald-300">[CLS]</p>
                </div>
              </div>
            </div>
          )}
        </Step>

        {/* 5: vMF Decode */}
        <Step title="5 · vMF Decode → CLIP Space" s={st('vmf_decode')} p={isA('vmf_decode') ? progress : isD('vmf_decode') ? 1 : 0}>
          <p className="text-[13px] text-slate-300">
            {isD('vmf_decode') ? `μ on unit sphere, κ = ${kappaStr ?? '—'}` : isA('vmf_decode') ? 'Projecting through vMF head → (μ, κ)...' : 'Waiting...'}
          </p>
          {clipReveal && (
            <>
              <div className="mt-2 flex h-14 w-full items-end gap-px overflow-hidden rounded-lg bg-black/30 p-1 ring-1 ring-white/[0.04]">
                {clipGrouped64.map((u, i) => (
                  <motion.div key={i}
                    className={`min-h-[2px] min-w-0 flex-1 rounded-[1px] ${CLIP_SEGMENT_GRADS[i % CLIP_SEGMENT_GRADS.length]}`}
                    initial={{ height: '2%' }}
                    animate={{ height: `${Math.max(6, u * 100)}%` }}
                    transition={{ delay: i * 0.012, duration: 0.22 }}
                  />
                ))}
              </div>
              {kappaStr && (
                <motion.div className="mt-2 flex gap-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                  <span className="rounded-lg border border-cyan-500/30 bg-cyan-950/25 px-2.5 py-1 font-mono text-[11px] font-bold text-cyan-200">κ = {kappaStr}</span>
                  <span className="rounded-lg border border-amber-500/30 bg-amber-950/25 px-2.5 py-1 font-mono text-[11px] font-bold text-amber-200">δ = {case_.uncertainty.delta.toFixed(3)}</span>
                </motion.div>
              )}
            </>
          )}
        </Step>

        {/* 6: Gallery Search */}
        <Step title="6 · CSLS Gallery Search" s={st('gallery_search')} p={isA('gallery_search') ? progress : isD('gallery_search') ? 1 : 0}>
          <p className="text-[13px] text-slate-300">
            {isD('gallery_search') ? 'Gallery ranked — top candidates found.' : isA('gallery_search') ? 'Computing CSLS-corrected cosine similarity...' : 'Waiting...'}
          </p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="font-mono text-3xl font-bold tabular-nums text-brain-accent">{galCount.toLocaleString()}</span>
            <span className="text-sm text-slate-600">/ 10,000 embeddings</span>
          </div>
        </Step>

        {/* 7: Top-5 Results */}
        <motion.section
          className={`overflow-hidden rounded-2xl border p-4 transition-colors duration-300 sm:p-5 ${
            st('results') === 'a' ? 'border-violet-500/25 bg-gradient-to-b from-violet-950/20 to-slate-900/50'
              : st('results') === 'd' || sub === 'done' ? 'border-emerald-500/15 bg-gradient-to-b from-emerald-950/10 to-slate-900/50'
              : 'border-white/[0.04] bg-slate-900/30'
          }`}
          animate={{ opacity: st('results') === 'w' ? 0.4 : 1 }}
        >
          <div className="mb-3 flex items-center gap-2">
            <h3 className={`text-[11px] font-bold uppercase tracking-[0.18em] ${
              st('results') === 'a' ? 'text-violet-300' : st('results') === 'd' || sub === 'done' ? 'text-emerald-400' : 'text-slate-600'
            }`}>7 · Top-{topK.length} Retrieved Candidates</h3>
            {(isD('results') || sub === 'done') && <span className="text-xs text-emerald-400">✓</span>}
          </div>

          <div className={`grid gap-3 ${topK.length <= 3 ? 'grid-cols-3' : 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5'}`}>
            {topK.map((item) => {
              const vis = visRanks.includes(item.rank);
              return (
                <motion.div key={item.rank}
                  className="overflow-hidden rounded-xl border border-white/[0.06] bg-black/20 transition-transform duration-300 hover:scale-105 hover:transition-transform"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: vis ? 1 : 0, scale: vis ? 1 : 0.9 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 25 }}
                >
                  <div className="relative aspect-square w-full">
                    <SafeImg src={item.image} alt={`Rank ${item.rank}`} className="h-full w-full object-cover" />
                    <div className="absolute left-1.5 top-1.5">
                      <span className={`rounded-md px-1.5 py-0.5 text-[9px] font-bold text-white shadow-lg ${
                        item.rank === 1 ? 'bg-violet-600' : 'bg-slate-700/80'
                      }`}>
                        #{item.rank}
                      </span>
                    </div>
                  </div>
                  <div className="space-y-1 p-2">
                    <LabelBadge label={item.label} />
                    <div className="font-mono text-[9px] text-slate-500">
                      cos {item.score.toFixed(3)}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </motion.section>
      </div>

      {/* Proceed */}
      <AnimatePresence>
        {showProceed && (
          <motion.div className="sticky bottom-4 z-10 mt-6 flex justify-center"
            initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}>
            <button type="button" onClick={onComplete}
              className="flex items-center gap-3 rounded-2xl border border-emerald-400/40 bg-emerald-950/40 px-8 py-4 text-sm font-bold text-emerald-200 shadow-[0_8px_40px_rgba(16,185,129,0.15)] backdrop-blur-xl transition hover:border-emerald-400/60">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald-500/20">
                <IconPlay className="h-4 w-4 text-emerald-200" aria-hidden />
              </span>
              Proceed to Reconstruction
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Step({ title, s, p, children }: { title: string; s: 'w' | 'a' | 'd'; p: number; children: React.ReactNode }) {
  return (
    <motion.section
      className={`overflow-hidden rounded-2xl border p-4 transition-colors duration-300 sm:p-5 ${
        s === 'a' ? 'border-brain-accent/20 bg-gradient-to-b from-cyan-950/15 to-slate-900/50'
          : s === 'd' ? 'border-emerald-500/15 bg-gradient-to-b from-emerald-950/8 to-slate-900/40'
          : 'border-white/[0.04] bg-slate-900/30'
      }`}
      animate={{ opacity: s === 'w' ? 0.35 : 1 }}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <h3 className={`text-[11px] font-bold uppercase tracking-[0.18em] ${
          s === 'a' ? 'text-brain-accent' : s === 'd' ? 'text-emerald-400' : 'text-slate-600'
        }`}>{title}</h3>
        {s === 'd' && <span className="text-xs text-emerald-400">✓</span>}
        {s === 'a' && <motion.span className="inline-block h-1.5 w-1.5 rounded-full bg-brain-accent" animate={{ opacity: [1, 0.3, 1] }} transition={{ duration: 0.8, repeat: Infinity }} />}
      </div>
      {(s === 'a' || s === 'd') && (
        <div className="mb-2.5 h-1 w-full overflow-hidden rounded-full bg-black/30">
          <motion.div className={`h-full rounded-full ${s === 'd' ? 'bg-emerald-500/80' : 'bg-gradient-to-r from-cyan-600 to-brain-accent'}`}
            animate={{ width: `${Math.min(100, p * 100)}%` }} transition={{ duration: 0.08 }} />
        </div>
      )}
      {children}
    </motion.section>
  );
}
