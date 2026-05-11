import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { ProvenanceBadge } from './ProvenanceBadge';
import type { Provenance } from '@/lib/provenance';
import { REPLAY_PROV, DERIVED_PROV, UNKNOWN_PROV, LIVE_PROV } from '@/lib/provenance';
import {
  hasFmriPreview,
  hasClipPreview,
} from '@/lib/pipelineNormalize';
import { isMetricAvailable } from '@/lib/metrics';

interface EvidenceIntegrityPanelProps {
  selectedCase: DemoCase | null;
  liveMode: boolean;
  backendHealth?: { server_online?: boolean; torch_available?: boolean; model_loaded?: boolean; features_loaded?: boolean; gallery_loaded?: boolean; inference_available?: boolean; device?: string; missing?: string[]; model_meta?: { encoder_type?: string; model_type?: string; embedding_dim?: number } | null } | null;
}

interface Row {
  element: string;
  source: string;
  status: string;
  provenance: Provenance;
}

function rowsForCase(c: DemoCase | null, live: boolean): Row[] {
  const hasFmri = c ? hasFmriPreview(c) : false;
  const hasClip = c ? hasClipPreview(c) : false;
  const hasRecon = c ? !!(c.diffusionFinal ?? c.reconstructionImage) : false;
  const hasPixcorr = c ? isMetricAvailable(c.metrics.pixcorr) : false;
  const hasSsim = c ? isMetricAvailable(c.metrics.ssim) : false;

  return [
    {
      element: 'fMRI preview',
      source: 'ROI feature vector from fmri_features.npy',
      status: hasFmri ? 'Available' : 'Not exported',
      provenance: hasFmri ? REPLAY_PROV : UNKNOWN_PROV,
    },
    {
      element: 'Reference CLIP',
      source: 'CLIP ViT-L/14 image embedding from clip.parquet',
      status: hasClip ? 'Available' : 'Not exported',
      provenance: hasClip ? REPLAY_PROV : UNKNOWN_PROV,
    },
    {
      element: 'Predicted CLIP',
      source: 'vMF decoder output',
      status: live ? 'Live from backend' : 'Not exported in replay',
      provenance: live ? LIVE_PROV : UNKNOWN_PROV,
    },
    {
      element: 'Retrieval ranking',
      source: 'CSLS gallery search',
      status: live ? 'Live inference' : 'Cached ranking',
      provenance: live ? LIVE_PROV : REPLAY_PROV,
    },
    {
      element: 'Reconstruction',
      source: 'SD 2.1 diffusion output',
      status: hasRecon ? 'Cached qualitative' : 'Not available',
      provenance: hasRecon ? REPLAY_PROV : UNKNOWN_PROV,
    },
    {
      element: 'PixCorr / SSIM',
      source: 'Measured from reconstruction',
      status: hasPixcorr || hasSsim ? 'Measured' : 'Unavailable',
      provenance: hasPixcorr || hasSsim ? REPLAY_PROV : UNKNOWN_PROV,
    },
    {
      element: 'κ (kappa)',
      source: live ? 'Live vMF decoder output' : 'Derived from retrieval rank',
      status: live ? 'Live' : 'Derived estimate',
      provenance: live ? LIVE_PROV : DERIVED_PROV,
    },
    {
      element: 'δ (delta)',
      source: live ? 'V62a MLP does not produce δ' : 'Derived from retrieval rank',
      status: live ? 'Unavailable' : 'Derived estimate',
      provenance: live ? UNKNOWN_PROV : DERIVED_PROV,
    },
    {
      element: 'DUA-CFG policy',
      source: 'Computed from uncertainty estimate',
      status: 'Derived',
      provenance: DERIVED_PROV,
    },
  ];
}

export function EvidenceIntegrityPanel({ selectedCase, liveMode, backendHealth }: EvidenceIntegrityPanelProps) {
  const [open, setOpen] = useState(false);
  const rows = rowsForCase(selectedCase, liveMode);
  const bh = backendHealth;

  return (
    <div className="rounded-2xl border border-white/[0.05] bg-[#0a0f1e]/60">
      <button
        type="button"
        className="flex w-full items-center justify-between px-5 py-3 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-2.5">
          <svg className="h-3.5 w-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span className="text-[11px] font-semibold tracking-wide text-slate-300">
            Evidence integrity
          </span>
          {!selectedCase && (
            <span className="text-[10px] text-slate-600">(select a trial for per-case detail)</span>
          )}
        </div>
        <motion.span
          className="text-[10px] text-slate-600"
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.15 }}
        >
          ▼
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="border-t border-white/[0.04] px-5 py-4">
              <table className="w-full text-left text-[10px]">
                <thead>
                  <tr className="border-b border-white/[0.04] text-[9px] font-semibold uppercase tracking-wider text-slate-600">
                    <th className="pb-2 pr-3">Element</th>
                    <th className="pb-2 pr-3">Source</th>
                    <th className="pb-2 pr-3">Status</th>
                    <th className="pb-2">Provenance</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.element} className="border-b border-white/[0.02]">
                      <td className="py-1.5 pr-3 font-medium text-slate-300">{r.element}</td>
                      <td className="py-1.5 pr-3 text-slate-500">{r.source}</td>
                      <td className="py-1.5 pr-3 text-slate-400">{r.status}</td>
                      <td className="py-1.5">
                        <ProvenanceBadge provenance={r.provenance} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {bh && (
                <div className="mt-3 rounded-lg bg-white/[0.02] px-3 py-2">
                  <p className="mb-1.5 text-[9px] font-semibold uppercase tracking-wider text-slate-500">Backend status</p>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-[9px]">
                    <span className="text-slate-500">Server: <span className="font-medium text-emerald-400">online</span></span>
                    <span className="text-slate-500">Torch: <span className={`font-medium ${bh.torch_available ? 'text-emerald-400' : 'text-amber-400'}`}>{bh.torch_available ? 'yes' : 'no'}</span></span>
                    <span className="text-slate-500">Device: <span className="font-medium text-slate-300">{bh.device ?? 'cpu'}</span></span>
                    <span className="text-slate-500">Features: <span className={`font-medium ${bh.features_loaded ? 'text-emerald-400' : 'text-amber-400'}`}>{bh.features_loaded ? 'loaded' : 'missing'}</span></span>
                    <span className="text-slate-500">Gallery: <span className={`font-medium ${bh.gallery_loaded ? 'text-emerald-400' : 'text-amber-400'}`}>{bh.gallery_loaded ? 'loaded' : 'missing'}</span></span>
                    <span className="text-slate-500">Model: <span className={`font-medium ${bh.model_loaded ? 'text-emerald-400' : 'text-amber-400'}`}>{bh.model_loaded ? 'loaded' : 'missing'}</span></span>
                    <span className="text-slate-500">Inference: <span className={`font-medium ${bh.inference_available ? 'text-emerald-400' : 'text-slate-600'}`}>{bh.inference_available ? 'available' : 'unavailable'}</span></span>
                    {bh.model_meta && (
                      <span className="text-slate-500">Encoder: <span className="font-medium text-slate-300">{bh.model_meta.encoder_type === 'mlp' ? 'MLP' : bh.model_meta.encoder_type ?? '?'} + {bh.model_meta.model_type ?? '?'}</span></span>
                    )}
                  </div>
                  {bh.missing && bh.missing.length > 0 && (
                    <p className="mt-1 text-[9px] text-amber-400/70">Missing: {bh.missing.join(', ')}</p>
                  )}
                </div>
              )}

              <p className="mt-3 text-[9px] leading-relaxed text-slate-600">
                For presentation reliability, this screen replays real experiment outputs.
                When the backend is connected, encoding and retrieval execute through the live model path.
                Reconstruction remains cached unless a diffusion backend is explicitly connected.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
