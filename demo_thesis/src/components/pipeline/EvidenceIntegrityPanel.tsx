import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';
import { ProvenanceBadge } from './ProvenanceBadge';
import type { Provenance } from '@/lib/provenance';
import { REPLAY_PROV, DERIVED_PROV, UNKNOWN_PROV, LIVE_PROV } from '@/lib/provenance';
import { hasFmriPreview, hasClipPreview } from '@/lib/pipelineNormalize';
import { isMetricAvailable } from '@/lib/metrics';

interface EvidenceIntegrityPanelProps {
  selectedCase: DemoCase | null;
  liveMode: boolean;
  backendHealth?: { server_online?: boolean; torch_available?: boolean; model_loaded?: boolean; features_loaded?: boolean; gallery_loaded?: boolean; inference_available?: boolean; device?: string; missing?: string[]; model_meta?: { encoder_type?: string; model_type?: string; embedding_dim?: number } | null } | null;
}

interface Row {
  element: string;
  status: string;
  provenance: Provenance;
}

function rowsForCase(c: DemoCase | null, live: boolean): Row[] {
  const hasFmri = c ? hasFmriPreview(c) : false;
  const hasClip = c ? hasClipPreview(c) : false;
  const hasRecon = c ? !!(c.diffusionFinal ?? c.reconstructionImage) : false;

  return [
    { element: 'fMRI',     status: hasFmri ? 'Cached' : 'Missing',      provenance: hasFmri ? REPLAY_PROV : UNKNOWN_PROV },
    { element: 'CLIP',     status: hasClip ? 'Cached' : 'Missing',      provenance: hasClip ? REPLAY_PROV : UNKNOWN_PROV },
    { element: 'Retrieval',status: live ? 'Live' : 'Replay',            provenance: live ? LIVE_PROV : REPLAY_PROV },
    { element: 'Recon',    status: hasRecon ? 'Cached' : 'Unavailable', provenance: hasRecon ? REPLAY_PROV : UNKNOWN_PROV },
    { element: 'κ',        status: live ? 'Live' : 'Derived',           provenance: live ? LIVE_PROV : DERIVED_PROV },
  ];
}

export function EvidenceIntegrityPanel({ selectedCase, liveMode, backendHealth }: EvidenceIntegrityPanelProps) {
  const [open, setOpen] = useState(false);
  const rows = rowsForCase(selectedCase, liveMode);
  const bh = backendHealth;

  return (
    <div className="rounded-lg border border-border-subtle bg-surface-raised">
      {/* Always-visible labeled strip */}
      <button
        type="button"
        className="flex w-full items-center justify-between gap-3 px-4 py-2.5 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center gap-4 overflow-hidden">
          {selectedCase ? (
            <div className="flex items-center gap-3 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none]">
              {rows.map((r) => (
                <span key={r.element} className="inline-flex items-center gap-2 whitespace-nowrap">
                  <span className="text-[11px] font-medium text-text-muted">{r.element}</span>
                  <ProvenanceBadge provenance={r.provenance} />
                </span>
              ))}
            </div>
          ) : (
            <span className="text-[11px] text-text-muted">Select a trial to view evidence provenance</span>
          )}
        </div>
        <motion.span
          className="shrink-0 text-[10px] text-text-muted"
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.15 }}
        >
          &#9660;
        </motion.span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            <div className="border-t border-border-subtle px-4 py-3">
              <table className="w-full text-left text-[10px]">
                <thead>
                  <tr className="border-b border-border-subtle text-[9px] font-semibold uppercase tracking-wider text-text-muted">
                    <th className="pb-2 pr-3">Element</th>
                    <th className="pb-2 pr-3">Source</th>
                    <th className="pb-2 pr-3">Status</th>
                    <th className="pb-2">Provenance</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.element} className="border-b border-border-subtle/50">
                      <td className="py-1.5 pr-3 font-medium text-text-secondary">{r.element}</td>
                      <td className="py-1.5 pr-3 text-text-muted">{getSource(r.element, liveMode, selectedCase)}</td>
                      <td className="py-1.5 pr-3 text-text-muted">{r.status}</td>
                      <td className="py-1.5"><ProvenanceBadge provenance={r.provenance} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {bh && (
                <div className="mt-3 rounded-lg bg-surface-elevated px-3 py-2">
                  <p className="mb-1.5 text-[9px] font-semibold uppercase tracking-wider text-text-muted">Backend</p>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-[9px]">
                    <span className="text-text-muted">Device: <span className="font-medium text-text-secondary">{bh.device ?? 'cpu'}</span></span>
                    <span className="text-text-muted">Features: <span className={`font-medium ${bh.features_loaded ? 'text-status-success' : 'text-status-warning'}`}>{bh.features_loaded ? 'loaded' : 'missing'}</span></span>
                    <span className="text-text-muted">Gallery: <span className={`font-medium ${bh.gallery_loaded ? 'text-status-success' : 'text-status-warning'}`}>{bh.gallery_loaded ? 'loaded' : 'missing'}</span></span>
                    <span className="text-text-muted">Model: <span className={`font-medium ${bh.model_loaded ? 'text-status-success' : 'text-status-warning'}`}>{bh.model_loaded ? 'loaded' : 'missing'}</span></span>
                  </div>
                </div>
              )}

              <p className="mt-2 text-[9px] leading-relaxed text-text-muted">
                This screen replays real experiment outputs. Live backend provides encoding and retrieval. Reconstruction and per-pixel metrics remain cached.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function getSource(element: string, live: boolean, c: DemoCase | null): string {
  const sources: Record<string, string> = {
    'fMRI':      c && hasFmriPreview(c) ? 'ROI feature vector from fmri_features.npy' : 'Not exported',
    'CLIP':      c && hasClipPreview(c) ? 'CLIP ViT-L/14 image embedding' : 'Not exported',
    'Retrieval': live ? 'Live backend — CSLS gallery search' : 'Cached experiment results',
    'Recon':     'Stable Diffusion 2.1 output',
    'κ':         live ? 'Live vMF decoder kappa' : 'Derived from retrieval rank',
  };
  return sources[element] ?? '—';
}
