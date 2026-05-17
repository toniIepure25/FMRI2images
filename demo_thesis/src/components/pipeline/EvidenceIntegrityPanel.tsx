import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { DemoCase } from '@/types';

const COLOR_MAP: Record<string, string> = {
  LIVE_LOCAL: 'bg-accent/10 text-accent ring-1 ring-accent/20',
  CACHED_LOCAL: 'bg-surface-active text-text-secondary ring-1 ring-border-subtle',
  DERIVED_LOCAL: 'bg-status-info/8 text-status-info ring-1 ring-status-info/15',
  UNAVAILABLE: 'bg-surface-raised text-text-muted ring-1 ring-border-subtle',
};

const DOT_MAP: Record<string, string> = {
  LIVE_LOCAL: 'bg-accent',
  CACHED_LOCAL: 'bg-text-muted',
  DERIVED_LOCAL: 'bg-status-info',
  UNAVAILABLE: 'bg-border-emphasis',
};

interface EvidenceRow {
  label: string;
  provenance: 'LIVE_LOCAL' | 'CACHED_LOCAL' | 'DERIVED_LOCAL' | 'UNAVAILABLE';
  detail: string;
}

export interface EvidenceIntegrityPanelProps {
  selectedCase: DemoCase | null;
  liveMode: boolean;
  backendHealth?: {
    live_retrieval_available?: boolean; v62_model_loaded?: boolean;
    features_loaded?: boolean; gallery_768_loaded?: boolean;
    gallery_768_size?: number; gallery_768_dim?: number;
    reconstruction_mode?: string; reconstruction_available?: boolean;
    device?: string; effective_mode?: string;
  } | null;
}

function rowsForState(c: DemoCase | null, h: EvidenceIntegrityPanelProps['backendHealth'] | null): EvidenceRow[] {
  const fmriOk = !!(c?.fmriPreview?.length);
  const reconAvail = h?.reconstruction_mode === 'live_local_reconstruction';
  const reconPending = h?.reconstruction_mode === 'live_pending_download';
  const liveRetrieval = h?.live_retrieval_available === true;
  const v62Loaded = h?.v62_model_loaded === true;
  const galleryLoaded = h?.gallery_768_loaded === true;

  return [
    { label: 'fMRI',   provenance: fmriOk ? 'CACHED_LOCAL' : 'UNAVAILABLE', detail: 'ROI-masked beta vector' },
    { label: 'Model',  provenance: v62Loaded ? 'LIVE_LOCAL' : 'UNAVAILABLE', detail: v62Loaded ? 'V62a MLP · CUDA' : 'Not loaded' },
    { label: 'Gallery',provenance: galleryLoaded ? 'CACHED_LOCAL' : 'UNAVAILABLE', detail: galleryLoaded ? `${h?.gallery_768_size?.toLocaleString() ?? '?'} × ${h?.gallery_768_dim ?? '?'}D` : 'Not loaded' },
    { label: 'Retrieval', provenance: liveRetrieval ? 'LIVE_LOCAL' : 'DERIVED_LOCAL', detail: liveRetrieval ? 'Live CSLS' : 'Replay' },
    { label: 'Recon',  provenance: reconAvail ? 'LIVE_LOCAL' : reconPending ? 'CACHED_LOCAL' : 'UNAVAILABLE',
      detail: reconAvail ? 'Karlo UnCLIP' : reconPending ? 'Pending download' : 'Not available' },
  ];
}

export function EvidenceIntegrityPanel({ selectedCase, liveMode, backendHealth }: EvidenceIntegrityPanelProps) {
  const [open, setOpen] = useState(false);
  const rows = rowsForState(selectedCase, backendHealth);

  return (
    <div className="rounded-2xl border border-border-subtle bg-surface-raised/80 shadow-surface">
      <button
        type="button"
        className="flex w-full items-center justify-between gap-4 px-5 py-3 text-left"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex min-w-0 items-center gap-4 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none]">
          <span className="premium-kicker shrink-0">Evidence integrity</span>
          {selectedCase ? (
            rows.map((r) => (
              <span key={r.label} className="inline-flex items-center gap-2 whitespace-nowrap rounded-xl border border-border-subtle bg-surface-elevated/70 px-2.5 py-1">
                <span className="text-[11px] font-semibold text-text-secondary">{r.label}</span>
                <span className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-[10px] font-medium ${COLOR_MAP[r.provenance]}`}>
                  <span className={`h-1 w-1 rounded-full ${DOT_MAP[r.provenance]}`} />
                  {r.provenance === 'LIVE_LOCAL' ? 'live' : r.provenance === 'CACHED_LOCAL' ? 'cached' : r.provenance === 'DERIVED_LOCAL' ? 'derived' : 'unavailable'}
                </span>
              </span>
            ))
          ) : (
            <span className="text-[11px] text-text-muted">Select a trial</span>
          )}
        </div>
        <motion.span className="shrink-0 text-[10px] text-text-muted"
          animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.15 }}>&#9660;</motion.span>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }} className="overflow-hidden">
            <div className="border-t border-border-subtle px-5 py-4 space-y-2">
              {rows.map((r) => (
                <div key={r.label} className="flex items-center gap-3 rounded-xl bg-surface-base/50 px-3 py-2 text-[11px]">
                  <span className="font-semibold w-16 text-text-secondary">{r.label}</span>
                  <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${COLOR_MAP[r.provenance]}`}>
                    {r.provenance.replace(/_/g, ' ')}
                  </span>
                  <span className="text-text-muted">{r.detail}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
