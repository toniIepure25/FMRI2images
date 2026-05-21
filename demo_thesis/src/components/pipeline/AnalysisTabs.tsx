import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { InferenceResponse } from '@/lib/api';
import { WhyTrustPanel } from './WhyTrustPanel';
import { ManifoldMetricsPanel } from './ManifoldMetricsPanel';

interface AnalysisTabsProps {
  inference: InferenceResponse | null;
  liveMode: boolean;
  reconstructionAvailable: boolean;
  reconstructionMode: string;
  kappa?: number | null;
  delta?: number | null;
  children: React.ReactNode;
}

const TABS = [
  { id: 'verdict', label: 'Retrieval' },
  { id: 'manifold', label: 'Manifold' },
  { id: 'reliability', label: 'Reliability' },
  { id: 'uncertainty', label: 'Uncertainty' },
  { id: 'semantics', label: 'Semantics' },
  { id: 'reconstruction', label: 'Recon' },
] as const;

type TabId = typeof TABS[number]['id'];

export function AnalysisTabs({
  inference, liveMode, reconstructionAvailable, reconstructionMode,
  kappa, delta, children,
}: AnalysisTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>('verdict');

  return (
    <div className="space-y-4">
      {/* Tab bar */}
      <div className="flex items-center gap-0.5 rounded-lg bg-surface-raised p-0.5 overflow-x-auto">
        {TABS.map((tab) => {
          const isActive = activeTab === tab.id;
          const disabled = tab.id === 'semantics' || (tab.id === 'reconstruction' && !reconstructionAvailable);
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => !disabled && setActiveTab(tab.id)}
              disabled={disabled}
              className={`relative rounded-md px-3 py-1.5 text-[12px] font-medium whitespace-nowrap transition-colors ${
                disabled
                  ? 'text-text-muted/40 cursor-not-allowed'
                  : isActive
                    ? 'text-text-primary bg-surface-elevated shadow-[0_1px_3px_0_rgb(0,0,0,0.2)]'
                    : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {tab.label}
              {disabled && tab.id === 'semantics' && (
                <span className="ml-1 text-[9px] opacity-60">(needs cache)</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === 'verdict' && children}
          {activeTab === 'manifold' && (
            <ManifoldMetricsPanel inference={inference} />
          )}
          {activeTab === 'reliability' && (
            <WhyTrustPanel inference={inference} liveMode={liveMode} reconstructionAvailable={reconstructionAvailable} />
          )}
          {activeTab === 'uncertainty' && (
            <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5 sm:p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4">Uncertainty analysis</h3>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg bg-surface-raised px-4 py-4">
                  <p className="text-[10px] font-medium text-text-muted">κ (directional concentration)</p>
                  <p className="mt-2 font-mono text-3xl font-bold text-accent">{kappa != null ? kappa.toFixed(1) : 'N/A'}</p>
                  <p className="text-[11px] text-text-muted mt-1">
                    {kappa != null
                      ? kappa > 100 ? 'High directional certainty · narrow vMF cone' : kappa > 30 ? 'Moderate certainty' : 'Broad uncertainty · wide vMF cone'
                      : 'Kappa not available'}
                  </p>
                </div>
                <div className="rounded-lg bg-surface-raised px-4 py-4">
                  <p className="text-[10px] font-medium text-text-muted">δ (ROI disagreement)</p>
                  <p className="mt-2 font-mono text-3xl font-bold text-text-primary">{delta != null ? delta.toFixed(4) : 'N/A'}</p>
                  <p className="text-[11px] text-text-muted mt-1">
                    {delta != null ? 'Cross-ROI directional tension' : 'V62a MLP has no per-ROI δ'}
                  </p>
                </div>
              </div>
              <p className="mt-4 text-[11px] leading-relaxed text-text-muted">
                High κ indicates the predicted direction is sharply concentrated on the CLIP hypersphere — the model is confident about the embedding direction. Low κ suggests broader uncertainty, which may affect retrieval precision.
              </p>
            </div>
          )}
          {activeTab === 'semantics' && (
            <>
              {inference?.semantic_probe?.available ? (
                <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5 sm:p-6">
                  <h3 className="text-sm font-semibold text-text-primary mb-4">Decoded semantic concepts</h3>
                  <p className="text-[11px] text-text-muted mb-3">
                    Cosine similarity between predicted embedding and {inference.semantic_probe.num_concepts ?? 0} text concept embeddings.
                  </p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {(inference.semantic_probe.top_concepts_pred ?? []).map((c, i) => (
                      <span key={c.concept} className={`rounded-lg border px-3 py-1.5 text-[12px] font-medium ${
                        i === 0 ? 'border-accent/30 bg-accent/10 text-accent' : 'border-border-subtle bg-surface-raised text-text-secondary'
                      }`}>
                        {c.concept}
                        <span className="ml-1.5 font-mono text-[10px] opacity-60">{c.score.toFixed(3)}</span>
                      </span>
                    ))}
                  </div>
                  <div className="rounded-lg bg-surface-raised px-4 py-2.5 flex items-center justify-between">
                    <span className="text-[11px] text-text-muted">Concept entropy</span>
                    <span className="font-mono text-[13px] font-semibold text-text-primary">{inference.semantic_probe.concept_distribution_entropy?.toFixed(3) ?? 'N/A'}</span>
                  </div>
                </div>
              ) : (
                <div className="rounded-xl border border-border-subtle bg-surface-elevated p-6">
                  <h3 className="text-sm font-semibold text-text-primary mb-2">Decoded semantic concepts</h3>
                  <div className="flex flex-col items-center gap-3 py-10 text-center">
                    <svg className="h-10 w-10 text-border-emphasis" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <p className="text-[13px] font-medium text-text-muted">Semantic text probe</p>
                    <p className="text-[11px] text-text-muted max-w-xs">
                      {inference?.semantic_probe?.reason || 'Requires CLIP text embedding cache. Start backend with open_clip installed to enable.'}
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
          {activeTab === 'reconstruction' && (
            <div className="rounded-xl border border-border-subtle bg-surface-elevated p-5 sm:p-6">
              <h3 className="text-sm font-semibold text-text-primary mb-4">Reconstruction status</h3>
              {reconstructionAvailable ? (
                <div className="rounded-lg bg-status-success/8 px-4 py-3 ring-1 ring-status-success/15">
                  <p className="text-[12px] font-semibold text-status-success">Live reconstruction available</p>
                  <p className="text-[11px] text-text-muted mt-1">
                    Karlo UnCLIP model is loaded. Images are generated from the predicted V62a CLIP embedding during each request.
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="rounded-lg bg-surface-raised px-4 py-3">
                    <p className="text-[12px] font-semibold text-text-muted">Reconstruction unavailable</p>
                    <p className="text-[11px] text-text-muted mt-1">
                      Live reconstruction requires the Karlo UnCLIP model (kakaobrain/karlo-v1-alpha) to be cached locally.
                      Retrieval remains fully live on this machine.
                    </p>
                  </div>
                  <div className="rounded-lg bg-surface-raised px-4 py-3">
                    <p className="text-[10px] font-medium text-text-muted">How to enable</p>
                    <p className="text-[11px] text-text-muted mt-1 font-mono">
                      C2C_RECON_MODE=live C2C_RECON_ALLOW_DOWNLOAD=true
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
