import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ProvenanceBadge } from '@/components/pipeline/ProvenanceBadge';
import { PremiumPanel } from '@/components/premium/PremiumPanel';
import { ManifoldSectionHeader } from './ManifoldSectionHeader';
import { KickerRule, MonoNote, LatentOnlyChip } from './atoms';
import {
  counterfactualAxes,
  counterfactualAxisStats,
} from '@/data/neuralManifoldExplorer';
import { DERIVED_PROV, REPLAY_PROV } from '@/lib/provenance';

/**
 * CounterfactualSemanticLab
 * ─────────────────────────────────────────────────────────────
 * Pick an interpretable axis; the right panel updates with the
 * resulting concept, top probes, and the real validation /
 * SHARED1000 robustness margin.
 *
 * Visual upgrades:
 *   • Axis selector cards have an active rail, a hover lift, and
 *     a left-↔-right direction glyph that points to the active end.
 *   • Right panel adds a "semantic shift" rail visualising the
 *     direction strength qualitatively, plus a "latent-only" chip.
 */
export function CounterfactualSemanticLab() {
  const [activeId, setActiveId] = useState(counterfactualAxes[0].id);
  const active = counterfactualAxes.find((a) => a.id === activeId) ?? counterfactualAxes[0];

  return (
    <motion.section
      initial={{ opacity: 0, y: 12 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="mx-auto max-w-[1280px] px-4 sm:px-6 lg:px-8"
    >
      <ManifoldSectionHeader
        kicker="Counterfactual lab"
        title="Counterfactual explanations in neural semantic space."
        description="Editing the decoded CLIP vector along an interpretable axis is a latent-space explanation — it shows what the model considers a small step toward another concept. It does not manipulate brain activity."
        action={
          <div className="flex items-center gap-2">
            <LatentOnlyChip />
            <ProvenanceBadge provenance={DERIVED_PROV} />
          </div>
        }
      />

      <PremiumPanel className="mt-6 p-5 sm:p-6">
        <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.55fr)]">

          {/* ── Axis selector ── */}
          <div className="flex flex-col gap-3">
            <KickerRule label="Semantic axes" />
            <div className="grid gap-2">
              {counterfactualAxes.map((a) => {
                const isActive = a.id === activeId;
                return (
                  <button
                    type="button"
                    key={a.id}
                    onClick={() => setActiveId(a.id)}
                    className={`group relative overflow-hidden rounded-xl border px-3 py-2.5 text-left transition ${
                      isActive
                        ? 'border-accent/30 bg-accent/[0.07] ring-1 ring-accent/15'
                        : 'border-white/[0.05] bg-white/[0.014] hover:border-white/[0.085] hover:bg-white/[0.03]'
                    }`}
                  >
                    {/* Active accent rail */}
                    {isActive ? (
                      <span className="absolute inset-y-2 left-0 w-[2px] rounded-full bg-accent/80" aria-hidden />
                    ) : null}
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-baseline gap-2.5">
                        <span
                          className={`font-mono text-[11px] tabular-nums ${
                            isActive
                              ? a.direction === -1 ? 'text-accent' : 'text-text-secondary'
                              : 'text-text-muted'
                          }`}
                        >
                          {a.leftLabel}
                        </span>
                        <AxisDirectionGlyph direction={a.direction} active={isActive} />
                        <span
                          className={`font-mono text-[11px] tabular-nums ${
                            isActive
                              ? a.direction === 1 ? 'text-accent' : 'text-text-secondary'
                              : 'text-text-muted'
                          }`}
                        >
                          {a.rightLabel}
                        </span>
                      </div>
                      <span
                        className={`font-mono text-[10px] uppercase tracking-[0.14em] ${
                          isActive ? 'text-accent/90' : 'text-text-muted/85'
                        }`}
                      >
                        {a.axisKey.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Margin stat strip (real aggregates) */}
            <div className="mt-1 rounded-xl border border-white/[0.05] bg-white/[0.015] p-3.5">
              <div className="flex items-center justify-between">
                <p className="premium-kicker">Robustness margin</p>
                <ProvenanceBadge provenance={REPLAY_PROV} />
              </div>
              <div className="mt-2 grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted/85">
                    Validation
                  </p>
                  <p className="mt-1 font-mono text-[19px] font-semibold leading-none tabular-nums text-text-primary">
                    {active.marginValidation.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted/85">
                    SHARED1000
                  </p>
                  <p className="mt-1 font-mono text-[19px] font-semibold leading-none tabular-nums text-text-primary">
                    {active.marginShared1000.toFixed(3)}
                  </p>
                </div>
              </div>
              <p className="mt-2.5 font-mono text-[10.5px] tabular-nums text-text-muted">
                Strongest axis · {counterfactualAxisStats.bestAxis.replace(/_/g, ' ')} · ρ {counterfactualAxisStats.meanSpearmanValidation.toFixed(3)} val
              </p>
            </div>
          </div>

          {/* ── Active concept readout ── */}
          <div className="workbench-inset flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <KickerRule label="After edit · z_pred + α · axis" />
              <MonoNote>{active.leftLabel} ↔ {active.rightLabel}</MonoNote>
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={active.id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
                className="flex flex-col gap-4"
              >
                {/* Concept headline */}
                <div className="flex items-baseline justify-between gap-3">
                  <div>
                    <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-text-muted">
                      Resulting concept
                    </p>
                    <p className="mt-1 text-[22px] font-semibold tracking-tight text-text-primary">
                      {active.resultingConcept}
                    </p>
                    <p className="mt-0.5 font-mono text-[11px] tabular-nums text-accent/90">
                      {active.resultingCategory}
                    </p>
                  </div>

                  {/* Direction rail */}
                  <div className="hidden sm:flex w-[180px] flex-col gap-1.5">
                    <p className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted">
                      Semantic shift
                    </p>
                    <ShiftRail direction={active.direction} />
                    <div className="flex justify-between font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted/85">
                      <span>{active.leftLabel}</span>
                      <span>{active.rightLabel}</span>
                    </div>
                  </div>
                </div>

                {/* Top probes */}
                <div className="rounded-xl border border-white/[0.05] bg-white/[0.015] p-3.5">
                  <p className="premium-kicker">Top probes</p>
                  <ul className="mt-2.5 space-y-1.5">
                    {active.resultingProbes.map((p, i) => (
                      <li key={p.concept} className="flex items-center gap-3">
                        <span
                          className={`min-w-[96px] text-[11.5px] tracking-tight ${
                            i === 0 ? 'font-semibold text-text-primary' : 'text-text-secondary'
                          }`}
                        >
                          {p.concept}
                        </span>
                        <div className="relative h-[4px] flex-1 overflow-hidden rounded-full bg-white/[0.04]">
                          <div
                            className={`absolute inset-y-0 left-0 rounded-full ${i === 0 ? 'bg-accent/85' : 'bg-text-secondary/55'}`}
                            style={{ width: `${(p.score * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span
                          className={`w-10 text-right font-mono text-[11px] tabular-nums ${
                            i === 0 ? 'text-text-primary' : 'text-text-muted'
                          }`}
                        >
                          {p.score.toFixed(2)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>

                <p className="text-[11.5px] leading-relaxed text-text-secondary">
                  {active.explanation}
                </p>
              </motion.div>
            </AnimatePresence>

            <p className="mt-1 border-t border-white/[0.05] pt-2.5 text-[10.5px] leading-snug text-text-muted">
              Counterfactual edits are latent-space explanations, not brain manipulation. Per-axis probe
              ranking shown is DERIVED demo data illustrating the aggregate margin metric.
            </p>
          </div>
        </div>
      </PremiumPanel>
    </motion.section>
  );
}

function AxisDirectionGlyph({ direction, active }: { direction: -1 | 1; active: boolean }) {
  const color = active ? 'rgb(123,156,255)' : 'rgb(150,158,172)';
  const opacity = active ? 0.9 : 0.55;
  return (
    <svg width="22" height="8" viewBox="0 0 22 8" fill="none" aria-hidden>
      {direction === 1 ? (
        <path d="M1 4h17M14 1l4 3-4 3" stroke={color} strokeOpacity={opacity} strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d="M21 4H4M8 1L4 4l4 3" stroke={color} strokeOpacity={opacity} strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />
      )}
    </svg>
  );
}

function ShiftRail({ direction }: { direction: -1 | 1 }) {
  return (
    <svg viewBox="0 0 180 14" className="h-[14px] w-full" aria-hidden>
      {/* Track */}
      <line x1="6" y1="7" x2="174" y2="7" stroke="rgb(255 255 255 / 0.08)" strokeWidth="1.2" strokeLinecap="round" />
      {/* Origin marker (center) */}
      <line x1="90" y1="2" x2="90" y2="12" stroke="rgb(255 255 255 / 0.20)" strokeWidth="0.6" />
      {/* Direction segment */}
      <line
        x1="90"
        y1="7"
        x2={direction === 1 ? 168 : 12}
        y2="7"
        stroke="rgb(123,156,255)"
        strokeOpacity="0.85"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      {/* Tip */}
      <circle cx={direction === 1 ? 168 : 12} cy="7" r="2.2" fill="rgb(123,156,255)" />
    </svg>
  );
}
