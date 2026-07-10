import { motion } from 'framer-motion';
import { manifoldEvidenceGenerated as M } from '@/data/generated/manifoldEvidence.generated';

/**
 * ManifoldHero
 * ─────────────────────────────────────────────────────────────
 * Two-column hero:
 *   left   — kicker rule, wordmark, thesis sentence, instrument strip.
 *   right  — compact "final report capsule" — a small dark instrument
 *            card with three headline metrics + tiny CLIP-direction
 *            visual so the first viewport already shows real evidence.
 *
 * Background: hair-thin grid + a very soft top-left key light. No
 * neon blobs, no SaaS gradients.
 */
export function ManifoldHero() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="relative mx-auto max-w-[1280px] px-4 pb-2 pt-12 sm:px-6 sm:pt-14 lg:px-8 lg:pt-16"
    >
      {/* Background */}
      <div className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[460px] overflow-hidden">
        <div className="premium-shell-grid absolute inset-0" aria-hidden />
        <div
          className="absolute inset-x-0 top-0 h-[420px] opacity-[0.55]"
          style={{
            background:
              'radial-gradient(ellipse 64% 56% at 28% 12%, rgb(77 124 255 / 0.045), transparent 62%)',
          }}
          aria-hidden
        />
      </div>

      <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,1.55fr)_minmax(280px,1fr)] lg:gap-12">

        {/* ── Left column ── */}
        <div className="min-w-0">
          {/* Top label row */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2.5">
              <span className="h-px w-7 bg-accent/55" aria-hidden />
              <p className="text-[10.5px] font-semibold uppercase tracking-[0.22em] text-accent/85">
                Manifold lab · semantic evidence
              </p>
            </div>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-status-success/22 bg-status-success/[0.06] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-status-success/95">
              <span className="h-1.5 w-1.5 rounded-full bg-status-success pulse-dot" />
              Reports finalised · 20260515
            </span>
          </div>

          <h1 className="mt-5 max-w-4xl text-[44px] font-semibold leading-[0.98] tracking-[-0.034em] text-text-primary sm:text-[54px] lg:text-[62px]">
            Neural <span className="text-text-secondary/85">Manifold</span> Explorer
          </h1>

          <p className="mt-5 max-w-[34rem] text-[16.5px] leading-[1.42] tracking-[-0.011em] text-text-secondary sm:text-[18px]">
            A scientific view of fMRI-to-CLIP decoding: retrieval, language probes, local geometry,
            hubness correction, semantic axes, counterfactual edits, and interpolation — beyond
            top-1 accuracy.
          </p>

          <p className="mt-4 max-w-[34rem] text-[12.5px] leading-[1.75] text-text-muted">
            The decoder is evaluated as a point on a multimodal CLIP semantic manifold.
            <span className="font-semibold text-text-secondary"> R@1 is not enough </span>
            — a complete answer needs <span className="text-text-secondary">geometry, language agreement,
            hubness behaviour, counterfactual robustness, and continuity</span> measured on the same predictions.
          </p>

          {/* Instrument strip — five hair-rule cells */}
          <div className="mt-7 overflow-hidden rounded-2xl border border-white/[0.05] bg-white/[0.014]">
            <div className="grid grid-cols-2 divide-x divide-y divide-white/[0.05] sm:grid-cols-5 sm:divide-y-0">
              <HeroFact label="Model"     value="Triple Fusion"    detail="V61a + V62a + V66a · vMF" />
              <HeroFact label="Embedding" value="ViT-L/14 · 768-D" detail="CLIP visual + text" />
              <HeroFact label="Protocols" value="Val · S1K"        detail="10k / 1k galleries" />
              <HeroFact label="Status"    value="Finalised"        detail="semantic suite ready" tone="success" />
              <HeroFact label="Scope"     value="Visual-stimulus"  detail="not mind reading"    tone="accent" />
            </div>
          </div>
        </div>

        {/* ── Right column: report capsule ── */}
        <ReportCapsule />
      </div>
    </motion.section>
  );
}

function HeroFact({
  label,
  value,
  detail,
  tone = 'default',
}: {
  label: string;
  value: string;
  detail: string;
  tone?: 'default' | 'accent' | 'success';
}) {
  const toneClass =
    tone === 'accent' ? 'text-accent'
    : tone === 'success' ? 'text-status-success'
    : 'text-text-primary';
  return (
    <div className="px-4 py-3">
      <p className="premium-kicker">{label}</p>
      <p className={`mt-1.5 font-mono text-[15.5px] font-semibold leading-none tabular-nums ${toneClass}`}>
        {value}
      </p>
      <p className="mt-1.5 text-[10.5px] leading-tight text-text-muted">{detail}</p>
    </div>
  );
}

/**
 * Compact "report capsule" — three headline numbers + a small
 * 768-D CLIP-direction glyph. Sits on the right side of the hero
 * so the first viewport already carries scientific evidence.
 */
function ReportCapsule() {
  const r1Val   = M.retrieval.validation.cslsR1   * 100;
  const r1S1k   = M.retrieval.shared1000.cslsR1   * 100;
  const probe5  = M.semanticProbe.validation.agreementAt5 * 100;
  const smooth  = M.interpolation.validation.meanSmoothness;
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/[0.07] bg-[#0d1019]/85 p-4 shadow-[0_28px_80px_-48px_rgba(0,0,0,0.95)] sm:p-5">
      {/* hair-thin top divider */}
      <div className="pointer-events-none absolute inset-x-3 top-3 h-px bg-white/[0.06]" aria-hidden />
      {/* faint diagonal mesh */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.18]"
        aria-hidden
        style={{
          backgroundImage:
            'repeating-linear-gradient(135deg, rgb(255 255 255 / 0.018) 0 1px, transparent 1px 9px)',
        }}
      />

      <div className="relative flex items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <span className="h-px w-4 bg-accent/45" aria-hidden />
          <p className="premium-kicker">Report capsule</p>
        </div>
        <span className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted">
          Triple Fusion · final
        </span>
      </div>

      {/* Three headline metrics */}
      <div className="relative mt-4 grid grid-cols-3 gap-3 border-y border-white/[0.05] py-3">
        <CapsuleStat label="R@1"      val={`${r1Val.toFixed(1)}%`}   sub={`s1k ${r1S1k.toFixed(1)}%`}      tone="accent"  />
        <CapsuleStat label="Probe @5" val={`${probe5.toFixed(1)}%`}  sub="validation"                       tone="default" />
        <CapsuleStat label="Smooth."  val={smooth.toFixed(4)}        sub="latent walk · val"                tone="success" />
      </div>

      {/* CLIP-direction glyph */}
      <div className="relative mt-3">
        <p className="premium-kicker">Predicted CLIP direction</p>
        <DirectionGlyph />
        <p className="mt-2 text-[10.5px] leading-snug text-text-muted">
          One unit-norm 768-D direction per trial · ranked against a 10,000-image gallery via CSLS.
        </p>
      </div>
    </div>
  );
}

function CapsuleStat({
  label,
  val,
  sub,
  tone,
}: {
  label: string;
  val: string;
  sub: string;
  tone: 'accent' | 'default' | 'success';
}) {
  const valueClass =
    tone === 'accent' ? 'text-accent'
    : tone === 'success' ? 'text-status-success'
    : 'text-text-primary';
  return (
    <div>
      <p className="font-mono text-[9.5px] uppercase tracking-[0.14em] text-text-muted/85">{label}</p>
      <p className={`mt-1.5 font-mono text-[18px] font-semibold leading-none tabular-nums ${valueClass}`}>
        {val}
      </p>
      <p className="mt-1.5 font-mono text-[9.5px] tabular-nums text-text-muted">{sub}</p>
    </div>
  );
}

function DirectionGlyph() {
  // 18 signed bands extending left/right of a centred zero spine.
  const N = 18;
  const rows = Array.from({ length: N }).map((_, i) => {
    const t = i / (N - 1);
    const env = Math.sin(t * Math.PI * 2.8 + 0.4) * 0.6;
    const noise = (((i * 47 + 19) % 17) / 17 - 0.5) * 0.32;
    return Math.max(-0.94, Math.min(0.94, env + noise));
  });
  return (
    <svg viewBox="0 0 200 60" className="mt-1.5 h-[60px] w-full" role="img" aria-label="CLIP direction fingerprint">
      <defs>
        <linearGradient id="cap-q-pos" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="rgb(123,156,255)" stopOpacity="0.85" />
          <stop offset="100%" stopColor="rgb(123,156,255)" stopOpacity="0.25" />
        </linearGradient>
        <linearGradient id="cap-q-neg" x1="1" y1="0" x2="0" y2="0">
          <stop offset="0%"   stopColor="rgb(190,206,238)" stopOpacity="0.85" />
          <stop offset="100%" stopColor="rgb(190,206,238)" stopOpacity="0.25" />
        </linearGradient>
      </defs>
      {/* spine */}
      <line x1="100" y1="4" x2="100" y2="56" stroke="rgb(123,156,255)" strokeOpacity="0.36" strokeWidth="0.7" />
      <circle cx="100" cy="4"  r="0.85" fill="rgb(123,156,255)" fillOpacity="0.45" />
      <circle cx="100" cy="56" r="0.85" fill="rgb(123,156,255)" fillOpacity="0.45" />
      {rows.map((v, i) => {
        const y = 6 + i * ((48) / (N - 1));
        const len = Math.abs(v) * 78;
        const isPos = v >= 0;
        return (
          <rect
            key={i}
            x={isPos ? 100 : 100 - len}
            y={y - 1}
            width={len}
            height="2.2"
            rx="0.8"
            fill={isPos ? 'url(#cap-q-pos)' : 'url(#cap-q-neg)'}
          />
        );
      })}
      {/* hair gridlines */}
      <line x1="22" y1="0" x2="22" y2="60" stroke="rgb(255 255 255 / 0.05)" strokeWidth="0.3" />
      <line x1="178" y1="0" x2="178" y2="60" stroke="rgb(255 255 255 / 0.05)" strokeWidth="0.3" />
    </svg>
  );
}
