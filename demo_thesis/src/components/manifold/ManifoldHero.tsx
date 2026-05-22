import { motion } from 'framer-motion';

/**
 * ManifoldHero
 * ─────────────────────────────────────────────────────────────
 * Calm scientific opener — no neon, no glow, no gradient blobs.
 * The header reads as the title page of a finalised report:
 *   kicker rule → wordmark → lead → instrument-strip footer.
 *
 * A very faint coordinate grid sits under the title block and
 * fades out toward the lead paragraph so the page feels like an
 * instrument panel without competing with the copy.
 */
export function ManifoldHero() {
  return (
    <motion.section
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="relative mx-auto max-w-[1280px] px-4 pb-2 pt-12 sm:px-6 sm:pt-14 lg:px-8 lg:pt-16"
    >
      {/* ── Background: hair-thin coordinate grid + soft top-left key light. */}
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

      {/* Top label row: kicker + status pill */}
      <div className="flex flex-wrap items-center justify-between gap-3">
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

      <h1 className="mt-6 max-w-4xl text-[44px] font-semibold leading-[0.98] tracking-[-0.034em] text-text-primary sm:text-[58px] lg:text-[68px]">
        Neural <span className="text-text-secondary/85">Manifold</span> Explorer
      </h1>

      <p className="mt-5 max-w-[44rem] text-[17.5px] leading-[1.42] tracking-[-0.011em] text-text-secondary sm:text-[19.5px]">
        A scientific view of fMRI-to-CLIP decoding: retrieval, language probes, local geometry,
        hubness correction, semantic axes, counterfactual edits, and interpolation — beyond
        top-1 accuracy.
      </p>

      <p className="mt-4 max-w-[44rem] text-[12.5px] leading-[1.75] text-text-muted">
        The decoder is evaluated as a point on a multimodal CLIP semantic manifold. R@1 is not enough —
        a complete answer needs <span className="text-text-secondary">geometry, language agreement,
        hubness behaviour, counterfactual robustness, and continuity</span> measured on the same predictions.
      </p>

      {/* ── Instrument strip — five hair-rule cells ─────────────── */}
      <div className="mt-9 overflow-hidden rounded-2xl border border-white/[0.05] bg-white/[0.014]">
        <div className="grid grid-cols-2 divide-x divide-y divide-white/[0.05] sm:grid-cols-5 sm:divide-y-0">
          <HeroFact label="Model"     value="V62a"             detail="residual MLP + vMF" />
          <HeroFact label="Embedding" value="ViT-L/14 · 768-D" detail="CLIP visual + text" />
          <HeroFact label="Protocols" value="Val · S1K"        detail="10k / 1k galleries" />
          <HeroFact label="Status"    value="Finalised"        detail="semantic suite ready" tone="success" />
          <HeroFact label="Scope"     value="Visual-stimulus"  detail="not mind reading"    tone="accent" />
        </div>
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
