import type { ReactNode } from 'react';

/**
 * Small atomic primitives shared by every Manifold section.
 * Keeping them isolated keeps the page panels visually consistent
 * (kicker grammar, lock chip, latent-only chip, hair-rule, etc.)
 * and avoids one-off styles being scattered across components.
 */

/* ─── Kicker rule ─────────────────────────────────────────── */

export function KickerRule({ label, className = '' }: { label: string; className?: string }) {
  return (
    <div className={`flex items-center gap-2.5 ${className}`}>
      <span className="h-px w-5 bg-accent/40" aria-hidden />
      <p className="premium-kicker">{label}</p>
    </div>
  );
}

/* ─── Mono note (right-aligned scientific annotation) ────── */

export function MonoNote({ children }: { children: ReactNode }) {
  return (
    <span className="font-mono text-[10px] uppercase tracking-[0.14em] text-text-muted">
      {children}
    </span>
  );
}

/* ─── Status chips ───────────────────────────────────────── */

export function LockChip({ children = 'Locked' }: { children?: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-white/[0.06] bg-white/[0.025] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-muted">
      <LockGlyph />
      {children}
    </span>
  );
}

export function LatentOnlyChip() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/22 bg-accent/[0.07] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-accent/90">
      <span className="h-1.5 w-1.5 rounded-full bg-accent/85" />
      Latent-space only
    </span>
  );
}

export function SchematicChip() {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-white/[0.07] bg-white/[0.025] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
      <span className="h-1.5 w-1.5 rounded-full bg-text-secondary/75" />
      Schematic
    </span>
  );
}

function LockGlyph() {
  return (
    <svg
      width="9"
      height="11"
      viewBox="0 0 12 14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <rect x="2" y="6.5" width="8" height="6.5" rx="1.4" />
      <path d="M3.5 6.5V4.5a2.5 2.5 0 015 0v2" />
    </svg>
  );
}

/* ─── Hair-rule horizontal divider ───────────────────────── */

export function HairRule({ className = '' }: { className?: string }) {
  return <div className={`h-px w-full bg-white/[0.05] ${className}`} aria-hidden />;
}
