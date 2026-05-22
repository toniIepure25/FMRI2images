import { useState } from 'react';
import type { ReactNode } from 'react';

/**
 * EvidenceImage
 * ─────────────────────────────────────────────────────────────
 * Drop-in <img> wrapper that NEVER shows a white empty rectangle.
 * If the source is missing, fails to load, or is undefined, the
 * card paints a dark "image unavailable" surface with a small
 * frame glyph + caption + optional status chip — same aspect
 * ratio as the intended image so the layout never collapses.
 */

export type EvidenceAspect = 'square' | '4/5' | '4/3' | '3/2' | '16/9';

const ASPECT_CLASS: Record<EvidenceAspect, string> = {
  square: 'aspect-square',
  '4/5':  'aspect-[4/5]',
  '4/3':  'aspect-[4/3]',
  '3/2':  'aspect-[3/2]',
  '16/9': 'aspect-[16/9]',
};

interface EvidenceImageProps {
  src?: string | null;
  alt: string;
  aspect?: EvidenceAspect;
  /** Optional overlay rendered ON TOP of a successfully-loaded image. */
  overlay?: ReactNode;
  /** Optional caption / chip shown inside the unavailable state. */
  unavailableNote?: string;
  className?: string;
}

export function EvidenceImage({
  src,
  alt,
  aspect = '4/5',
  overlay,
  unavailableNote = 'image unavailable',
  className = '',
}: EvidenceImageProps) {
  const [errored, setErrored] = useState(false);
  const missing = !src || errored;

  return (
    <div
      className={`relative ${ASPECT_CLASS[aspect]} overflow-hidden bg-surface-raised ${className}`}
    >
      {missing ? (
        <MissingMedia note={unavailableNote} />
      ) : (
        <img
          src={src ?? undefined}
          alt={alt}
          loading="lazy"
          onError={() => setErrored(true)}
          className="h-full w-full object-cover"
        />
      )}
      {/* hair-thin inner ring keeps the photo edge crisp */}
      <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/[0.04]" aria-hidden />
      {!missing && overlay ? <div className="pointer-events-none absolute inset-0">{overlay}</div> : null}
    </div>
  );
}

/**
 * Standalone fallback surface — re-used by EvidenceImage but also
 * exposed so callers can paint a row of locked frames without an <img>.
 */
export function MissingMedia({ note = 'image unavailable' }: { note?: string }) {
  return (
    <div
      className="absolute inset-0 flex flex-col items-center justify-center gap-1.5"
      style={{
        background:
          'radial-gradient(ellipse 60% 50% at 50% 45%, rgb(255 255 255 / 0.025), transparent 75%), linear-gradient(180deg, #121419 0%, #0c0e13 100%)',
      }}
    >
      <FrameGlyph />
      <p className="font-mono text-[9.5px] uppercase tracking-[0.16em] text-text-muted/85">
        {note}
      </p>
    </div>
  );
}

function FrameGlyph() {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-text-muted/70"
      aria-hidden
    >
      <rect x="3" y="4" width="18" height="16" rx="2.2" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="M21 16l-5-5-9 9" />
    </svg>
  );
}
