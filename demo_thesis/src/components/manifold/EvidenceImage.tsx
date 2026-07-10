import { useState } from 'react';
import type { ReactNode } from 'react';

/**
 * EvidenceImage
 * ─────────────────────────────────────────────────────────────
 * Image card that CAN NEVER show a white block. Strategy:
 *   1. A dark "MissingMedia" surface is rendered as the wrapper's
 *      background, ALWAYS — before/during load, on error, and even
 *      if the image is a transparent or pure-white asset.
 *   2. The <img> is layered on top, starting at opacity 0 and
 *      fading to 1 only on successful onLoad. If onError fires
 *      (or no src given), the img stays hidden and the dark
 *      surface remains visible.
 *   3. The wrapper preserves the requested aspect ratio so layout
 *      never collapses — whether the asset succeeds, fails, or is
 *      still in flight.
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
  /** Optional overlay rendered on top of the (resolved) image. */
  overlay?: ReactNode;
  /** Optional caption rendered inside the unavailable state. */
  unavailableNote?: string;
  className?: string;
}

export function EvidenceImage({
  src,
  alt,
  aspect = '4/5',
  overlay,
  unavailableNote = '',
  className = '',
}: EvidenceImageProps) {
  const [loaded, setLoaded] = useState(false);
  const [errored, setErrored] = useState(false);
  const hasSrc = !!src;
  const showImage = hasSrc && !errored;

  return (
    <div
      className={`relative ${ASPECT_CLASS[aspect]} overflow-hidden ${className}`}
      style={{
        // Hardcoded dark backplate — guarantees no white frame can ever
        // paint, even before MissingMedia mounts or during img decode.
        backgroundColor: '#0c0e13',
      }}
    >
      {/* Dark surface — always present, behind everything. */}
      <MissingMedia note={errored || !hasSrc ? unavailableNote : ''} />

      {/* Image — fades in only after successful load. */}
      {showImage ? (
        <img
          src={src ?? undefined}
          alt={alt}
          loading="lazy"
          decoding="async"
          onLoad={() => setLoaded(true)}
          onError={() => {
            setErrored(true);
            setLoaded(false);
          }}
          className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${
            loaded ? 'opacity-100' : 'opacity-0'
          }`}
        />
      ) : null}

      {/* Hair-thin inner ring keeps the photo edge crisp. */}
      <div className="pointer-events-none absolute inset-0 ring-1 ring-inset ring-white/[0.05]" aria-hidden />

      {/* Overlay (badges/chips) only when the image is visible. */}
      {showImage && loaded && overlay ? (
        <div className="pointer-events-none absolute inset-0">{overlay}</div>
      ) : null}
    </div>
  );
}

/**
 * Dark scientific placeholder surface. Used by EvidenceImage and
 * available standalone for callers that want to render a row of
 * locked frames without a live <img>.
 *
 * `note === ''` renders a silent surface (no caption) — useful as
 * the loading state behind an <img>. `note !== ''` renders the
 * glyph + caption "image unavailable" treatment.
 */
export function MissingMedia({ note = '' }: { note?: string }) {
  return (
    <div
      className="absolute inset-0 flex flex-col items-center justify-center gap-1.5"
      style={{
        background:
          'radial-gradient(ellipse 60% 50% at 50% 45%, rgb(255 255 255 / 0.022), transparent 75%), linear-gradient(180deg, #121419 0%, #0c0e13 100%)',
      }}
    >
      {/* Hair-thin diagonal noise so the surface never looks completely flat. */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.22]"
        aria-hidden
        style={{
          backgroundImage:
            'repeating-linear-gradient(135deg, rgb(255 255 255 / 0.015) 0 1px, transparent 1px 8px)',
        }}
      />
      {note ? (
        <>
          <FrameGlyph />
          <p className="font-mono text-[9.5px] uppercase tracking-[0.16em] text-text-muted/85">
            {note}
          </p>
        </>
      ) : null}
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
