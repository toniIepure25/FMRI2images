import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';

function SafeImage({ src, alt }: { src?: string | null; alt: string }) {
  const [ok, setOk] = useState(true);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    setOk(true);
    setLoaded(false);
  }, [src]);

  if (!src || !ok) {
    return (
      <div className="flex h-full w-full items-center justify-center bg-surface-raised">
        <span className="text-xs text-text-muted">Unavailable</span>
      </div>
    );
  }

  return (
    <>
      {!loaded ? <div className="absolute inset-0 shimmer-bg" aria-hidden /> : null}
      <img
        src={src}
        alt={alt}
        className={`h-full w-full object-cover transition duration-500 ${loaded ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setLoaded(true)}
        onError={() => setOk(false)}
        loading="lazy"
      />
    </>
  );
}

export function ImageEvidenceCard({
  title,
  subtitle,
  imageSrc,
  badge,
  footer,
  emphasis = false,
  aspectClass = 'aspect-[4/3]',
  className = '',
}: {
  title: string;
  subtitle?: string;
  imageSrc?: string | null;
  badge?: ReactNode;
  footer?: ReactNode;
  emphasis?: boolean;
  aspectClass?: string;
  className?: string;
}) {
  return (
    <article
      className={`premium-image-card ${emphasis ? 'premium-image-card-emphasis' : ''} ${className}`}
    >
      <div className={`relative ${aspectClass} overflow-hidden rounded-[13px] bg-surface-base`}>
        <SafeImage src={imageSrc} alt={title} />
        <div className="absolute inset-0 bg-gradient-to-t from-black/78 via-black/5 to-transparent" />
        <div className="absolute left-3 top-3">{badge}</div>
        <div className="absolute inset-x-0 bottom-0 p-3">
          <p className="text-sm font-semibold text-white">{title}</p>
          {subtitle ? <p className="mt-0.5 text-[11px] text-white/68">{subtitle}</p> : null}
        </div>
      </div>
      {footer ? <div className="px-1 pt-3">{footer}</div> : null}
    </article>
  );
}
