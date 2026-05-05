import { forwardRef, useCallback, useState, type ImgHTMLAttributes, type SyntheticEvent } from 'react';

export type SafeImgProps = ImgHTMLAttributes<HTMLImageElement>;

/**
 * Image with loading shimmer and an accessible fallback when loading fails.
 * Forwards all standard `<img>` attributes and ref to the underlying image element.
 * Use `className` for layout/size on the wrapper; the inner image fills it (`object-cover`).
 */
export const SafeImg = forwardRef<HTMLImageElement, SafeImgProps>(function SafeImg(
  { className, onLoad, onError, alt = '', ...rest },
  ref
) {
  const [phase, setPhase] = useState<'loading' | 'loaded' | 'error'>('loading');

  const handleLoad = useCallback(
    (e: SyntheticEvent<HTMLImageElement>) => {
      setPhase('loaded');
      onLoad?.(e);
    },
    [onLoad]
  );

  const handleError = useCallback(
    (e: SyntheticEvent<HTMLImageElement>) => {
      setPhase('error');
      onError?.(e);
    },
    [onError]
  );

  if (phase === 'error') {
    return (
      <div
        role="img"
        aria-label={alt}
        className={`flex min-h-[120px] w-full items-center justify-center rounded-lg border border-slate-700/60 bg-slate-900/95 px-4 text-center text-xs font-medium text-slate-500 ${className ?? ''}`}
      >
        Image unavailable
      </div>
    );
  }

  return (
    <span className={`relative inline-block overflow-hidden ${className ?? ''}`}>
      {phase === 'loading' ? (
        <span className="pointer-events-none absolute inset-0 z-10 block bg-slate-800/90 shimmer-bg" aria-hidden />
      ) : null}
      <img
        ref={ref}
        alt={alt}
        className={`block h-full w-full max-w-full object-cover transition-opacity duration-300 ${
          phase === 'loading' ? 'opacity-0' : 'opacity-100'
        }`}
        onLoad={handleLoad}
        onError={handleError}
        {...rest}
      />
    </span>
  );
});
