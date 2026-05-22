import type { ReactNode } from 'react';

/**
 * Shared kicker-rule header for every Manifold page section. Matches the
 * grammar used elsewhere in the premium pipeline UI (hairline + uppercase
 * kicker + verdict + lead paragraph + optional right-side metadata).
 */
export function ManifoldSectionHeader({
  kicker,
  title,
  description,
  action,
}: {
  kicker: string;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div className="min-w-0">
        <div className="flex items-center gap-2.5">
          <span className="h-px w-6 bg-accent/45" aria-hidden />
          <p className="premium-kicker">{kicker}</p>
        </div>
        <h2 className="mt-2 text-[22px] font-semibold tracking-tight text-text-primary sm:text-[26px]">
          {title}
        </h2>
        {description ? (
          <p className="mt-2 max-w-[44rem] text-[12.5px] leading-relaxed text-text-secondary">
            {description}
          </p>
        ) : null}
      </div>
      {action ? <div className="shrink-0 sm:self-end">{action}</div> : null}
    </header>
  );
}
