import type { ReactNode } from 'react';

export function SectionHeader({
  eyebrow,
  title,
  description,
  action,
  className = '',
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between ${className}`}>
      <div className="min-w-0">
        {eyebrow ? <p className="premium-kicker mb-2">{eyebrow}</p> : null}
        <h2 className="text-2xl font-semibold tracking-tight text-text-primary sm:text-3xl">{title}</h2>
        {description ? <p className="mt-2 max-w-2xl text-sm leading-relaxed text-text-secondary">{description}</p> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}
