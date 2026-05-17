import type { ReactNode } from 'react';

export function EmptyState({
  title,
  detail,
  compact = false,
  action,
  className = '',
}: {
  title: string;
  detail?: ReactNode;
  compact?: boolean;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`premium-empty ${compact ? 'premium-empty-compact' : ''} ${className}`}>
      <div className="h-7 w-7 rounded-full border border-border-subtle bg-surface-raised shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]" />
      <div>
        <p className="text-sm font-semibold text-text-primary">{title}</p>
        {detail ? <p className="mt-1 max-w-md text-xs leading-relaxed text-text-muted">{detail}</p> : null}
      </div>
      {action ? <div className="mt-1">{action}</div> : null}
    </div>
  );
}
