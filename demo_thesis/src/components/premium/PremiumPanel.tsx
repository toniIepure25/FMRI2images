import type { ReactNode } from 'react';

type PremiumPanelVariant = 'default' | 'hero' | 'inset' | 'muted';

const variantClass: Record<PremiumPanelVariant, string> = {
  default: 'premium-panel',
  hero: 'premium-panel premium-panel-hero',
  inset: 'premium-panel premium-panel-inset',
  muted: 'premium-panel premium-panel-muted',
};

export function PremiumPanel({
  children,
  className = '',
  variant = 'default',
}: {
  children: ReactNode;
  className?: string;
  variant?: PremiumPanelVariant;
}) {
  return <section className={`${variantClass[variant]} ${className}`}>{children}</section>;
}
