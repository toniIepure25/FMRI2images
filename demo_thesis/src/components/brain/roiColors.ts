import type { RoiLayout } from '@/types';

/** Category-driven accent colors (early / intermediate / face-body / scene / default). */
const CATEGORY_PALETTE: Record<string, string> = {
  early: '#22d3ee',
  v1: '#22d3ee',
  v2: '#22d3ee',
  v3: '#06b6d4',
  intermediate: '#a855f7',
  v4: '#a855f7',
  ventral: '#c084fc',
  face: '#ec4899',
  body: '#f472b6',
  ffa: '#ec4899',
  eba: '#f472b6',
  scene: '#22c55e',
  place: '#10b981',
  ppa: '#22c55e',
  rsc: '#16a34a',
  opa: '#15803d',
  general: '#f59e0b',
  other: '#f59e0b',
  dorsal: '#fb923c',
};

export function resolveRoiColor(
  categoryKey: string,
  layout: RoiLayout | null | undefined,
  fallbackHex?: string,
): string {
  if (fallbackHex && /^#[a-f0-9]{6}$/i.test(fallbackHex)) {
    return fallbackHex;
  }
  const cat = layout?.categories?.[categoryKey];
  if (cat?.color && /^#[a-f0-9]{6}$/i.test(cat.color)) {
    return cat.color;
  }
  const lower = categoryKey.toLowerCase();
  for (const [key, hex] of Object.entries(CATEGORY_PALETTE)) {
    if (lower.includes(key)) return hex;
  }
  return '#f59e0b';
}
