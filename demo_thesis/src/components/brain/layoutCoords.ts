import type { RoiDefinition } from '@/types';

export interface LayoutNormalization {
  scale: number;
  cx: number;
  cy: number;
  cz: number;
}

/** Compute uniform scale and center so all ROI points fit roughly inside the ellipsoid brain. */
export function computeLayoutNormalization(rois: RoiDefinition[]): LayoutNormalization {
  if (!rois.length) {
    return { scale: 0.35, cx: 0, cy: 0, cz: 0 };
  }
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (const r of rois) {
    const { x, y, z } = r.position;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    minZ = Math.min(minZ, z);
    maxZ = Math.max(maxZ, z);
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const rawMax = Math.max(dx, dy, dz, 1e-6);
  const targetRadius = 0.82;
  const scale = targetRadius / rawMax;
  return { scale, cx, cy, cz };
}

/**
 * Map stored layout coordinates to Three.js scene space (Y-up).
 * Assumes common neuro-style axes; swaps so superior maps to +Y.
 */
export function layoutPositionToScene(
  pos: { x: number; y: number; z: number },
  norm: LayoutNormalization,
): [number, number, number] {
  const lx = (pos.x - norm.cx) * norm.scale;
  const ly = (pos.y - norm.cy) * norm.scale;
  const lz = (pos.z - norm.cz) * norm.scale;
  return [lx * 0.92, lz * 0.88, -ly * 0.92];
}
