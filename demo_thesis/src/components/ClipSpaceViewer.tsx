import Plot from 'react-plotly.js';
import type { Config, Data, Layout } from 'plotly.js';
import { useMemo, useState } from 'react';
import type { ClipCluster, ClipPoint, ClipProjection, ClipSpaceRef } from '@/types';

const BG = '#0a0e1a';
const QUERY_CYAN = '#00e5ff';
const TARGET_GREEN = '#34d399';
const RETRIEVED_PURPLE = '#c084fc';
const LINE_COLOR = 'rgba(192, 132, 252, 0.55)';

export interface ClipSpaceViewerProps {
  projection?: ClipProjection | null;
  caseClipSpace?: ClipSpaceRef;
  className?: string;
  showLines?: boolean;
  flyToQuery?: boolean;
  mode?: '2d' | '3d';
  onModeChange?: (mode: '2d' | '3d') => void;
}

function buildPointIndex(points: ClipPoint[]): Map<string, ClipPoint> {
  return new Map(points.map((p) => [p.id, p]));
}

function resolveQueryPoint(
  points: ClipPoint[],
  byId: Map<string, ClipPoint>,
  ref?: ClipSpaceRef
): ClipPoint | undefined {
  if (ref?.queryPointId) {
    const p = byId.get(ref.queryPointId);
    if (p) return p;
  }
  return points.find((p) => p.type === 'query_predicted');
}

function resolveTargetPoint(
  points: ClipPoint[],
  byId: Map<string, ClipPoint>,
  ref?: ClipSpaceRef
): ClipPoint | undefined {
  if (ref?.targetPointId) {
    const p = byId.get(ref.targetPointId);
    if (p) return p;
  }
  return points.find((p) => p.type === 'query_target');
}

function resolveRetrievedPoints(
  points: ClipPoint[],
  byId: Map<string, ClipPoint>,
  ref?: ClipSpaceRef
): ClipPoint[] {
  const fromIds = (ref?.topKPointIds ?? [])
    .map((id) => byId.get(id))
    .filter((p): p is ClipPoint => Boolean(p));
  if (fromIds.length > 0) {
    return [...fromIds].sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
  }
  return points
      .filter((p) => p.type === 'retrieved')
      .sort((a, b) => (a.rank ?? 99) - (b.rank ?? 99));
}

function coord2d(p: ClipPoint): { x: number; y: number } {
  return { x: p.x2d, y: p.y2d };
}

function coord3d(p: ClipPoint): { x: number; y: number; z: number } {
  return { x: p.x, y: p.y, z: p.z };
}

function scatterRange(values: number[], padRatio = 0.08): [number, number] {
  if (values.length === 0) return [-1, 1];
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min -= 0.5;
    max += 0.5;
  }
  const span = max - min;
  const pad = Math.max(span * padRatio, 0.02);
  return [min - pad, max + pad];
}

function buildLineSegments(
  query: ClipPoint | undefined,
  retrieved: ClipPoint[],
  use3d: boolean
): { x: (number | null)[]; y: (number | null)[]; z?: (number | null)[] } {
  const x: (number | null)[] = [];
  const y: (number | null)[] = [];
  const z: (number | null)[] = [];
  if (!query) return use3d ? { x, y, z } : { x, y };
  if (use3d) {
    const q3 = coord3d(query);
    for (const r of retrieved) {
      const c3 = coord3d(r);
      x.push(q3.x, c3.x, null);
      y.push(q3.y, c3.y, null);
      z.push(q3.z, c3.z, null);
    }
    return { x, y, z };
  }
  const q2 = coord2d(query);
  for (const r of retrieved) {
    const c2 = coord2d(r);
    x.push(q2.x, c2.x, null);
    y.push(q2.y, c2.y, null);
  }
  return { x, y };
}

export function ClipSpaceViewer({
  projection,
  caseClipSpace,
  className = '',
  showLines = true,
  flyToQuery = false,
  mode,
  onModeChange,
}: ClipSpaceViewerProps) {
  const [internalMode, setInternalMode] = useState<'2d' | '3d'>('3d');
  const viewMode = mode ?? internalMode;

  const setViewMode = (m: '2d' | '3d') => {
    if (mode === undefined) setInternalMode(m);
    onModeChange?.(m);
  };

  const { traces, layout } = useMemo(() => {
    if (!projection || !projection.points?.length) {
      return {
        traces: [] as Data[],
        layout: {} as Partial<Layout>,
      };
    }

    const points = projection.points;
    const byId = buildPointIndex(points);
    const queryPt = resolveQueryPoint(points, byId, caseClipSpace);
    const targetPt = resolveTargetPoint(points, byId, caseClipSpace);
    const retrievedPts = resolveRetrievedPoints(points, byId, caseClipSpace);

    const specialIds = new Set<string>([
      ...(queryPt ? [queryPt.id] : []),
      ...(targetPt ? [targetPt.id] : []),
      ...retrievedPts.map((p) => p.id),
    ]);

    const galleryPoints = points.filter(
      (p) => p.type === 'gallery' && !specialIds.has(p.id)
    );

    const galleryByCluster = new Map<string, ClipPoint[]>();
    for (const p of galleryPoints) {
      const key = p.category;
      const list = galleryByCluster.get(key) ?? [];
      list.push(p);
      galleryByCluster.set(key, list);
    }

    const tracesOut: Data[] = [];
    const use3d = viewMode === '3d';

    for (const cluster of projection.clusters) {
      const clusterPoints = galleryByCluster.get(cluster.name);
      if (!clusterPoints?.length) continue;

      const xs = use3d ? clusterPoints.map((p) => p.x) : clusterPoints.map((p) => p.x2d);
      const ys = use3d ? clusterPoints.map((p) => p.y) : clusterPoints.map((p) => p.y2d);
      const zs = use3d ? clusterPoints.map((p) => p.z) : undefined;

      const customdata = clusterPoints.map((p) => [
        p.category,
        p.label,
        p.score != null ? `similarity: ${p.score}` : '',
        p.nsdId != null ? `nsdId: ${p.nsdId}` : '',
        p.rank != null ? `rank: ${p.rank}` : '',
      ]);

      const baseTrace: Partial<Data> = {
        name: cluster.name,
        legendgroup: cluster.name,
        customdata,
        hovertemplate:
          '<b>%{customdata[1]}</b><br>category: %{customdata[0]}' +
          '<br>%{customdata[2]}<br>%{customdata[3]}<br>%{customdata[4]}<extra></extra>',
        marker: {
          size: 4,
          color: cluster.color,
          opacity: 0.35,
          line: { width: 0 },
        },
        showlegend: true,
      };

      if (use3d) {
        tracesOut.push({
          type: 'scatter3d',
          mode: 'markers',
          x: xs,
          y: ys,
          z: zs as number[],
          ...baseTrace,
        } as Data);
      } else {
        tracesOut.push({
          type: 'scatter',
          mode: 'markers',
          x: xs,
          y: ys,
          ...baseTrace,
        } as Data);
      }
    }

    const labelXs: number[] = [];
    const labelYs: number[] = [];
    const labelZs: number[] = [];
    const labelTexts: string[] = [];
    const labelColors: string[] = [];

    for (const cluster of projection.clusters) {
      const members = galleryByCluster.get(cluster.name) ?? [];
      labelTexts.push(cluster.name);
      labelColors.push(cluster.color);
      if (use3d) {
        labelXs.push(cluster.centroid.x);
        labelYs.push(cluster.centroid.y);
        labelZs.push(cluster.centroid.z);
      } else if (members.length) {
        const mx = members.reduce((s, p) => s + p.x2d, 0) / members.length;
        const my = members.reduce((s, p) => s + p.y2d, 0) / members.length;
        labelXs.push(mx);
        labelYs.push(my);
        labelZs.push(0);
      } else {
        labelXs.push(cluster.centroid.x);
        labelYs.push(cluster.centroid.y);
        labelZs.push(0);
      }
    }

    if (labelTexts.length) {
      if (use3d) {
        tracesOut.push({
          type: 'scatter3d',
          mode: 'text',
          x: labelXs,
          y: labelYs,
          z: labelZs,
          text: labelTexts,
          textfont: {
            color: labelColors.map((c) => c),
            size: 11,
            family: 'Inter, system-ui, sans-serif',
          },
          textposition: 'top center',
          hoverinfo: 'skip',
          showlegend: false,
        } as Data);
      } else {
        tracesOut.push({
          type: 'scatter',
          mode: 'text',
          x: labelXs,
          y: labelYs,
          text: labelTexts,
          textfont: {
            color: labelColors.map((c) => c),
            size: 12,
            family: 'Inter, system-ui, sans-serif',
          },
          textposition: 'top center',
          hoverinfo: 'skip',
          showlegend: false,
        } as Data);
      }
    }

    if (showLines && queryPt && retrievedPts.length) {
      const seg = buildLineSegments(queryPt, retrievedPts, use3d);
      if (use3d) {
        tracesOut.push({
          type: 'scatter3d',
          mode: 'lines',
          x: seg.x,
          y: seg.y,
          z: seg.z,
          line: { color: LINE_COLOR, width: 2, dash: 'dash' },
          hoverinfo: 'skip',
          showlegend: false,
        } as Data);
      } else {
        tracesOut.push({
          type: 'scatter',
          mode: 'lines',
          x: seg.x,
          y: seg.y,
          line: { color: LINE_COLOR, width: 2, dash: 'dash' },
          hoverinfo: 'skip',
          showlegend: false,
        } as Data);
      }
    }

    if (queryPt) {
      const qCustom = [
        queryPt.category,
        queryPt.label,
        queryPt.score ?? '',
        queryPt.rank ?? '',
      ];
      if (use3d) {
        const q3 = coord3d(queryPt);
        tracesOut.push({
          type: 'scatter3d',
          mode: 'markers',
          name: 'Query (predicted)',
          x: [q3.x],
          y: [q3.y],
          z: [q3.z],
          customdata: [qCustom],
          hovertemplate:
            '<b>Predicted query</b><br>%{customdata[1]}<br>category: %{customdata[0]}<br>' +
            'similarity: %{customdata[2]}<extra></extra>',
          marker: {
            size: 12,
            color: QUERY_CYAN,
            opacity: 1,
            symbol: 'circle',
            line: { color: 'rgba(0, 229, 255, 0.95)', width: 3 },
          },
          showlegend: true,
        } as Data);
      } else {
        const q2 = coord2d(queryPt);
        tracesOut.push({
          type: 'scatter',
          mode: 'markers',
          name: 'Query (predicted)',
          x: [q2.x],
          y: [q2.y],
          customdata: [qCustom],
          hovertemplate:
            '<b>Predicted query</b><br>%{customdata[1]}<br>category: %{customdata[0]}<br>' +
            'similarity: %{customdata[2]}<extra></extra>',
          marker: {
            size: 16,
            color: QUERY_CYAN,
            opacity: 1,
            symbol: 'circle',
            line: { color: 'rgba(0, 229, 255, 0.95)', width: 3 },
          },
          showlegend: true,
        } as Data);
      }
    }

    if (targetPt) {
      const tCustom = [
        targetPt.category,
        targetPt.label,
        targetPt.score ?? '',
        targetPt.nsdId ?? '',
      ];
      if (use3d) {
        const t3 = coord3d(targetPt);
        tracesOut.push({
          type: 'scatter3d',
          mode: 'markers',
          name: 'Target',
          x: [t3.x],
          y: [t3.y],
          z: [t3.z],
          customdata: [tCustom],
          hovertemplate:
            '<b>Ground-truth target</b><br>%{customdata[1]}<br>category: %{customdata[0]}<br>' +
            'nsdId: %{customdata[3]}<extra></extra>',
          marker: {
            size: 10,
            color: TARGET_GREEN,
            opacity: 1,
            symbol: 'star',
            line: { color: '#065f46', width: 1 },
          },
          showlegend: true,
        } as Data);
      } else {
        const t2 = coord2d(targetPt);
        tracesOut.push({
          type: 'scatter',
          mode: 'markers',
          name: 'Target',
          x: [t2.x],
          y: [t2.y],
          customdata: [tCustom],
          hovertemplate:
            '<b>Ground-truth target</b><br>%{customdata[1]}<br>category: %{customdata[0]}<br>' +
            'nsdId: %{customdata[3]}<extra></extra>',
          marker: {
            size: 18,
            color: TARGET_GREEN,
            opacity: 1,
            symbol: 'star',
            line: { color: '#065f46', width: 1 },
          },
          showlegend: true,
        } as Data);
      }
    }

    if (retrievedPts.length) {
      const rx = use3d ? retrievedPts.map((p) => p.x) : retrievedPts.map((p) => p.x2d);
      const ry = use3d ? retrievedPts.map((p) => p.y) : retrievedPts.map((p) => p.y2d);
      const rz = use3d ? retrievedPts.map((p) => p.z) : undefined;
      const rCustom = retrievedPts.map((p) => [
        p.category,
        p.label,
        p.score ?? '',
        p.rank ?? '',
      ]);
      if (use3d) {
        tracesOut.push({
          type: 'scatter3d',
          mode: 'markers',
          name: 'Top-K retrieved',
          x: rx,
          y: ry,
          z: rz as number[],
          customdata: rCustom,
          hovertemplate:
            '<b>Retrieved #%{customdata[3]}</b><br>%{customdata[1]}<br>' +
            'category: %{customdata[0]}<br>similarity: %{customdata[2]}<extra></extra>',
          marker: {
            size: 8,
            color: RETRIEVED_PURPLE,
            opacity: 0.95,
            symbol: 'diamond',
            line: { color: '#7c3aed', width: 1 },
          },
          showlegend: true,
        } as Data);
      } else {
        tracesOut.push({
          type: 'scatter',
          mode: 'markers',
          name: 'Top-K retrieved',
          x: rx,
          y: ry,
          customdata: rCustom,
          hovertemplate:
            '<b>Retrieved #%{customdata[3]}</b><br>%{customdata[1]}<br>' +
            'category: %{customdata[0]}<br>similarity: %{customdata[2]}<extra></extra>',
          marker: {
            size: 12,
            color: RETRIEVED_PURPLE,
            opacity: 0.95,
            symbol: 'diamond',
            line: { color: '#7c3aed', width: 1 },
          },
          showlegend: true,
        } as Data);
      }
    }

    const allX = use3d ? points.map((p) => p.x) : points.map((p) => p.x2d);
    const allY = use3d ? points.map((p) => p.y) : points.map((p) => p.y2d);

    const baseLayout: Partial<Layout> = {
      paper_bgcolor: BG,
      plot_bgcolor: BG,
      font: { color: '#f8fafc', family: 'Inter, system-ui, sans-serif', size: 12 },
      margin: { l: 0, r: 12, t: 28, b: 0 },
      legend: {
        font: { color: '#e2e8f0', size: 11 },
        bgcolor: 'rgba(10, 14, 26, 0.85)',
        bordercolor: 'rgba(148, 163, 184, 0.25)',
        borderwidth: 1,
        orientation: 'v',
        x: 1.02,
        y: 1,
      },
      title: {
        text: `CLIP space (${projection.method})`,
        font: { color: '#f1f5f9', size: 14 },
        x: 0,
        xanchor: 'left',
      },
    };

    if (use3d) {
      let camera = {
        eye: { x: 1.65, y: 1.55, z: 1.35 },
        center: { x: 0, y: 0, z: 0 },
      };
      if (flyToQuery && queryPt) {
        const c = coord3d(queryPt);
        camera = {
          eye: { x: c.x + 1.4, y: c.y + 1.3, z: c.z + 1.2 },
          center: { x: c.x, y: c.y, z: c.z },
        };
      }
      Object.assign(baseLayout, {
        scene: {
          xaxis: {
            backgroundcolor: BG,
            gridcolor: 'rgba(148, 163, 184, 0.08)',
            showbackground: true,
            zerolinecolor: 'rgba(148, 163, 184, 0.12)',
            color: '#cbd5f5',
            title: { text: 'UMAP-1', font: { color: '#94a3b8', size: 11 } },
          },
          yaxis: {
            backgroundcolor: BG,
            gridcolor: 'rgba(148, 163, 184, 0.08)',
            showbackground: true,
            zerolinecolor: 'rgba(148, 163, 184, 0.12)',
            color: '#cbd5f5',
            title: { text: 'UMAP-2', font: { color: '#94a3b8', size: 11 } },
          },
          zaxis: {
            backgroundcolor: BG,
            gridcolor: 'rgba(148, 163, 184, 0.08)',
            showbackground: true,
            zerolinecolor: 'rgba(148, 163, 184, 0.12)',
            color: '#cbd5f5',
            title: { text: 'UMAP-3', font: { color: '#94a3b8', size: 11 } },
          },
          bgcolor: BG,
          camera,
          aspectmode: 'data',
        },
      } as Partial<Layout>);
    } else {
      const xr = flyToQuery && queryPt ? scatterRange([queryPt.x2d, ...retrievedPts.map((p) => p.x2d)], 0.35) : scatterRange(allX);
      const yr = flyToQuery && queryPt ? scatterRange([queryPt.y2d, ...retrievedPts.map((p) => p.y2d)], 0.35) : scatterRange(allY);
      Object.assign(baseLayout, {
        xaxis: {
          range: xr,
          showgrid: true,
          gridcolor: 'rgba(148, 163, 184, 0.08)',
          zerolinecolor: 'rgba(148, 163, 184, 0.12)',
          color: '#cbd5f5',
          title: { text: 'UMAP-1', font: { color: '#94a3b8', size: 11 } },
        },
        yaxis: {
          range: yr,
          showgrid: true,
          gridcolor: 'rgba(148, 163, 184, 0.08)',
          zerolinecolor: 'rgba(148, 163, 184, 0.12)',
          color: '#cbd5f5',
          title: { text: 'UMAP-2', font: { color: '#94a3b8', size: 11 } },
        },
      } as Partial<Layout>);
    }

    return { traces: tracesOut, layout: baseLayout };
  }, [projection, caseClipSpace, viewMode, showLines, flyToQuery]);

  const plotRevision = useMemo(() => {
    const s = `${viewMode}:${flyToQuery}:${projection?.points?.length ?? 0}:${projection?.method ?? ''}`;
    let h = 0;
    for (let i = 0; i < s.length; i += 1) h = (h * 33 + s.charCodeAt(i)) | 0;
    return h;
  }, [viewMode, flyToQuery, projection?.points?.length, projection?.method]);

  const config = useMemo<Partial<Config>>(
    () => ({
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
      toImageButtonOptions: {
        format: 'png',
        filename: 'clip_space',
      },
    }),
    []
  );

  if (!projection) {
    return (
      <div
        className={`rounded-xl border border-brain-border/50 bg-[#0a0e1a] p-8 text-center text-slate-400 ${className}`}
      >
        <p className="text-sm">Loading CLIP embedding projection…</p>
      </div>
    );
  }

  if (!projection.points?.length) {
    return (
      <div
        className={`rounded-xl border border-brain-border/50 bg-[#0a0e1a] p-8 text-center text-slate-400 ${className}`}
      >
        <p className="text-sm">No projection points available for this view.</p>
        <p className="mt-3 max-w-xl mx-auto text-xs text-slate-500 leading-relaxed">
          The model does not directly see the image. It predicts a visual-semantic CLIP embedding from
          fMRI activity, then retrieves or reconstructs visual content from that embedding.
        </p>
      </div>
    );
  }

  return (
    <div className={`flex flex-col gap-3 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-xs uppercase tracking-wider text-slate-500">View</span>
        <div className="flex rounded-lg border border-brain-border/50 bg-brain-panel/60 p-0.5">
          <button
            type="button"
            onClick={() => setViewMode('2d')}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              viewMode === '2d'
                ? 'bg-brain-accent/20 text-brain-accent shadow-[0_0_12px_rgba(0,212,255,0.15)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            2D map
          </button>
          <button
            type="button"
            onClick={() => setViewMode('3d')}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
              viewMode === '3d'
                ? 'bg-brain-accent/20 text-brain-accent shadow-[0_0_12px_rgba(0,212,255,0.15)]'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            3D map
          </button>
        </div>
      </div>

      <div className="neon-border overflow-hidden rounded-xl bg-[#0a0e1a]" style={{ minHeight: 420 }}>
        <Plot
          data={traces}
          layout={layout}
          config={config}
          style={{ width: '100%', height: 420 }}
          useResizeHandler
          revision={plotRevision}
        />
      </div>

      <p className="text-xs leading-relaxed text-slate-500 max-w-3xl">
        The model does not directly see the image. It predicts a visual-semantic CLIP embedding from
        fMRI activity, then retrieves or reconstructs visual content from that embedding.
      </p>
    </div>
  );
}
