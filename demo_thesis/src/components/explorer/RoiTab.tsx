import { useMemo, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import type { DemoCase } from '@/types';
import { useRoiLayout } from '@/lib/hooks';
import { resolveRoiColor } from '@/components/brain/roiColors';

export interface RoiTabProps {
  case_: DemoCase;
}

export function RoiTab({ case_ }: RoiTabProps) {
  const { layout, loading } = useRoiLayout();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const chartData = useMemo(() => {
    return [...case_.roiScores]
      .sort((a, b) => b.contribution - a.contribution)
      .map((s) => {
        const def = layout?.rois.find((r) => r.name === s.name);
        const cat = def?.category ?? 'general';
        const fill = resolveRoiColor(cat, layout ?? undefined, def?.color);
        return {
          name: s.name,
          contribution: s.contribution,
          activation: s.activation,
          category: cat,
          fill,
        };
      });
  }, [case_.roiScores, layout]);

  const interpretation = useMemo(() => {
    if (!chartData.length) return '';
    const top = chartData[0];
    const runners = chartData.slice(1, 3).map((r) => r.name);
    return `This trial is dominated by ${top.name} (${top.category}), accounting for the largest share of fused CLIP direction. ${runners.length ? `Secondary drives include ${runners.join(' and ')}.` : ''} Use the table for fine-grained activation and agreement cues.`;
  }, [chartData]);

  const roiDefs = layout?.rois ?? [];

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <div className="glass-panel p-5">
        <h2 className="text-sm font-semibold text-white">ROI explainability</h2>
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          Contributions sum how each visual area pushes the final hyperspherical prediction before diffusion. Categories
          are color-coded consistently with the 3D viewer.
        </p>
      </div>

      <div className="glass-panel p-4" style={{ height: Math.max(360, chartData.length * 36) }}>
        {loading && !layout ? (
          <div className="flex h-[320px] items-center justify-center text-sm text-slate-500">Loading layout…</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={chartData}
              margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
              <XAxis
                type="number"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={108}
                stroke="#64748b"
                tick={{ fill: '#e2e8f0', fontSize: 10 }}
                tickLine={false}
              />
              <Tooltip
                cursor={{ fill: 'rgba(15,23,42,0.35)' }}
                contentStyle={{
                  background: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: '#f1f5f9' }}
                formatter={(value: number, name: string) => [value.toFixed(3), name === 'contribution' ? 'Contribution' : name]}
              />
              <Bar dataKey="contribution" radius={[0, 6, 6, 0]} barSize={14}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${entry.name}-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="glass-panel p-5">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Auto summary</p>
        <p className="mt-2 text-sm leading-relaxed text-slate-300">{interpretation}</p>
      </div>

      <div className="glass-panel overflow-hidden">
        <div className="border-b border-brain-border/30 px-5 py-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">All ROI scores</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-xs text-slate-300">
            <thead className="bg-black/30 text-[10px] uppercase tracking-wider text-slate-500">
              <tr>
                <th className="px-4 py-3">ROI</th>
                <th className="px-4 py-3">Hemisphere</th>
                <th className="px-4 py-3">Activation</th>
                <th className="px-4 py-3">Contribution</th>
                <th className="px-4 py-3">Confidence</th>
                <th className="px-4 py-3">Agreement</th>
              </tr>
            </thead>
            <tbody>
              {case_.roiScores.map((r) => (
                <tr key={r.name} className="border-t border-brain-border/20 hover:bg-white/[0.03]">
                  <td className="px-4 py-2.5 font-medium text-slate-200">{r.name}</td>
                  <td className="px-4 py-2.5 text-slate-500">{r.hemisphere}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-400">{r.activation.toFixed(3)}</td>
                  <td className="px-4 py-2.5 font-mono text-cyan-200/90">{r.contribution.toFixed(3)}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-400">{r.confidence.toFixed(3)}</td>
                  <td className="px-4 py-2.5 font-mono text-slate-400">{r.agreement.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="space-y-3">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          What each region means
        </p>
        <div className="space-y-2">
          {roiDefs.map((def) => {
            const open = expanded[def.name];
            return (
              <div key={def.name} className="glass-panel overflow-hidden">
                <button
                  type="button"
                  onClick={() => setExpanded((e) => ({ ...e, [def.name]: !open }))}
                  className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left transition hover:bg-white/[0.03]"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <span
                      className="h-2.5 w-2.5 flex-shrink-0 rounded-full"
                      style={{ backgroundColor: resolveRoiColor(def.category, layout ?? undefined, def.color) }}
                    />
                    <span className="truncate text-sm font-medium text-white">{def.fullName}</span>
                    <span className="hidden text-[10px] text-slate-500 sm:inline">{def.category}</span>
                  </div>
                  <span className="text-slate-500">{open ? '−' : '+'}</span>
                </button>
                {open && (
                  <div className="border-t border-brain-border/30 bg-black/20 px-4 py-3 text-xs leading-relaxed text-slate-400">
                    <p className="font-medium text-slate-300">{def.functionSummary}</p>
                    <p className="mt-2">{def.description}</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
