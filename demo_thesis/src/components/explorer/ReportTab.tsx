import type { DemoCase } from '@/types';
import { getConfidenceColor, getConfidenceLabel, getDifficultyColor } from '@/lib/data';
import { SafeImg } from '@/components/ui/SafeImg';

export interface ReportTabProps {
  case_: DemoCase;
}

export function ReportTab({ case_ }: ReportTabProps) {
  const top1 = case_.retrievedImages.find((r) => r.rank === 1) ?? case_.retrievedImages[0];
  const confColor = getConfidenceColor(case_.uncertainty.confidenceLevel);
  const diffColor = getDifficultyColor(case_.difficulty);

  return (
    <>
      <style>{`
        @media print {
          .explorer-report-root {
            background: white !important;
            color: #0f172a !important;
            padding: 0 !important;
          }
          .explorer-report-root * {
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
            box-shadow: none !important;
          }
          .explorer-report-root .glass-panel,
          .explorer-report-root .report-card {
            background: #f8fafc !important;
            backdrop-filter: none !important;
          }
          .report-no-print {
            display: none !important;
          }
          .explorer-report-root img {
            max-height: 220px;
            object-fit: contain;
          }
          .explorer-report-root a {
            color: #0f172a !important;
          }
        }
      `}</style>

      <div className="explorer-report-root mx-auto max-w-4xl space-y-6 text-slate-200">
        <div className="report-no-print flex flex-wrap items-center justify-between gap-3">
          <p className="text-xs text-slate-500">Print-friendly trial dossier for committee review.</p>
          <button
            type="button"
            onClick={() => window.print()}
            className="rounded-lg border border-brain-accent/40 bg-brain-accent/15 px-4 py-2 text-xs font-semibold text-brain-accent transition hover:bg-brain-accent/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent"
          >
            Print / Save PDF
          </button>
        </div>

        <header className="report-card glass-panel p-6">
          <h1 className="text-2xl font-semibold tracking-tight text-white print:text-slate-900">Trial report</h1>
          <p className="mt-2 text-sm text-slate-400 print:text-slate-600">{case_.title}</p>
          <div className="mt-4 flex flex-wrap gap-2 text-[11px]">
            <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 font-mono print:border-slate-300 print:bg-slate-100">
              {case_.id}
            </span>
            <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 print:border-slate-300 print:bg-slate-100">
              {case_.subject}
            </span>
            <span className="rounded-md border border-white/10 bg-white/5 px-2 py-1 print:border-slate-300 print:bg-slate-100">
              nsdId {case_.nsdId}
            </span>
            <span
              className="rounded-md border px-2 py-1 capitalize print:border-slate-300"
              style={{ borderColor: `${diffColor}55`, color: diffColor }}
            >
              {case_.difficulty}
            </span>
            <span
              className="rounded-md border px-2 py-1 print:border-slate-300"
              style={{ borderColor: `${confColor}55`, color: confColor }}
            >
              {getConfidenceLabel(case_.uncertainty.confidenceLevel)}
            </span>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-slate-300 print:text-slate-700">{case_.description}</p>
        </header>

        <section className="report-card glass-panel p-6">
          <h2 className="text-sm font-semibold text-white print:text-slate-900">Visual evidence</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <figure className="overflow-hidden rounded-lg border border-brain-border/40 bg-black/30 print:border-slate-300 print:bg-white">
              <figcaption className="border-b border-brain-border/30 px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500 print:border-slate-200 print:text-slate-600">
                Target
              </figcaption>
              <SafeImg src={case_.targetImage} alt="Target" className="block w-full" />
            </figure>
            <figure className="overflow-hidden rounded-lg border border-brain-border/40 bg-black/30 print:border-slate-300 print:bg-white">
              <figcaption className="border-b border-brain-border/30 px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500 print:border-slate-200 print:text-slate-600">
                Top-1 retrieval
              </figcaption>
              {top1 ? (
                <SafeImg src={top1.image} alt="Top-1" className="block w-full" />
              ) : (
                <div className="p-6 text-center text-sm text-slate-500">—</div>
              )}
            </figure>
            <figure className="overflow-hidden rounded-lg border border-brain-border/40 bg-black/30 print:border-slate-300 print:bg-white">
              <figcaption className="border-b border-brain-border/30 px-3 py-2 text-[11px] uppercase tracking-wider text-slate-500 print:border-slate-200 print:text-slate-600">
                Reconstruction
              </figcaption>
              <SafeImg src={case_.reconstructionImage} alt="Reconstruction" className="block w-full" />
            </figure>
          </div>
        </section>

        <section className="report-card glass-panel overflow-hidden p-0">
          <div className="border-b border-brain-border/30 px-6 py-4 print:border-slate-200">
            <h2 className="text-sm font-semibold text-white print:text-slate-900">Metrics</h2>
          </div>
          <table className="w-full text-left text-sm">
            <tbody className="divide-y divide-brain-border/25 print:divide-slate-200">
              <ReportRow label="Retrieval rank" value={`#${case_.metrics.rank}`} />
              <ReportRow label="Cosine similarity" value={case_.metrics.cosine.toFixed(4)} />
              <ReportRow label="CSLS" value={case_.metrics.csls.toFixed(4)} />
              <ReportRow label="R@1 correct" value={case_.metrics.r1Correct ? 'Yes' : 'No'} />
              <ReportRow label="R@5 correct" value={case_.metrics.r5Correct ? 'Yes' : 'No'} />
              <ReportRow label="PixCorr" value={case_.metrics.pixcorr?.toFixed(3) ?? '—'} />
              <ReportRow label="SSIM" value={case_.metrics.ssim?.toFixed(3) ?? '—'} />
              <ReportRow label="AlexNet-2" value={case_.metrics.alex2?.toFixed(3) ?? '—'} />
              <ReportRow label="AlexNet-5" value={case_.metrics.alex5?.toFixed(3) ?? '—'} />
            </tbody>
          </table>
        </section>

        <section className="report-card glass-panel overflow-hidden p-0">
          <div className="border-b border-brain-border/30 px-6 py-4 print:border-slate-200">
            <h2 className="text-sm font-semibold text-white print:text-slate-900">Uncertainty and DUA-CFG</h2>
          </div>
          <table className="w-full text-left text-sm">
            <tbody className="divide-y divide-brain-border/25 print:divide-slate-200">
              <ReportRow label="κ" value={case_.uncertainty.kappa.toFixed(4)} />
              <ReportRow label="κ (normalized)" value={case_.uncertainty.kappaNorm.toFixed(4)} />
              <ReportRow label="δ" value={case_.uncertainty.delta.toFixed(4)} />
              <ReportRow label="Confidence" value={getConfidenceLabel(case_.uncertainty.confidenceLevel)} />
              <ReportRow label="Guidance scale" value={case_.duaCfg.guidanceScale.toFixed(2)} />
              <ReportRow label="Diffusion steps" value={String(case_.duaCfg.diffusionSteps)} />
              <ReportRow label="Ensemble K" value={String(case_.duaCfg.ensembleK)} />
              <ReportRow label="Abstain" value={case_.duaCfg.abstain ? 'Yes' : 'No'} />
            </tbody>
          </table>
        </section>

        <section className="report-card glass-panel overflow-hidden p-0">
          <div className="border-b border-brain-border/30 px-6 py-4 print:border-slate-200">
            <h2 className="text-sm font-semibold text-white print:text-slate-900">ROI contributions</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-xs">
              <thead className="bg-black/25 text-[11px] uppercase tracking-wider text-slate-500 print:bg-slate-100 print:text-slate-600">
                <tr>
                  <th className="px-4 py-2">ROI</th>
                  <th className="px-4 py-2">Contribution</th>
                  <th className="px-4 py-2">Activation</th>
                  <th className="px-4 py-2">Agreement</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-brain-border/20 print:divide-slate-200">
                {case_.roiScores.map((r) => (
                  <tr key={r.name}>
                    <td className="px-4 py-2 font-medium">{r.name}</td>
                    <td className="px-4 py-2 font-mono">{r.contribution.toFixed(3)}</td>
                    <td className="px-4 py-2 font-mono">{r.activation.toFixed(3)}</td>
                    <td className="px-4 py-2 font-mono">{r.agreement.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="report-card glass-panel p-6">
          <h2 className="text-sm font-semibold text-white print:text-slate-900">Interpretation</h2>
          <p className="mt-3 text-sm leading-relaxed text-slate-300 print:text-slate-700">{case_.interpretation}</p>
        </section>
      </div>
    </>
  );
}

function ReportRow({ label, value }: { label: string; value: string }) {
  return (
    <tr>
      <td className="px-6 py-3 text-slate-400 print:text-slate-600">{label}</td>
      <td className="px-6 py-3 font-mono text-slate-100 print:text-slate-900">{value}</td>
    </tr>
  );
}
