# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Cortex2Canvas** — presentation-grade React/Vite web app demonstrating an fMRI-to-image neural decoding pipeline (fMRI → ROI Transformer → CLIP space → retrieval → diffusion reconstruction) for a bachelor thesis defense. The frontend is the primary deliverable; the FastAPI backend is optional and only needed for "Live" inference mode.

## Commands

| Command | Purpose |
|---|---|
| `npm run dev` | Vite dev server on **port 3000** (strict; proxies `/api` → `localhost:8000`) |
| `npm run build` | `tsc -b` typecheck + production build |
| `npm run typecheck` | Type check only (no emit) — use this to validate edits without a full build |
| `npm run preview` | Preview the production build |
| `python3 scripts/prepare_demo_data.py` | Copies real results into `public/assets/cases/` or generates placeholders (needs Pillow) |
| `node scripts/export_manifold_evidence_data.mjs` | Regenerate manifold evidence JSON |
| `node scripts/sync_manifold_figures.mjs` | Sync manifold figures from `reports/` |
| `node scripts/verify_manifold_evidence.mjs` | Verify manifold evidence assets |

There are no JS/TS unit tests configured. Playwright is installed but used only for screenshot capture scripts (`scripts/capture_lab*_screenshots.mjs`).

Backend (optional): see `BACKEND_LOCAL_RUNBOOK.md`. Start with `uvicorn main:app --reload --port 8000` from `backend/`. Modes are selected via `C2C_BACKEND_MODE` env var: `v62_single` (default), `v61_single`, `v61_mctta`, `fusion_v61_v62`. Required artifacts live under `local_backend_data/` and are fetched via `scripts/fetch_live_backend_artifacts_from_pod.sh` (requires KUBECONFIG to the runai-romania-dev cluster).

## Architecture

### Routing & pages (`src/App.tsx`)
Five route pages, all lazy-loaded except `Home`:
- `/` Home — landing
- `/film` Film — cinematic 8-step walkthrough
- `/explorer`, `/explorer/:caseId` Explorer — tabbed dashboard
- `/pipeline` Pipeline — the scientifically-honest replay/live decoding view (most actively developed)
- `/evidence` Evidence — manifold/evidence artifacts view
- `/challenge` Challenge — blind-guess game

The path alias `@/*` resolves to `src/*` (configured in both `vite.config.ts` and `tsconfig.json`).

### Data flow
All demo data is **precomputed static JSON** served from `public/data/` and fetched at runtime via `src/lib/data.ts` (module-level memoized caches). Core files:
- `demo_cases.json` — 14 cases (best/medium/hard/mystery)
- `pipeline_cases.json` — optional, used by `/pipeline` if present, falls back to `demo_cases.json`
- `roi_layout.json`, `clip_projection.json`, `metrics_summary.json`

`src/lib/api.ts` is the live-backend client; `src/lib/pipelineNormalize.ts` reconciles live backend responses into the same shape as cached replay JSON.

### Provenance system (critical invariant)
Every scientific value, image, and metric carries a `Provenance` (`src/lib/provenance.ts`) with kind `live | replay | derived | placeholder | unknown`. The Pipeline page renders provenance badges everywhere. **Never display synthetic or placeholder scientific data as if it were real** — unavailable artifacts must show an honest "unavailable" state (`—`) or be hidden. This is enforced visually via `ProvenanceBadge` and conceptually throughout `src/components/pipeline/*`.

Modes shown in `PipelineStatusHeader`: Live (backend in this session), Replay (cached JSON), Offline replay (backend unreachable), Hybrid (backend up but cached assets active).

### Component organization (`src/components/`)
Grouped by feature surface: `brain/` (Three.js 3D brain + 17 ROI markers), `film/`, `explorer/`, `pipeline/` (the honest decoding UI), `premium/`, `challenge/`, `manifold/`, `uncertainty/`, plus shared primitives at the top level (`GlassCard`, `Layout`, `TabBar`, etc.).

### Heavy deps and chunking
`vite.config.ts` manually splits `plotly`, `three`, and `framer-motion` into separate chunks — keep imports tree-shake-friendly and avoid importing these from `src/lib/` shared modules unless intended.

### Backend (`backend/main.py`)
Single FastAPI file. Auto-injects `../../src` (the parent project's `fmri2img` package) into `sys.path`. Endpoints under `/api/*`: `health`, `trials`, `infer/{idx}`, `infer-stream/{idx}` (SSE), `nsd-id-to-trial/{nsd_id}`. Reconstructions are never fabricated — the backend only reports them as available when truly produced live.

## Conventions specific to this repo

- **TypeScript strict mode is on** but `noUnusedLocals` / `noUnusedParameters` are off — don't add lint cleanup that fights the existing style.
- Vite's file watcher ignores `results/`, `local_backend_data/`, `dist/`, etc. — large model artifacts live there and must not be imported from `src/`.
- The "scientific honesty" rule (provenance + no fake placeholders) overrides UI polish concerns. When adding any new metric/visualization to the Pipeline page, wire it through `Provenance` and the badge system before considering it done.
- Pipeline replay assets are refreshed from the K8s pod via `scripts/fetch_live_backend_artifacts_from_pod.sh` + `scripts/integrate_real_artifacts.py`.
