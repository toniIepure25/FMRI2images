import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { Home } from '@/pages/Home';
import { LoadingState } from '@/components/LoadingState';

const Film = lazy(() => import('@/pages/Film').then((m) => ({ default: m.Film })));
const Explorer = lazy(() => import('@/pages/Explorer').then((m) => ({ default: m.Explorer })));
const Pipeline = lazy(() => import('@/pages/Pipeline').then((m) => ({ default: m.Pipeline })));
const Evidence = lazy(() => import('@/pages/Evidence').then((m) => ({ default: m.Evidence })));
const Challenge = lazy(() => import('@/pages/Challenge').then((m) => ({ default: m.Challenge })));
const NeuralManifoldExplorer = lazy(() =>
  import('@/pages/NeuralManifoldExplorer').then((m) => ({ default: m.NeuralManifoldExplorer })),
);

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/film" element={<Suspense fallback={<LoadingState />}><Film /></Suspense>} />
          <Route path="/explorer" element={<Suspense fallback={<LoadingState />}><Explorer /></Suspense>} />
          <Route path="/explorer/:caseId" element={<Suspense fallback={<LoadingState />}><Explorer /></Suspense>} />
          <Route path="/pipeline" element={<Suspense fallback={<LoadingState />}><Pipeline /></Suspense>} />
          <Route path="/evidence" element={<Suspense fallback={<LoadingState />}><Evidence /></Suspense>} />
          <Route path="/manifold" element={<Suspense fallback={<LoadingState />}><NeuralManifoldExplorer /></Suspense>} />
          <Route path="/challenge" element={<Suspense fallback={<LoadingState />}><Challenge /></Suspense>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
