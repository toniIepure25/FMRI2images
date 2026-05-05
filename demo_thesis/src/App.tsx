import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { Layout } from '@/components/Layout';
import { Home } from '@/pages/Home';
import { Film } from '@/pages/Film';
import { Explorer } from '@/pages/Explorer';
import { Challenge } from '@/pages/Challenge';
import { Pipeline } from '@/pages/Pipeline';

export default function App() {
  return (
    <BrowserRouter>
      <AnimatePresence mode="wait">
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Home />} />
            <Route path="/film" element={<Film />} />
            <Route path="/explorer" element={<Explorer />} />
            <Route path="/explorer/:caseId" element={<Explorer />} />
            <Route path="/challenge" element={<Challenge />} />
            <Route path="/pipeline" element={<Pipeline />} />
          </Route>
        </Routes>
      </AnimatePresence>
    </BrowserRouter>
  );
}
