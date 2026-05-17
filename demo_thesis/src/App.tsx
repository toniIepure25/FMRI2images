import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { Home } from '@/pages/Home';
import { Film } from '@/pages/Film';
import { Explorer } from '@/pages/Explorer';
import { Pipeline } from '@/pages/Pipeline';
import { Evidence } from '@/pages/Evidence';
import { Challenge } from '@/pages/Challenge';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Home />} />
          <Route path="/film" element={<Film />} />
          <Route path="/explorer" element={<Explorer />} />
          <Route path="/explorer/:caseId" element={<Explorer />} />
          <Route path="/pipeline" element={<Pipeline />} />
          <Route path="/evidence" element={<Evidence />} />
          <Route path="/challenge" element={<Challenge />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
