import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import type { DemoCase, ExplorerTab } from '@/types';
import { useCases } from '@/lib/hooks';
import { CaseSelector } from '@/components/CaseSelector';
import { TabBar } from '@/components/TabBar';
import { LoadingState } from '@/components/LoadingState';
import { ErrorState } from '@/components/ErrorState';
import { OverviewTab } from '@/components/explorer/OverviewTab';
import { BrainTab } from '@/components/explorer/BrainTab';
import { ClipTab } from '@/components/explorer/ClipTab';
import { RetrievalTab } from '@/components/explorer/RetrievalTab';
import { ReconstructionTab } from '@/components/explorer/ReconstructionTab';
import { UncertaintyTab } from '@/components/explorer/UncertaintyTab';
import { RoiTab } from '@/components/explorer/RoiTab';
import { ReportTab } from '@/components/explorer/ReportTab';
import { QuickStats } from '@/components/explorer/QuickStats';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'brain', label: 'Brain' },
  { id: 'clip', label: 'CLIP' },
  { id: 'retrieval', label: 'Retrieval' },
  { id: 'reconstruction', label: 'Recon' },
  { id: 'uncertainty', label: 'Uncertainty' },
  { id: 'roi', label: 'ROI' },
  { id: 'report', label: 'Report' },
];

export function Explorer() {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const { cases, loading, error } = useCases();
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [activeTab, setActiveTab] = useState<ExplorerTab>('overview');
  const [casesSidebarOpen, setCasesSidebarOpen] = useState(false);

  useEffect(() => {
    if (cases.length === 0) return;
    if (caseId) {
      const found = cases.find((c) => c.id === caseId);
      if (found) {
        setSelectedCase(found);
      } else {
        setSelectedCase(cases[0]);
        navigate(`/explorer/${cases[0].id}`, { replace: true });
      }
    } else {
      setSelectedCase(cases[0]);
      navigate(`/explorer/${cases[0].id}`, { replace: true });
    }
  }, [cases, caseId, navigate]);

  const handleCaseSelect = (c: DemoCase) => {
    setSelectedCase(c);
    navigate(`/explorer/${c.id}`, { replace: true });
    setCasesSidebarOpen(false);
  };

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;

  const renderTab = () => {
    if (!selectedCase) return null;
    switch (activeTab) {
      case 'overview':
        return <OverviewTab case_={selectedCase} />;
      case 'brain':
        return <BrainTab case_={selectedCase} />;
      case 'clip':
        return <ClipTab case_={selectedCase} />;
      case 'retrieval':
        return <RetrievalTab case_={selectedCase} />;
      case 'reconstruction':
        return <ReconstructionTab case_={selectedCase} />;
      case 'uncertainty':
        return <UncertaintyTab case_={selectedCase} />;
      case 'roi':
        return <RoiTab case_={selectedCase} />;
      case 'report':
        return <ReportTab case_={selectedCase} />;
      default:
        return null;
    }
  };

  return (
    <div className="relative flex h-[calc(100vh-3.5rem)]">
      {casesSidebarOpen ? (
        <button
          type="button"
          aria-label="Close cases list"
          className="fixed inset-0 z-30 bg-slate-950/70 backdrop-blur-[2px] lg:hidden"
          onClick={() => setCasesSidebarOpen(false)}
        />
      ) : null}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-shrink-0 flex-col overflow-y-auto border-r border-brain-border/30 bg-slate-950/98 p-4 shadow-xl backdrop-blur-md transition-transform duration-300 ease-out lg:static lg:z-auto lg:bg-transparent lg:shadow-none lg:backdrop-blur-none ${
          casesSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        <div className="mb-4 flex items-center justify-between lg:hidden">
          <h2 className="text-sm font-semibold text-slate-200">Cases</h2>
          <button
            type="button"
            onClick={() => setCasesSidebarOpen(false)}
            className="rounded-lg border border-slate-600/80 bg-slate-900/80 px-2.5 py-1 text-xs font-medium text-slate-300 transition hover:border-slate-500 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent"
          >
            Close
          </button>
        </div>
        <h2 className="mb-4 hidden text-sm font-semibold text-slate-200 lg:block">Cases</h2>
        <CaseSelector cases={cases} selectedId={selectedCase?.id} onSelect={handleCaseSelect} />
      </aside>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex items-stretch gap-2 border-b border-brain-border/30 p-3">
          <button
            type="button"
            onClick={() => setCasesSidebarOpen(true)}
            className="shrink-0 rounded-lg border border-brain-border/50 bg-slate-900/70 px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-brain-accent/35 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brain-accent lg:hidden"
          >
            Cases
          </button>
          <div className="min-w-0 flex-1 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
            <TabBar
              layoutIdPrefix="explorer"
              tabs={TABS}
              activeTab={activeTab}
              onChange={(id) => setActiveTab(id as ExplorerTab)}
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2 }}
          >
            {renderTab()}
          </motion.div>
        </div>
      </div>

      {selectedCase && (
        <div className="hidden w-64 flex-shrink-0 overflow-y-auto border-l border-brain-border/30 p-4 xl:block">
          <QuickStats case_={selectedCase} />
        </div>
      )}
    </div>
  );
}
