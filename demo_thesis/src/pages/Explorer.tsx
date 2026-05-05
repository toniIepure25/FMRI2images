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
  { id: 'overview', label: 'Overview', icon: '◈' },
  { id: 'brain', label: '3D Brain', icon: '🧠' },
  { id: 'clip', label: 'CLIP Space', icon: '◎' },
  { id: 'retrieval', label: 'Retrieval', icon: '🔍' },
  { id: 'reconstruction', label: 'Reconstruction', icon: '🎨' },
  { id: 'uncertainty', label: 'Uncertainty', icon: '📊' },
  { id: 'roi', label: 'ROI', icon: '🗺' },
  { id: 'report', label: 'Report', icon: '📄' },
];

export function Explorer() {
  const { caseId } = useParams();
  const navigate = useNavigate();
  const { cases, loading, error } = useCases();
  const [selectedCase, setSelectedCase] = useState<DemoCase | null>(null);
  const [activeTab, setActiveTab] = useState<ExplorerTab>('overview');

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
    <div className="flex h-[calc(100vh-3.5rem)]">
      <div className="w-72 flex-shrink-0 overflow-y-auto border-r border-brain-border/30 p-4">
        <h2 className="mb-4 text-sm font-semibold text-gray-300">Cases</h2>
        <CaseSelector cases={cases} selectedId={selectedCase?.id} onSelect={handleCaseSelect} />
      </div>

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="border-b border-brain-border/30 p-3">
          <TabBar tabs={TABS} activeTab={activeTab} onChange={(id) => setActiveTab(id as ExplorerTab)} />
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
        <div className="w-64 flex-shrink-0 overflow-y-auto border-l border-brain-border/30 p-4">
          <QuickStats case_={selectedCase} />
        </div>
      )}
    </div>
  );
}
