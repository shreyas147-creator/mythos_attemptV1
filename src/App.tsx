import { useState } from 'react';
import { ArchitectureAnalysis } from './components/ArchitectureAnalysis';
import { BenchmarkComparison } from './components/BenchmarkComparison';
import { ArchitecturalImprovements } from './components/ArchitecturalImprovements';
import { PerformanceProjections } from './components/PerformanceProjections';
import { DeepDive } from './components/DeepDive';

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'benchmarks' | 'improvements' | 'projections' | 'deepdive'>('overview');

  const tabs = [
    { id: 'overview', label: 'Architecture Analysis', icon: '🏗️' },
    { id: 'benchmarks', label: 'Benchmark Gaps', icon: '📊' },
    { id: 'improvements', label: 'Critical Improvements', icon: '⚡' },
    { id: 'projections', label: 'Performance Projections', icon: '📈' },
    { id: 'deepdive', label: 'Deep Technical Dive', icon: '🔬' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="mb-4 inline-flex items-center gap-3 rounded-2xl bg-gradient-to-r from-violet-600/20 to-indigo-600/20 px-6 py-3 backdrop-blur-sm">
            <div className="text-4xl">🧠</div>
            <div>
              <h1 className="text-2xl font-bold text-white">Mythos-Level Architecture Analysis</h1>
              <p className="text-sm text-slate-400">Deep Performance & Architectural Improvement Roadmap</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <div className="mb-6 flex flex-wrap justify-center gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                activeTab === tab.id
                  ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-lg shadow-violet-500/50'
                  : 'bg-slate-800/50 text-slate-300 hover:bg-slate-700/50'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="rounded-2xl bg-slate-900/50 p-6 backdrop-blur-sm">
          {activeTab === 'overview' && <ArchitectureAnalysis />}
          {activeTab === 'benchmarks' && <BenchmarkComparison />}
          {activeTab === 'improvements' && <ArchitecturalImprovements />}
          {activeTab === 'projections' && <PerformanceProjections />}
          {activeTab === 'deepdive' && <DeepDive />}
        </div>
      </div>
    </div>
  );
}
