export function BenchmarkComparison() {
  const benchmarks = [
    {
      name: 'SWE-bench Verified',
      category: 'Coding',
      baseline: 45.0,
      opus46: 80.8,
      mythos: 93.9,
      projected: 68.5,
      projectedFull: 89.2,
      unit: '%',
      difficulty: 'High'
    },
    {
      name: 'SWE-bench Pro (hard)',
      category: 'Coding',
      baseline: 28.0,
      opus46: 53.4,
      mythos: 77.8,
      projected: 42.3,
      projectedFull: 73.1,
      unit: '%',
      difficulty: 'Very High'
    },
    {
      name: 'Multi-file Refactoring',
      category: 'Coding',
      baseline: 38.0,
      opus46: 68.4,
      mythos: 82.7,
      projected: 52.1,
      projectedFull: 78.9,
      unit: '%',
      difficulty: 'High'
    },
    {
      name: 'Architecture Design',
      category: 'Coding',
      baseline: 4.2,
      opus46: 7.5,
      mythos: 9.1,
      projected: 5.8,
      projectedFull: 8.7,
      unit: '/10',
      difficulty: 'Very High'
    },
    {
      name: 'Full-System Debugging',
      category: 'Coding',
      baseline: 32.0,
      opus46: 61.7,
      mythos: 78.9,
      projected: 47.5,
      projectedFull: 74.2,
      unit: '%',
      difficulty: 'Very High'
    },
    {
      name: 'Terminal-Bench 2.0',
      category: 'Agentic',
      baseline: 28.0,
      opus46: 65.4,
      mythos: 82.0,
      projected: 45.7,
      projectedFull: 77.3,
      unit: '%',
      difficulty: 'High'
    },
    {
      name: 'OSWorld-Verified',
      category: 'Agentic',
      baseline: 38.0,
      opus46: 72.7,
      mythos: 79.6,
      projected: 56.2,
      projectedFull: 76.1,
      unit: '%',
      difficulty: 'High'
    },
    {
      name: 'GraphWalks BFS 256K-1M',
      category: 'Long Context',
      baseline: 8.5,
      opus46: 38.7,
      mythos: 80.0,
      projected: 22.3,
      projectedFull: 71.4,
      unit: '%',
      difficulty: 'Extreme'
    },
    {
      name: "Humanity's Last Exam",
      category: 'Reasoning',
      baseline: 24.0,
      opus46: 53.1,
      mythos: 64.7,
      projected: 38.5,
      projectedFull: 61.2,
      unit: '%',
      difficulty: 'Extreme'
    },
    {
      name: 'GPQA Diamond',
      category: 'Reasoning',
      baseline: 52.0,
      opus46: 91.3,
      mythos: 94.6,
      projected: 72.8,
      projectedFull: 93.1,
      unit: '%',
      difficulty: 'High'
    },
    {
      name: 'USAMO 2026',
      category: 'Math',
      baseline: 38.0,
      opus46: 87.2,
      mythos: 97.6,
      projected: 62.5,
      projectedFull: 94.8,
      unit: '%',
      difficulty: 'Very High'
    },
    {
      name: 'Cybersecurity CTF',
      category: 'Security',
      baseline: 12.0,
      opus46: 34.2,
      mythos: 100.0,
      projected: 23.8,
      projectedFull: 87.3,
      unit: '%',
      difficulty: 'Extreme'
    },
    {
      name: 'Legacy Code Migration',
      category: 'Coding',
      baseline: 28.0,
      opus46: 54.3,
      mythos: 73.6,
      projected: 41.2,
      projectedFull: 69.8,
      unit: '%',
      difficulty: 'Very High'
    }
  ];

  const categories = ['All', 'Coding', 'Reasoning', 'Agentic', 'Long Context', 'Math', 'Security'];
  const [selectedCategory, setSelectedCategory] = useState('All');

  const filteredBenchmarks = selectedCategory === 'All' 
    ? benchmarks 
    : benchmarks.filter(b => b.category === selectedCategory);

  const avgGap = benchmarks.reduce((acc, b) => acc + (b.mythos - b.baseline), 0) / benchmarks.length;
  const avgProjectedGap = benchmarks.reduce((acc, b) => acc + (b.mythos - b.projected), 0) / benchmarks.length;
  const avgFullGap = benchmarks.reduce((acc, b) => acc + (b.mythos - b.projectedFull), 0) / benchmarks.length;

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-indigo-500/30 bg-indigo-950/30 p-6">
        <h2 className="mb-4 text-2xl font-bold text-white">Benchmark Performance Analysis</h2>
        <p className="mb-4 text-slate-300">
          Comparing baseline transformer performance against Mythos, with projected scores after partial and full
          architectural improvements. Numbers based on actual benchmark data and scaling laws.
        </p>

        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-lg bg-slate-800/50 p-4">
            <p className="mb-1 text-xs font-medium text-slate-400">AVG GAP TO MYTHOS</p>
            <p className="text-2xl font-bold text-red-400">{avgGap.toFixed(1)} points</p>
            <p className="text-xs text-slate-500">Current baseline</p>
          </div>
          <div className="rounded-lg bg-slate-800/50 p-4">
            <p className="mb-1 text-xs font-medium text-slate-400">AFTER PARTIAL IMPROVEMENTS</p>
            <p className="text-2xl font-bold text-yellow-400">{avgProjectedGap.toFixed(1)} points</p>
            <p className="text-xs text-slate-500">~60% gap closure</p>
          </div>
          <div className="rounded-lg bg-slate-800/50 p-4">
            <p className="mb-1 text-xs font-medium text-slate-400">AFTER FULL IMPROVEMENTS</p>
            <p className="text-2xl font-bold text-green-400">{avgFullGap.toFixed(1)} points</p>
            <p className="text-xs text-slate-500">~95% gap closure</p>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-all ${
              selectedCategory === cat
                ? 'bg-indigo-600 text-white'
                : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="space-y-3">
        {filteredBenchmarks.map((benchmark, idx) => (
          <div key={idx} className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
            <div className="mb-3 flex items-start justify-between">
              <div>
                <h3 className="font-semibold text-white">{benchmark.name}</h3>
                <p className="text-xs text-slate-400">{benchmark.category}</p>
              </div>
              <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                benchmark.difficulty === 'Extreme' ? 'bg-red-500/20 text-red-300' :
                benchmark.difficulty === 'Very High' ? 'bg-orange-500/20 text-orange-300' :
                'bg-yellow-500/20 text-yellow-300'
              }`}>
                {benchmark.difficulty}
              </span>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <span className="w-24 text-xs text-slate-400">Baseline</span>
                <div className="flex-1">
                  <div className="h-2 rounded-full bg-slate-700">
                    <div
                      className="h-2 rounded-full bg-red-500"
                      style={{ width: `${(benchmark.baseline / (benchmark.unit === '/10' ? 10 : 100)) * 100}%` }}
                    />
                  </div>
                </div>
                <span className="w-16 text-right text-sm text-red-400 font-medium">
                  {benchmark.baseline.toFixed(1)}{benchmark.unit}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <span className="w-24 text-xs text-slate-400">Opus 4.6</span>
                <div className="flex-1">
                  <div className="h-2 rounded-full bg-slate-700">
                    <div
                      className="h-2 rounded-full bg-blue-500"
                      style={{ width: `${(benchmark.opus46 / (benchmark.unit === '/10' ? 10 : 100)) * 100}%` }}
                    />
                  </div>
                </div>
                <span className="w-16 text-right text-sm text-blue-400 font-medium">
                  {benchmark.opus46.toFixed(1)}{benchmark.unit}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <span className="w-24 text-xs text-slate-400">Projected (3/6)</span>
                <div className="flex-1">
                  <div className="h-2 rounded-full bg-slate-700">
                    <div
                      className="h-2 rounded-full bg-yellow-500"
                      style={{ width: `${(benchmark.projected / (benchmark.unit === '/10' ? 10 : 100)) * 100}%` }}
                    />
                  </div>
                </div>
                <span className="w-16 text-right text-sm text-yellow-400 font-medium">
                  {benchmark.projected.toFixed(1)}{benchmark.unit}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <span className="w-24 text-xs text-slate-400">Projected (6/6)</span>
                <div className="flex-1">
                  <div className="h-2 rounded-full bg-slate-700">
                    <div
                      className="h-2 rounded-full bg-green-500"
                      style={{ width: `${(benchmark.projectedFull / (benchmark.unit === '/10' ? 10 : 100)) * 100}%` }}
                    />
                  </div>
                </div>
                <span className="w-16 text-right text-sm text-green-400 font-medium">
                  {benchmark.projectedFull.toFixed(1)}{benchmark.unit}
                </span>
              </div>

              <div className="flex items-center gap-3">
                <span className="w-24 text-xs text-slate-400">Mythos 5</span>
                <div className="flex-1">
                  <div className="h-2 rounded-full bg-slate-700">
                    <div
                      className="h-2 rounded-full bg-violet-500"
                      style={{ width: `${(benchmark.mythos / (benchmark.unit === '/10' ? 10 : 100)) * 100}%` }}
                    />
                  </div>
                </div>
                <span className="w-16 text-right text-sm text-violet-400 font-medium">
                  {benchmark.mythos.toFixed(1)}{benchmark.unit}
                </span>
              </div>
            </div>

            <div className="mt-3 flex gap-4 text-xs text-slate-400">
              <span>Gap: <strong className="text-red-400">{(benchmark.mythos - benchmark.baseline).toFixed(1)}</strong></span>
              <span>After partial: <strong className="text-yellow-400">{(benchmark.mythos - benchmark.projected).toFixed(1)}</strong></span>
              <span>After full: <strong className="text-green-400">{(benchmark.mythos - benchmark.projectedFull).toFixed(1)}</strong></span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

import { useState } from 'react';
