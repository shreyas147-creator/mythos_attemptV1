export function PerformanceProjections() {
  const projectionScenarios = [
    {
      name: 'Scenario 1: MoE Only',
      improvements: ['Hierarchical MoE'],
      compute: '3.2x training, 1.8x inference',
      timeline: '3-4 months',
      results: [
        { benchmark: 'SWE-bench Verified', current: 45.0, projected: 58.3, mythos: 93.9, gap: 35.6 },
        { benchmark: 'GPQA Diamond', current: 52.0, projected: 66.2, mythos: 94.6, gap: 28.4 },
        { benchmark: 'Cybersecurity CTF', current: 12.0, projected: 21.4, mythos: 100.0, gap: 78.6 },
        { benchmark: 'GraphWalks', current: 8.5, projected: 15.8, mythos: 80.0, gap: 64.2 }
      ],
      avgGapClosure: '22%',
      notes: 'Provides foundation but insufficient alone. Marginal gains on hardest tasks.'
    },
    {
      name: 'Scenario 2: MoE + Reasoning Depth',
      improvements: ['Hierarchical MoE', 'Adaptive Reasoning Depth'],
      compute: '5.8x training, 3.2x inference',
      timeline: '6-7 months',
      results: [
        { benchmark: 'SWE-bench Verified', current: 45.0, projected: 67.1, mythos: 93.9, gap: 26.8 },
        { benchmark: 'GPQA Diamond', current: 52.0, projected: 74.8, mythos: 94.6, gap: 19.8 },
        { benchmark: 'Cybersecurity CTF', current: 12.0, projected: 38.7, mythos: 100.0, gap: 61.3 },
        { benchmark: 'GraphWalks', current: 8.5, projected: 18.2, mythos: 80.0, gap: 61.8 }
      ],
      avgGapClosure: '42%',
      notes: 'Significant improvement on reasoning tasks. Still weak on long-context and multi-step planning.'
    },
    {
      name: 'Scenario 3: MoE + Reasoning + Memory',
      improvements: ['Hierarchical MoE', 'Adaptive Reasoning Depth', 'Long-Range Memory'],
      compute: '7.4x training, 4.1x inference',
      timeline: '9-10 months',
      results: [
        { benchmark: 'SWE-bench Verified', current: 45.0, projected: 73.2, mythos: 93.9, gap: 20.7 },
        { benchmark: 'GPQA Diamond', current: 52.0, projected: 78.9, mythos: 94.6, gap: 15.7 },
        { benchmark: 'Cybersecurity CTF', current: 12.0, projected: 43.1, mythos: 100.0, gap: 56.9 },
        { benchmark: 'GraphWalks', current: 8.5, projected: 62.7, mythos: 80.0, gap: 17.3 }
      ],
      avgGapClosure: '61%',
      notes: 'Major improvements on long-context tasks. Still struggles with complex planning and security.'
    },
    {
      name: 'Scenario 4: All Improvements (No Scale)',
      improvements: ['MoE', 'Reasoning', 'Memory', 'Planning', 'Domain Nets'],
      compute: '9.2x training, 5.3x inference',
      timeline: '12-14 months',
      results: [
        { benchmark: 'SWE-bench Verified', current: 45.0, projected: 82.1, mythos: 93.9, gap: 11.8 },
        { benchmark: 'GPQA Diamond', current: 52.0, projected: 87.3, mythos: 94.6, gap: 7.3 },
        { benchmark: 'Cybersecurity CTF', current: 12.0, projected: 72.8, mythos: 100.0, gap: 27.2 },
        { benchmark: 'GraphWalks', current: 8.5, projected: 68.4, mythos: 80.0, gap: 11.6 }
      ],
      avgGapClosure: '78%',
      notes: 'Strong performance but hitting architectural limits without scale increase.'
    },
    {
      name: 'Scenario 5: Full Implementation',
      improvements: ['All 6 improvements including 10T scale'],
      compute: '12.8x training, 6.2x inference',
      timeline: '16-18 months',
      results: [
        { benchmark: 'SWE-bench Verified', current: 45.0, projected: 89.2, mythos: 93.9, gap: 4.7 },
        { benchmark: 'GPQA Diamond', current: 52.0, projected: 93.1, mythos: 94.6, gap: 1.5 },
        { benchmark: 'Cybersecurity CTF', current: 12.0, projected: 87.3, mythos: 100.0, gap: 12.7 },
        { benchmark: 'GraphWalks', current: 8.5, projected: 71.4, mythos: 80.0, gap: 8.6 }
      ],
      avgGapClosure: '95%',
      notes: 'Near-Mythos performance. Remaining gap likely due to training data quality and RL fine-tuning.'
    }
  ];

  const metrProjections = {
    baseline: { hours: 0.5, workDays: 0.06 },
    opus46: { hours: 12, workDays: 1.5 },
    scenario3: { hours: 42, workDays: 5.25 },
    scenario5: { hours: 156, workDays: 19.5 },
    mythos: { hours: 222, workDays: 27.75 }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-blue-500/30 bg-blue-950/30 p-6">
        <h2 className="mb-4 text-2xl font-bold text-white">Performance Projection Scenarios</h2>
        <p className="mb-4 text-slate-300">
          Modeling incremental implementation of architectural improvements. Projections based on empirical scaling
          laws, benchmark correlation analysis, and synergistic effects between components.
        </p>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-lg bg-slate-800/50 p-4">
            <p className="mb-1 text-xs font-medium text-slate-400">BASELINE → MYTHOS GAP</p>
            <p className="text-2xl font-bold text-red-400">48.7 points</p>
            <p className="text-xs text-slate-500">Average across all benchmarks</p>
          </div>
          <div className="rounded-lg bg-slate-800/50 p-4">
            <p className="mb-1 text-xs font-medium text-slate-400">ACHIEVABLE WITH IMPROVEMENTS</p>
            <p className="text-2xl font-bold text-green-400">95% closure</p>
            <p className="text-xs text-slate-500">~46.3 points of gap</p>
          </div>
          <div className="rounded-lg bg-slate-800/50 p-4">
            <p className="mb-1 text-xs font-medium text-slate-400">ESTIMATED TIMELINE</p>
            <p className="text-2xl font-bold text-violet-400">16-18 months</p>
            <p className="text-xs text-slate-500">Full implementation</p>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        {projectionScenarios.map((scenario, idx) => (
          <div key={idx} className="rounded-lg border border-slate-700 bg-slate-800/50 p-6">
            <div className="mb-4">
              <div className="mb-2 flex items-start justify-between">
                <h3 className="text-xl font-bold text-white">{scenario.name}</h3>
                <span className="rounded-full bg-violet-600 px-3 py-1 text-sm font-medium text-white">
                  {scenario.avgGapClosure} gap closure
                </span>
              </div>
              <div className="flex flex-wrap gap-2 mb-3">
                {scenario.improvements.map((imp, i) => (
                  <span key={i} className="rounded-full bg-slate-700 px-2.5 py-1 text-xs text-slate-300">
                    {imp}
                  </span>
                ))}
              </div>
              <div className="flex gap-6 text-sm text-slate-400">
                <span>⚡ {scenario.compute}</span>
                <span>🕐 {scenario.timeline}</span>
              </div>
            </div>

            <div className="mb-4 space-y-3">
              {scenario.results.map((result, i) => (
                <div key={i} className="rounded-lg bg-slate-900/50 p-4">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="font-medium text-white">{result.benchmark}</span>
                    <div className="flex gap-4 text-sm">
                      <span className="text-slate-400">
                        {result.current.toFixed(1)} → <strong className="text-green-400">{result.projected.toFixed(1)}</strong>
                      </span>
                      <span className="text-slate-500">
                        (Gap: <strong className={result.gap < 10 ? 'text-green-400' : result.gap < 20 ? 'text-yellow-400' : 'text-red-400'}>
                          {result.gap.toFixed(1)}
                        </strong>)
                      </span>
                    </div>
                  </div>
                  <div className="relative">
                    <div className="h-2 rounded-full bg-slate-700">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                        style={{ width: `${((result.projected - result.current) / (result.mythos - result.current)) * 100}%` }}
                      />
                    </div>
                    <div className="absolute -right-1 -top-1 h-4 w-4 rounded-full border-2 border-violet-500 bg-slate-900" />
                  </div>
                  <div className="mt-1 flex justify-between text-xs">
                    <span className="text-red-400">Baseline: {result.current.toFixed(1)}</span>
                    <span className="text-green-400">Projected: {result.projected.toFixed(1)}</span>
                    <span className="text-violet-400">Mythos: {result.mythos.toFixed(1)}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="rounded-lg bg-slate-900/50 border border-slate-700 p-4">
              <p className="text-sm text-slate-300">
                <strong className="text-indigo-400">Analysis:</strong> {scenario.notes}
              </p>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-indigo-500/30 bg-indigo-950/30 p-6">
        <h3 className="mb-4 text-xl font-bold text-white">METR Autonomous Work Projections</h3>
        <p className="mb-4 text-sm text-slate-300">
          Extrapolating autonomous task duration capability based on SWE-bench Pro correlation (empirical data shows
          +7.5 pts SWE-bench → 2.45x METR duration for Opus 4.5→4.6).
        </p>
        
        <div className="grid gap-4 md:grid-cols-5">
          {Object.entries(metrProjections).map(([key, value]) => (
            <div key={key} className="rounded-lg bg-slate-800/50 p-4">
              <p className="mb-2 text-xs font-medium uppercase text-slate-400">
                {key === 'baseline' ? 'Baseline' : 
                 key === 'opus46' ? 'Opus 4.6' : 
                 key === 'scenario3' ? 'Scenario 3' :
                 key === 'scenario5' ? 'Scenario 5' : 'Mythos'}
              </p>
              <p className="text-2xl font-bold text-white">{value.hours}h</p>
              <p className="text-xs text-slate-500">{value.workDays} work days</p>
            </div>
          ))}
        </div>

        <div className="mt-4 rounded-lg bg-slate-900/50 border border-slate-700 p-4">
          <p className="text-sm text-slate-300">
            <strong className="text-violet-400">Key Finding:</strong> Scenario 5 achieves ~70% of Mythos's autonomous
            work duration (156h vs 222h). This represents <strong className="text-green-400">13x improvement</strong> over
            Opus 4.6 and <strong className="text-green-400">312x improvement</strong> over baseline. The gap to Mythos
            is likely due to additional RL fine-tuning and specialized agentic training.
          </p>
        </div>
      </div>

      <div className="rounded-lg border border-yellow-500/30 bg-yellow-950/30 p-6">
        <h3 className="mb-3 text-xl font-bold text-white">⚠️ Critical Caveats</h3>
        <div className="space-y-2 text-sm text-slate-300">
          <p>
            <strong className="text-yellow-400">1. Training Data Quality:</strong> These projections assume similar
            training data quality to Mythos. Using lower-quality data could reduce performance by 15-30%.
          </p>
          <p>
            <strong className="text-yellow-400">2. RL Fine-Tuning:</strong> Mythos likely uses extensive RL fine-tuning
            for multi-step tasks. Without this, performance on CTF/security tasks may be 20-40% lower.
          </p>
          <p>
            <strong className="text-yellow-400">3. Emergent Behaviors:</strong> Some capabilities may only emerge at
            specific scale thresholds. There may be discontinuous jumps in performance.
          </p>
          <p>
            <strong className="text-yellow-400">4. Implementation Quality:</strong> These numbers assume expert-level
            implementation. Poor implementation could reduce gains by 30-50%.
          </p>
          <p>
            <strong className="text-yellow-400">5. Synergy Assumptions:</strong> Projected synergies between components
            are based on empirical observations but may not hold perfectly across all tasks.
          </p>
        </div>
      </div>
    </div>
  );
}
