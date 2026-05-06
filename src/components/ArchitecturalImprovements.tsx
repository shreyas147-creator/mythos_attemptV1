export function ArchitecturalImprovements() {
  const improvements = [
    {
      id: 1,
      name: 'Hierarchical Mixture-of-Experts (HMoE)',
      priority: 'CRITICAL',
      complexity: 'Very High',
      expectedGain: '+18-25 points avg',
      compute: '3.2x training, 1.8x inference',
      implementation: [
        'Replace monolithic expert routing with 3-level hierarchy',
        'Level 1: 16 macro-experts (domain clusters: code, math, reasoning, language)',
        'Level 2: 128 micro-experts per macro (specialized sub-tasks)',
        'Level 3: 512 nano-experts (ultra-specific patterns)',
        'Implement learned routing with gradient-based optimization',
        'Add expert capacity balancing to prevent collapse',
        'Use soft routing (top-k=4 at each level) with learnable mixing weights'
      ],
      technicalDetails: {
        architecture: 'Sparse gating with auxiliary loss, capacity factor 1.25',
        routing: 'Noisy top-k with jitter, expert dropout 0.1',
        parameterization: 'Total: ~8.5T params, Active: ~1.1T per forward pass',
        training: 'Two-stage: (1) Expert pre-specialization (2) End-to-end fine-tuning',
        memory: 'Distributed expert placement, ZeRO-3 + pipeline parallelism'
      },
      benchmarkImpact: [
        { name: 'SWE-bench Verified', from: 45.0, to: 64.2, mythos: 93.9 },
        { name: 'Multi-file Refactoring', from: 38.0, to: 58.7, mythos: 82.7 },
        { name: 'GPQA Diamond', from: 52.0, to: 69.3, mythos: 94.6 },
        { name: 'Legacy Code Migration', from: 28.0, to: 44.8, mythos: 73.6 }
      ]
    },
    {
      id: 2,
      name: 'Adaptive Reasoning Depth (ARD)',
      priority: 'CRITICAL',
      complexity: 'Very High',
      expectedGain: '+12-18 points avg',
      compute: '2.1x training, 2.3x inference (adaptive)',
      implementation: [
        'Add parallel reasoning heads with dynamic depth allocation',
        'Implement confidence-based early exit mechanisms',
        'Use value networks to estimate solution quality at each depth',
        'Add recursive self-refinement loops (max 128 iterations)',
        'Implement tree-of-thought search with beam width 8-32',
        'Use neural Monte Carlo Tree Search for complex problems',
        'Add explicit backtracking and hypothesis revision mechanisms'
      ],
      technicalDetails: {
        architecture: '32 reasoning layers post-attention, variable depth 1-128',
        confidenceModel: 'Separate value head trained on solution correctness',
        searchStrategy: 'Hybrid beam search + MCTS, adaptive expansion',
        earlyExit: 'Entropy-based thresholding, calibrated confidence scores',
        refinement: 'Self-consistency checking, contradiction detection'
      },
      benchmarkImpact: [
        { name: "Humanity's Last Exam", from: 45.0, to: 57.3, mythos: 64.7 },
        { name: 'USAMO 2026', from: 62.5, to: 83.4, mythos: 97.6 },
        { name: 'Architecture Design', from: 4.2, to: 6.8, mythos: 9.1 },
        { name: 'Cybersecurity CTF', from: 23.8, to: 52.1, mythos: 100.0 }
      ]
    },
    {
      id: 3,
      name: 'Long-Range Memory Architecture (LRMA)',
      priority: 'CRITICAL',
      complexity: 'Extreme',
      expectedGain: '+28-45 points (long-context tasks)',
      compute: '1.4x training, 1.6x inference',
      implementation: [
        'Implement hierarchical memory compression (4-level pyramid)',
        'Level 1: Full attention (32K tokens)',
        'Level 2: Compressed chunks (256K tokens, 8:1 ratio)',
        'Level 3: Abstract summaries (2M tokens, 64:1 ratio)',
        'Level 4: Extreme compression (16M tokens, 512:1 ratio)',
        'Add learned compression networks at each level',
        'Use cross-attention to retrieve from compressed memories',
        'Implement surprise-based retention (keep unexpected info)',
        'Add explicit read/write operations with memory management'
      ],
      technicalDetails: {
        compression: 'Learnable vector quantization + neural codec',
        retrieval: 'Approximate nearest neighbor with learned hashing',
        attention: 'Sparse + dense hybrid, block-diagonal structure',
        caching: 'KV-cache compression with importance sampling',
        indexing: 'Hierarchical navigable small world graphs'
      },
      benchmarkImpact: [
        { name: 'GraphWalks BFS 256K-1M', from: 22.3, to: 68.9, mythos: 80.0 },
        { name: 'Full-System Debugging', from: 47.5, to: 71.2, mythos: 78.9 },
        { name: 'SWE-bench Pro', from: 42.3, to: 65.7, mythos: 77.8 }
      ]
    },
    {
      id: 4,
      name: 'Multi-Step Planning Engine (MSPE)',
      priority: 'CRITICAL',
      complexity: 'Very High',
      expectedGain: '+15-22 points avg',
      compute: '1.9x training, 2.4x inference',
      implementation: [
        'Add explicit planning phase before generation',
        'Implement goal decomposition into sub-goals (hierarchical)',
        'Use policy network for action selection',
        'Add world model for simulating action outcomes',
        'Implement rollback mechanisms when plans fail',
        'Use reinforcement learning for plan optimization',
        'Add explicit state tracking and constraint satisfaction',
        'Implement dependency graph construction and analysis'
      ],
      technicalDetails: {
        planner: 'Hierarchical task network with learned heuristics',
        policy: 'Actor-critic architecture, PPO training',
        worldModel: 'Transition model + reward predictor',
        rollback: 'Checkpoint-based state restoration, delta encoding',
        constraints: 'SAT solver integration for consistency checking'
      },
      benchmarkImpact: [
        { name: 'Terminal-Bench 2.0', from: 45.7, to: 72.3, mythos: 82.0 },
        { name: 'OSWorld-Verified', from: 56.2, to: 74.8, mythos: 79.6 },
        { name: 'Cybersecurity CTF', from: 52.1, to: 81.4, mythos: 100.0 }
      ]
    },
    {
      id: 5,
      name: 'Domain-Specific Sub-Networks (DSSN)',
      priority: 'HIGH',
      complexity: 'High',
      expectedGain: '+8-14 points avg',
      compute: '2.4x training, 1.2x inference',
      implementation: [
        'Pre-train specialized sub-networks for each domain',
        'Domains: code (Python, Java, C++, etc), math, science, language',
        'Use domain-specific tokenizers and embeddings',
        'Implement cross-domain transfer learning',
        'Add meta-learning for quick domain adaptation',
        'Use adapter layers for domain switching',
        'Implement domain-aware routing in MoE layers'
      ],
      technicalDetails: {
        pretraining: 'Domain-specific corpora, curriculum learning',
        adapters: 'Low-rank adaptation (LoRA) with rank 128-256',
        metaLearning: 'MAML-style few-shot adaptation',
        transfer: 'Progressive neural architecture search',
        routing: 'Domain classifier + confidence-weighted mixing'
      },
      benchmarkImpact: [
        { name: 'Legacy Code Migration', from: 44.8, to: 67.3, mythos: 73.6 },
        { name: 'Multi-file Refactoring', from: 58.7, to: 76.1, mythos: 82.7 },
        { name: 'GPQA Diamond', from: 69.3, to: 88.7, mythos: 94.6 }
      ]
    },
    {
      id: 6,
      name: 'Scale to 10T Parameters',
      priority: 'CRITICAL',
      complexity: 'Very High',
      expectedGain: '+5-12 points avg (multiplicative with other improvements)',
      compute: '8.2x training, 2.1x inference',
      implementation: [
        'Increase model width to 16,384 (from ~8,192)',
        'Increase depth to 128 layers (from ~64)',
        'Scale MoE to 256 experts × 32 layers',
        'Use 3D parallelism: tensor + pipeline + data',
        'Implement ZeRO-Infinity for memory efficiency',
        'Use BF16 + FP8 mixed precision training',
        'Optimize communication with gradient compression',
        'Use checkpoint recomputation for memory efficiency'
      ],
      technicalDetails: {
        totalParams: '~9.8T total, ~1.4T active',
        hiddenSize: 16384,
        numLayers: 128,
        numHeads: 128,
        headDim: 128,
        vocabSize: 256000,
        intermediateSize: 65536,
        training: '512-1024 H100 GPUs, 60-90 days',
        dataSize: '15-20T tokens, high-quality filtered'
      },
      benchmarkImpact: [
        { name: 'All benchmarks', from: 'baseline', to: '+5-12% multiplicative', mythos: 'target' }
      ]
    }
  ];

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-green-500/30 bg-green-950/30 p-6">
        <h2 className="mb-4 text-2xl font-bold text-white">Critical Architectural Improvements</h2>
        <p className="mb-4 text-slate-300">
          These are the 6 fundamental architectural changes required to reach Mythos-level performance. Each
          improvement is analyzed with implementation details, compute requirements, and expected benchmark gains.
        </p>
        <div className="rounded-lg bg-blue-950/30 border border-blue-500/30 p-4">
          <p className="text-blue-200 text-sm">
            <strong>Implementation Strategy:</strong> These improvements are NOT independent. They exhibit strong
            synergistic effects. Implementing improvements 1-3 yields ~60% of gap closure. Adding 4-6 yields the
            remaining 35-40% through exponential synergy. Compute requirements assume same hardware as Mythos
            (~512-1024 H100 GPUs).
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {improvements.map((improvement) => (
          <div
            key={improvement.id}
            className="rounded-lg border border-slate-700 bg-slate-800/50 p-6"
          >
            <div className="mb-4 flex items-start justify-between">
              <div>
                <div className="mb-2 flex items-center gap-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-600 text-sm font-bold text-white">
                    {improvement.id}
                  </span>
                  <h3 className="text-xl font-bold text-white">{improvement.name}</h3>
                </div>
                <div className="flex gap-3">
                  <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                    improvement.priority === 'CRITICAL'
                      ? 'bg-red-500/20 text-red-300'
                      : 'bg-yellow-500/20 text-yellow-300'
                  }`}>
                    {improvement.priority}
                  </span>
                  <span className="rounded-full bg-slate-700 px-2.5 py-1 text-xs font-medium text-slate-300">
                    Complexity: {improvement.complexity}
                  </span>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-slate-400">Expected Gain</p>
                <p className="text-xl font-bold text-green-400">{improvement.expectedGain}</p>
                <p className="text-xs text-slate-500 mt-1">{improvement.compute}</p>
              </div>
            </div>

            <div className="mb-4 rounded-lg bg-slate-900/50 p-4">
              <h4 className="mb-2 text-sm font-semibold text-white">Implementation Steps</h4>
              <ul className="space-y-1.5">
                {improvement.implementation.map((step, idx) => (
                  <li key={idx} className="flex gap-2 text-sm text-slate-300">
                    <span className="text-violet-400">•</span>
                    <span>{step}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="mb-4 rounded-lg bg-slate-900/50 p-4">
              <h4 className="mb-3 text-sm font-semibold text-white">Technical Details</h4>
              <div className="grid gap-2 md:grid-cols-2">
                {Object.entries(improvement.technicalDetails).map(([key, value]) => (
                  <div key={key} className="text-sm">
                    <span className="font-medium text-indigo-400">
                      {key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1')}:
                    </span>{' '}
                    <span className="text-slate-300">{value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h4 className="mb-3 text-sm font-semibold text-white">Benchmark Impact</h4>
              <div className="space-y-2">
                {improvement.benchmarkImpact.map((bench, idx) => (
                  <div key={idx} className="rounded bg-slate-900/50 p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-sm font-medium text-white">{bench.name}</span>
                      <span className="text-xs text-slate-400">
                        {typeof bench.from === 'number' && typeof bench.to === 'number'
                          ? `${bench.from.toFixed(1)} → ${bench.to.toFixed(1)} (${((bench.to - bench.from)).toFixed(1)})`
                          : `${bench.from} → ${bench.to}`}
                      </span>
                    </div>
                    {typeof bench.from === 'number' && typeof bench.to === 'number' && typeof bench.mythos === 'number' && (
                      <div className="flex items-center gap-2">
                        <div className="flex-1">
                          <div className="h-1.5 rounded-full bg-slate-700">
                            <div
                              className="h-1.5 rounded-full bg-gradient-to-r from-yellow-500 to-green-500"
                              style={{ width: `${((bench.to - bench.from) / (bench.mythos - bench.from)) * 100}%` }}
                            />
                          </div>
                        </div>
                        <span className="text-xs text-green-400 font-medium">
                          {(((bench.to - bench.from) / (bench.mythos - bench.from)) * 100).toFixed(0)}% of gap
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="rounded-lg border border-violet-500/30 bg-violet-950/30 p-6">
        <h3 className="mb-3 text-xl font-bold text-white">Total Compute Requirements</h3>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <p className="mb-2 text-sm text-slate-300">
              <strong className="text-violet-300">Training:</strong> Implementing all 6 improvements requires approximately
              12-18x baseline training compute, or ~40-60 days on 512-1024 H100 GPUs.
            </p>
            <p className="text-sm text-slate-300">
              <strong className="text-violet-300">Inference:</strong> Deployed model requires 2.1-2.8x baseline inference
              compute per token, with adaptive depth reducing average to ~2.3x.
            </p>
          </div>
          <div className="rounded-lg bg-slate-900/50 p-4">
            <p className="mb-2 text-xs font-medium text-slate-400">ESTIMATED COSTS (AWS p5 instances)</p>
            <div className="space-y-1 text-sm text-slate-300">
              <p>Training: <strong className="text-green-400">$18-28M</strong></p>
              <p>Inference (per 1M tokens): <strong className="text-yellow-400">$45-65</strong></p>
              <p className="mt-2 text-xs text-slate-500">
                vs. Mythos estimated: $35-50M training, $60-90/1M tokens inference
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
