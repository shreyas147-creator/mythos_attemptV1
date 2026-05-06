# Mythos Transformer: Training Guide

Complete step-by-step guide for training the Mythos Transformer from scratch.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Training Phases](#training-phases)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Optimization Tips](#optimization-tips)

---

## Prerequisites

### Hardware Requirements

**Minimum (Validation Run):**
- 64 H100 (80GB) GPUs
- 2 TB system RAM
- 200 TB NVMe storage
- 1600 Gbps InfiniBand

**Recommended (Full Training):**
- 512-1024 H100 (80GB) GPUs
- 8-16 TB system RAM
- 2 PB NVMe storage
- 3200 Gbps InfiniBand or NVLink

**Estimated Costs:**
- AWS p5.48xlarge: ~$98/hour/instance (8 GPUs)
- For 1024 GPUs: 128 instances × $98/hour = $12,544/hour
- Full training (60-90 days): $18M - $27M

### Software Requirements

```bash
OS: Ubuntu 22.04 LTS
CUDA: 12.1+
cuDNN: 8.9+
NCCL: 2.18+
Python: 3.10+
PyTorch: 2.3+
DeepSpeed: 0.14+
```

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/mythos-transformer.git
cd mythos-transformer
```

### 2. Create Environment

```bash
# Create conda environment
conda create -n mythos python=3.10
conda activate mythos

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 3. Install DeepSpeed with Optimizations

```bash
# Build DeepSpeed with all ops
DS_BUILD_CPU_ADAM=1 \
DS_BUILD_FUSED_ADAM=1 \
DS_BUILD_FUSED_LAMB=1 \
DS_BUILD_UTILS=1 \
pip install deepspeed --global-option="build_ext"
```

### 4. Install Apex (for fused kernels)

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
cd ..
```

### 5. Install Flash Attention 2

```bash
pip install flash-attn --no-build-isolation
```

### 6. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
python -c "from flash_attn import flash_attn_func; print('Flash Attention: OK')"
```

---

## Data Preparation

### Data Requirements

**Total:** 15-20 trillion tokens

**Mixture:**
- Code: 35% (5.25T tokens)
  - Python: 40%, JavaScript: 20%, Java: 15%, C++: 15%, Other: 10%
- Mathematics: 15% (2.25T tokens)
- Science: 20% (3T tokens)
- General Text: 25% (3.75T tokens)
- Reasoning: 5% (750B tokens)

### Step 1: Download Raw Data

```bash
# Create data directory
mkdir -p /mnt/data/raw
mkdir -p /mnt/data/processed

# Download datasets (example)
# Code
git clone --depth 1 https://github.com/github/CodeSearchNet /mnt/data/raw/code
wget https://data.deepmind.com/bigcode-dataset.tar.gz -P /mnt/data/raw/code

# Mathematics
wget https://math-datasets.huggingface.co/mathematics.tar.gz -P /mnt/data/raw/math

# Science
wget https://arxiv.org/src/papers.tar.gz -P /mnt/data/raw/science

# General text
wget https://pile.eleuther.ai/pile.tar.gz -P /mnt/data/raw/text
```

### Step 2: Quality Filtering

```bash
python scripts/filter_quality.py \
    --input /mnt/data/raw \
    --output /mnt/data/filtered \
    --min_quality_score 0.7 \
    --num_workers 64
```

**Filtering criteria:**
- Remove duplicates (MinHash + SimHash)
- Filter low-quality (perplexity, toxicity, repetition)
- Remove PII (personally identifiable information)
- Normalize encoding (UTF-8)

### Step 3: Tokenization

```bash
# Train domain-aware tokenizer
python scripts/train_tokenizer.py \
    --data_path /mnt/data/filtered \
    --vocab_size 256000 \
    --output_path tokenizers/mythos_tokenizer \
    --domain_aware \
    --num_workers 64

# Tokenize datasets
python scripts/tokenize_data.py \
    --tokenizer tokenizers/mythos_tokenizer \
    --input_path /mnt/data/filtered \
    --output_path /mnt/data/tokenized \
    --num_workers 128 \
    --chunk_size 8192
```

### Step 4: Create Training Shards

```bash
python scripts/create_shards.py \
    --input_path /mnt/data/tokenized \
    --output_path /mnt/data/shards \
    --shard_size 100M \
    --shuffle \
    --seed 42
```

**Result:** ~150-200 shards of 100M tokens each

---

## Training Phases

### Phase 1: Validation Run (2T Parameters, 2-3 months)

**Goal:** Validate architecture before full-scale training

```bash
# Configuration: configs/mythos_2t_validation.yaml

# Launch training
bash scripts/launch_training.sh configs/mythos_2t_validation.yaml

# Expected:
# - 64-128 H100 GPUs
# - 40-60 days training time
# - Cost: ~$1.5-3M
```

**Key Metrics to Monitor:**

1. **Expert Utilization:**
   ```python
   Target: >80% across all experts
   If <60%: Increase auxiliary loss weight
   If <40%: Check routing network gradients
   ```

2. **Routing Entropy:**
   ```python
   Target: >4.0
   If <3.0: Add routing jitter
   If <2.0: Reduce expert dropout
   ```

3. **Value Network Calibration:**
   ```python
   Target MSE: <0.05
   If >0.1: Retrain value network
   If >0.2: Check training data quality
   ```

4. **Memory Compression:**
   ```python
   Target loss: <0.02
   If >0.05: Increase codebook size
   If >0.1: Adjust compression ratios
   ```

**Success Criteria:**
- GraphWalks BFS: >60% (baseline: 8.5%, target: 62.7%)
- SWE-bench Verified: >70% (baseline: 45%, target: 73.2%)
- HLE: >55% (baseline: 24%, target: 38.5%)
- All core metrics within target ranges

### Phase 2: Full Architecture (3T Parameters, 4-5 months)

**Goal:** Validate all 6 components together

```bash
# Configuration: configs/mythos_3t_full_architecture.yaml

# Launch training
bash scripts/launch_training.sh configs/mythos_3t_full_architecture.yaml

# Expected:
# - 128-256 H100 GPUs
# - 60-90 days training time
# - Cost: ~$4-8M
```

**Additional Metrics:**

5. **Planning Success Rate:**
   ```python
   Target: >70% on multi-step tasks
   If <50%: Improve world model
   If <30%: Increase planning horizon
   ```

6. **Domain Adaptation:**
   ```python
   Target: >85% with 10% domain data
   If <70%: Increase LoRA rank
   If <60%: Pre-train domains separately
   ```

**Success Criteria:**
- Cybersecurity CTF: >70% (baseline: 12%, target: 72.8%)
- Legacy Migration: >65% (baseline: 28%, target: 67.3%)
- Terminal-Bench 2.0: >72% (baseline: 28%, target: 72.3%)

### Phase 3: Full Scale (10T Parameters, 3-4 months)

**Goal:** Achieve 95% gap closure vs Mythos

```bash
# Configuration: configs/mythos_10t_full.yaml

# Launch training
bash scripts/launch_training.sh configs/mythos_10t_full.yaml

# Expected:
# - 512-1024 H100 GPUs
# - 60-90 days training time
# - Cost: ~$18-45M
```

**Launch Command (Multi-Node):**

```bash
# On master node
export MASTER_ADDR=<master_ip>
export MASTER_PORT=29500
export NUM_NODES=128
export NUM_GPUS=8

# On each node
export NODE_RANK=<0,1,2,...127>

bash scripts/launch_training.sh configs/mythos_10t_full.yaml
```

**Success Criteria:**
- SWE-bench Verified: >88% (target: 89.2%, Mythos: 93.9%)
- GraphWalks BFS: >71% (target: 71.4%, Mythos: 80.0%)
- Cybersecurity CTF: >87% (target: 87.3%, Mythos: 100%)
- GPQA Diamond: >92% (target: 93.1%, Mythos: 94.6%)
- **Average gap to Mythos: <5 points**

---

## Monitoring

### WandB Dashboard

```python
# Automatic tracking:
- Training loss (per component)
- Expert utilization heatmaps
- Routing entropy over time
- Value network calibration
- Memory compression efficiency
- Planning success rates
- Gradient norms
- Learning rate schedules
- Throughput (tokens/sec)
- GPU utilization
- Memory usage
```

**Access:** https://wandb.ai/your-org/mythos-transformer

### TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir /mnt/logs/tensorboard --port 6006

# Access: http://localhost:6006
```

### Real-time Metrics

```bash
# Monitor training progress
tail -f /mnt/logs/training.log

# Check GPU utilization
watch -n 1 nvidia-smi

# Monitor network
iftop -i ib0  # InfiniBand interface

# Check disk I/O
iostat -x 1
```

### Custom Monitoring Script

```python
# scripts/monitor.py
import wandb
import time

api = wandb.Api()
run = api.run("your-org/mythos-transformer/run_id")

while True:
    metrics = run.summary
    
    print(f"Step: {metrics['step']}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Expert Util: {metrics['expert_utilization']:.2%}")
    print(f"Tokens/sec: {metrics['tokens_per_sec']:.0f}")
    print(f"ETA: {metrics['eta_hours']:.1f}h")
    print("-" * 50)
    
    time.sleep(60)
```

---

## Troubleshooting

### Common Issues

#### 1. Expert Collapse

**Symptoms:**
- Expert utilization <60%
- Routing entropy <3.0
- Some experts always selected

**Solutions:**
```yaml
# Increase load balancing loss weight
moe:
  load_balance_loss_weight: 0.02  # From 0.01

# Add more routing jitter
moe:
  jitter_eps: 0.02  # From 0.01

# Reduce expert dropout
moe:
  expert_dropout: 0.05  # From 0.1
```

#### 2. Value Network Miscalibration

**Symptoms:**
- Value MSE >0.1
- Poor confidence estimates
- Inefficient reasoning depth allocation

**Solutions:**
```bash
# Retrain value network with more data
python scripts/retrain_value_network.py \
    --checkpoint checkpoints/latest \
    --training_data data/value_training \
    --steps 10000

# Apply temperature scaling
python scripts/calibrate_value_network.py \
    --checkpoint checkpoints/latest \
    --calibration_data data/calibration
```

#### 3. Memory Compression Failure

**Symptoms:**
- Compression loss >0.05
- Poor recall on long contexts
- High surprise variance

**Solutions:**
```yaml
# Increase codebook size
memory:
  num_codebook_vectors: 16384  # From 8192

# Adjust compression ratios
memory:
  l2_compression: 6  # From 8 (less aggressive)

# Lower surprise threshold
memory:
  surprise_threshold: 0.4  # From 0.5 (retain more)
```

#### 4. OOM (Out of Memory)

**Symptoms:**
- CUDA OOM errors
- Killed processes
- Slow training

**Solutions:**
```yaml
# Reduce micro batch size
training:
  micro_batch_size: 1  # Already minimum

# Increase gradient accumulation
training:
  gradient_accumulation_steps: 4  # From 2

# Enable more aggressive checkpointing
distributed:
  activation_checkpointing: true
  checkpoint_num_layers: 1  # Checkpoint every layer
```

#### 5. Slow Training

**Symptoms:**
- <400 tokens/sec/GPU
- High communication overhead
- GPU underutilization

**Solutions:**
```bash
# Check interconnect
ibstatus  # Should show ACTIVE

# Optimize NCCL
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NRINGS=8

# Enable gradient compression
distributed:
  gradient_compression: "powersgd"

# Overlap communication
zero_optimization:
  overlap_comm: true
```

---

## Optimization Tips

### 1. Data Loading Optimization

```yaml
data:
  num_workers: 16  # Increase for faster loading
  prefetch_factor: 4  # Prefetch batches
  pin_memory: true  # Pin to GPU memory
```

### 2. Mixed Precision

```yaml
training:
  precision: "bf16"  # Use BF16 for stability
  use_fp8: true  # Enable FP8 for compute

optimization:
  use_fused_kernels: true  # Fused operations
  use_flash_attention: true  # Flash Attention 2
```

### 3. Compilation

```yaml
optimization:
  compile_model: true
  compile_mode: "max-autotune"  # Aggressive optimization
```

### 4. Gradient Accumulation Strategy

```python
# Dynamic gradient accumulation based on batch size
if global_batch_size < target_batch_size:
    grad_accum_steps = target_batch_size // (micro_batch * num_gpus)
else:
    grad_accum_steps = 1
```

### 5. Learning Rate Schedule

```yaml
training:
  # Cosine with warmup
  lr_scheduler: "cosine"
  warmup_steps: 2000  # 0.4% of total steps
  min_learning_rate: 1.2e-5  # 10% of max LR
```

### 6. Checkpointing Strategy

```yaml
checkpointing:
  save_interval: 1000  # Save every 1000 steps
  keep_last_n: 10  # Keep last 10 checkpoints
  async_save: true  # Non-blocking saves
```

---

## Performance Targets

### Throughput Expectations

| Configuration | GPUs | Tokens/sec/GPU | Total Tokens/sec | Days to 15T |
|---------------|------|----------------|------------------|-------------|
| Validation (2T) | 64 | 450 | 28,800 | 603 |
| Validation (2T) | 128 | 500 | 64,000 | 271 |
| Full (10T) | 512 | 450 | 230,400 | 75 |
| Full (10T) | 1024 | 500 | 512,000 | 34 |

### Cost Estimates

| Phase | GPUs | Days | GPU-Days | Cost (AWS p5) |
|-------|------|------|----------|---------------|
| Validation | 128 | 60 | 7,680 | $3M |
| Full Architecture | 256 | 75 | 19,200 | $8M |
| Full Scale | 1024 | 67 | 68,608 | $35M |

---

## Next Steps

After successful training:

1. **Run Full Benchmark Suite:**
   ```bash
   bash scripts/evaluate.sh checkpoints/mythos-10t-final
   ```

2. **Fine-tune for Specific Tasks:**
   ```bash
   python scripts/finetune.py \
       --checkpoint checkpoints/mythos-10t-final \
       --task swe_bench \
       --steps 5000
   ```

3. **Deploy for Inference:**
   ```bash
   python scripts/deploy.py \
       --checkpoint checkpoints/mythos-10t-final \
       --deployment_config configs/inference.yaml
   ```

4. **Continuous Improvement:**
   - Analyze failure cases
   - Augment training data
   - Fine-tune underperforming components
   - Iterate!

---

**Ready to train? Start with Phase 1 validation to verify your setup before committing to full-scale training!**

For questions or issues, file an issue on GitHub or contact the team at team@your-org.com.
