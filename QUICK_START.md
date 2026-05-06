# Mythos Transformer: Quick Start Guide

## ⚡ Start Training in 15 Minutes

This guide gets you from zero to training in 15 minutes.

---

## Step 1: Clone & Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/your-org/mythos-transformer.git
cd mythos-transformer

# Create environment
conda create -n mythos python=3.10 -y
conda activate mythos

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

---

## Step 2: Install Dependencies (5 minutes)

```bash
# Install core dependencies
pip install -r requirements.txt

# Install DeepSpeed with optimizations
DS_BUILD_CPU_ADAM=1 \
DS_BUILD_FUSED_ADAM=1 \
DS_BUILD_UTILS=1 \
pip install deepspeed

# Install Apex (for fused kernels)
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
cd ..

# Install Flash Attention
pip install flash-attn --no-build-isolation
```

---

## Step 3: Verify Installation (1 minute)

```bash
# Test all components
python -c "import torch; print(f'PyTorch {torch.__version__}: CUDA {torch.cuda.is_available()}')"
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
python -c "from flash_attn import flash_attn_func; print('Flash Attention: OK')"
python -c "from src.models import MythosTransformer; print('Mythos Model: OK')"
```

**Expected output:**
```
PyTorch 2.3.0: CUDA True
DeepSpeed 0.14.0
Flash Attention: OK
Mythos Model: OK
```

---

## Step 4: Prepare Data (2 minutes)

### Option A: Use Sample Data (for testing)

```bash
# Create sample dataset
python -c "
import torch
import os

os.makedirs('data/sample', exist_ok=True)

# Create 10 shards of random data for testing
for i in range(10):
    data = {
        'input_ids': torch.randint(0, 256000, (1000, 8192)),
        'labels': torch.randint(0, 256000, (1000, 8192)),
    }
    torch.save(data, f'data/sample/shard_{i}.pt')
    
print('Sample data created!')
"
```

### Option B: Prepare Real Data (for production)

See [docs/TRAINING.md](docs/TRAINING.md#data-preparation) for full instructions.

---

## Step 5: Configure Training (1 minute)

### Option A: Small Test Run (8 GPUs)

```bash
# Copy and modify config
cp configs/mythos_10t_full.yaml configs/test_run.yaml

# Edit for small test
# Change: distributed.tensor_parallel_size: 2
# Change: distributed.pipeline_parallel_size: 2
# Change: distributed.data_parallel_size: 2
# Change: training.total_steps: 1000
```

### Option B: Validation Run (64 GPUs)

```bash
# Use pre-configured validation config
# configs/mythos_2t_validation.yaml (to be created)
```

### Option C: Full Training (1024 GPUs)

```bash
# Use as-is
# configs/mythos_10t_full.yaml
```

---

## Step 6: Launch Training! (1 minute)

### Single Node (8 GPUs)

```bash
# Set environment variables
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch!
bash scripts/launch_training.sh configs/test_run.yaml
```

### Multi-Node (1024 GPUs)

```bash
# On master node (node 0)
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export NUM_NODES=128
export NUM_GPUS=8
export NODE_RANK=0

bash scripts/launch_training.sh configs/mythos_10t_full.yaml

# On worker nodes (node 1, 2, 3, ..., 127)
export MASTER_ADDR=<master_node_ip>
export MASTER_PORT=29500
export NUM_NODES=128
export NUM_GPUS=8
export NODE_RANK=<1,2,3,...,127>  # Different for each node

bash scripts/launch_training.sh configs/mythos_10t_full.yaml
```

---

## Step 7: Monitor Progress (ongoing)

### Option A: Terminal Output

```bash
# In a separate terminal
tail -f logs/training.log
```

**Example output:**
```
Step 10/500000 | Loss: 8.2341 | LR: 1.20e-04 | Tokens/sec: 512 | ETA: 45.2h
Step 20/500000 | Loss: 8.1203 | LR: 1.20e-04 | Tokens/sec: 518 | ETA: 45.0h
Step 30/500000 | Loss: 7.9876 | LR: 1.20e-04 | Tokens/sec: 521 | ETA: 44.8h
```

### Option B: WandB Dashboard

```bash
# Login to WandB
wandb login

# View at: https://wandb.ai/your-org/mythos-transformer
```

### Option C: TensorBoard

```bash
# Launch TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# View at: http://localhost:6006
```

---

## Troubleshooting

### Issue: CUDA OOM

```bash
# Reduce micro batch size
# Edit config: training.micro_batch_size: 1
# Increase gradient accumulation: training.gradient_accumulation_steps: 4
```

### Issue: Slow Training (<400 tokens/sec/GPU)

```bash
# Check interconnect
ibstatus  # Should show ACTIVE

# Optimize NCCL
export NCCL_ALGO=Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NRINGS=8

# Restart training
```

### Issue: Expert Collapse

```python
# Monitor expert utilization
python scripts/monitor_experts.py checkpoints/latest

# If <60%, increase auxiliary loss
# Edit config: moe.load_balance_loss_weight: 0.02
```

---

## Next Steps

### After First 1000 Steps

```bash
# Run evaluation
python -m src.evaluation.benchmarks \
    --checkpoint checkpoints/step_1000 \
    --benchmarks swe_bench,graphwalks

# Check metrics
python scripts/analyze_metrics.py checkpoints/step_1000
```

### After Phase 1 Complete (Validation)

```bash
# Evaluate on full benchmark suite
bash scripts/evaluate.sh checkpoints/phase1_final

# Compare to targets
python scripts/compare_targets.py \
    --checkpoint checkpoints/phase1_final \
    --targets configs/phase1_targets.yaml
```

### Scale to Full Training

```bash
# Phase 3: Full 10T scale
bash scripts/launch_training.sh configs/mythos_10t_full.yaml
```

---

## Estimated Timelines & Costs

### Test Run (8 GPUs, 1K steps)
- **Time:** 2-4 hours
- **Cost:** ~$100 (AWS p5.48xlarge × 1 instance)
- **Purpose:** Verify setup, test throughput

### Validation Run (64 GPUs, 100K steps)
- **Time:** 40-60 days
- **Cost:** $1.5-3M
- **Purpose:** Validate architecture
- **Expected:** 60% gap closure

### Full Training (1024 GPUs, 500K steps)
- **Time:** 60-90 days
- **Cost:** $32-50M
- **Purpose:** Achieve Mythos-level performance
- **Expected:** 95% gap closure

---

## Success Criteria

### After 1K Steps
- ✅ Training loss decreasing
- ✅ No OOM errors
- ✅ Throughput >400 tokens/sec/GPU
- ✅ Expert utilization >70%

### After 10K Steps
- ✅ Loss < 7.0
- ✅ Expert utilization >80%
- ✅ Value network MSE <0.1
- ✅ No expert collapse

### After 100K Steps (Phase 1)
- ✅ Loss < 4.0
- ✅ GraphWalks >60%
- ✅ SWE-bench >70%
- ✅ All metrics within target ranges

### After 500K Steps (Phase 3)
- ✅ Loss < 2.0
- ✅ SWE-bench Verified >88%
- ✅ GraphWalks >71%
- ✅ Average gap to Mythos <5 points
- ✅ **Success!** 🎉

---

## Support

**Issues?** Check [docs/TRAINING.md](docs/TRAINING.md) for detailed troubleshooting.

**Questions?** Open a GitHub issue or contact team@your-org.com.

**Want to contribute?** See CONTRIBUTING.md (to be created).

---

## Checklist

Use this checklist to track your progress:

- [ ] Repository cloned
- [ ] Environment created and activated
- [ ] Dependencies installed (PyTorch, DeepSpeed, Apex, Flash Attention)
- [ ] Installation verified (all imports work)
- [ ] Data prepared (sample or real)
- [ ] Config selected/modified
- [ ] Training launched successfully
- [ ] Monitoring setup (terminal/WandB/TensorBoard)
- [ ] First 1K steps completed
- [ ] Metrics look good (no OOM, good throughput)
- [ ] Evaluation run successfully
- [ ] Ready to scale up!

---

**You're ready! Start training and achieve Mythos-level performance! 🚀**

Time from zero to training: **~15 minutes**  
Time to Mythos-level performance: **16-18 months**  
Expected gap closure: **95%**

**Good luck!**
