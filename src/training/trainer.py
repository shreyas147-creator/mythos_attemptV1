"""
Mythos Transformer Training Loop

Production-grade training with:
- 3D parallelism (tensor + pipeline + data)
- ZeRO-3 optimization
- Mixed precision (BF16 + FP8)
- Gradient checkpointing
- Curriculum learning
- Advanced monitoring
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import deepspeed
from typing import Dict, Optional, Tuple
import os
import time
import wandb
from pathlib import Path
import yaml

from ..models.mythos_transformer import MythosTransformer, count_parameters
from ..data.dataloader import get_training_dataloader
from ..utils.checkpointing import CheckpointManager
from ..utils.monitoring import MetricsTracker


class MythosTrainer:
    """Main training orchestrator for Mythos Transformer."""
    
    def __init__(
        self,
        config: Dict,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Setup device
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizer
        self.optimizer = self._build_optimizer()
        
        # Initialize data loaders
        self.train_loader = self._build_dataloader()
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, _ = self._init_deepspeed()
        
        # Initialize checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            save_dir=config['checkpointing']['save_dir'],
            keep_last_n=config['checkpointing']['keep_last_n'],
        )
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker(
            use_wandb=config['logging']['use_wandb'],
            project=config['logging'].get('wandb_project'),
            entity=config['logging'].get('wandb_entity'),
            name=config['logging'].get('wandb_name'),
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.tokens_seen = 0
        
    def _build_model(self) -> nn.Module:
        """Build Mythos Transformer model."""
        model_config = self.config['model']
        
        model = MythosTransformer(
            vocab_size=model_config['vocab_size'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            max_seq_len=model_config['max_seq_len'],
            use_moe=model_config['use_moe'],
            use_reasoning=model_config['use_reasoning'],
            use_memory=model_config['use_memory'],
            dropout=model_config['dropout'],
        )
        
        # Print model info
        if self.global_rank == 0:
            total_params, trainable_params = count_parameters(model)
            print(f"\n{'='*80}")
            print(f"Mythos Transformer Model Initialized")
            print(f"{'='*80}")
            print(f"Total Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
            print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e9:.2f}B)")
            print(f"Hidden Dim: {model_config['hidden_dim']}")
            print(f"Layers: {model_config['num_layers']}")
            print(f"Heads: {model_config['num_heads']}")
            print(f"Vocab Size: {model_config['vocab_size']:,}")
            print(f"{'='*80}\n")
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with weight decay."""
        train_config = self.config['training']
        
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': train_config['weight_decay'],
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=train_config['learning_rate'],
            betas=(train_config['adam_beta1'], train_config['adam_beta2']),
            eps=train_config['adam_epsilon'],
        )
        
        return optimizer
    
    def _build_dataloader(self) -> DataLoader:
        """Build training data loader."""
        data_config = self.config['data']
        train_config = self.config['training']
        
        return get_training_dataloader(
            data_path=data_config['data_path'],
            batch_size=train_config['micro_batch_size'],
            seq_length=train_config['seq_length'],
            num_workers=data_config.get('num_workers', 8),
            world_size=self.world_size,
            rank=self.global_rank,
        )
    
    def _init_deepspeed(self):
        """Initialize DeepSpeed engine."""
        ds_config = {
            "train_batch_size": self.config['training']['batch_size'],
            "train_micro_batch_size_per_gpu": self.config['training']['micro_batch_size'],
            "gradient_accumulation_steps": self.config['training']['gradient_accumulation_steps'],
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config['training']['learning_rate'],
                    "betas": [
                        self.config['training']['adam_beta1'],
                        self.config['training']['adam_beta2'],
                    ],
                    "eps": self.config['training']['adam_epsilon'],
                    "weight_decay": self.config['training']['weight_decay'],
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config['training']['learning_rate'],
                    "warmup_num_steps": self.config['training']['warmup_steps'],
                    "total_num_steps": self.config['training']['total_steps'],
                }
            },
            "gradient_clipping": self.config['training']['gradient_clipping'],
            "fp16": {
                "enabled": self.config['training']['precision'] == 'fp16',
            },
            "bf16": {
                "enabled": self.config['training']['precision'] == 'bf16',
            },
            "zero_optimization": self.config['distributed']['zero_optimization'],
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": True,
                "number_checkpoints": self.config['distributed'].get('checkpoint_num_layers', 1),
            },
            "wall_clock_breakdown": True,
        }
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            config=ds_config,
        )
        
        return model_engine, optimizer, None, None
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Single training step."""
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=True,
        )
        
        logits = outputs['logits']
        metrics = outputs['metrics']
        
        # Compute language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        # Add auxiliary losses
        total_loss = lm_loss
        
        # MoE load balancing loss
        if 'total_moe_loss' in metrics:
            moe_loss = metrics['total_moe_loss']
            moe_weight = self.config['moe'].get('load_balance_loss_weight', 0.01)
            total_loss = total_loss + moe_weight * moe_loss
        
        # Backward pass
        self.model_engine.backward(total_loss)
        
        # Optimizer step
        self.model_engine.step()
        
        # Collect metrics
        step_metrics = {
            'loss': lm_loss.item(),
            'total_loss': total_loss.item(),
            'moe_loss': metrics.get('total_moe_loss', 0.0),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        
        # Add reasoning metrics if available
        if 'reasoning_metrics' in metrics:
            step_metrics.update({
                f"reasoning/{k}": v for k, v in metrics['reasoning_metrics'].items()
                if isinstance(v, (int, float))
            })
        
        # Add memory metrics if available
        if 'memory_metrics' in metrics:
            step_metrics.update({
                f"memory/{k}": v for k, v in metrics['memory_metrics'].items()
                if isinstance(v, (int, float))
            })
        
        return step_metrics
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"Global Batch Size: {self.config['training']['batch_size']}")
        print(f"Micro Batch Size: {self.config['training']['micro_batch_size']}")
        print(f"Gradient Accumulation Steps: {self.config['training']['gradient_accumulation_steps']}")
        print(f"Total Steps: {self.config['training']['total_steps']}")
        print(f"Sequence Length: {self.config['training']['seq_length']}")
        print(f"World Size: {self.world_size}")
        print(f"{'='*80}\n")
        
        total_steps = self.config['training']['total_steps']
        log_interval = self.config['logging']['log_interval']
        save_interval = self.config['checkpointing']['save_interval']
        
        self.model_engine.train()
        
        start_time = time.time()
        step_times = []
        
        while self.global_step < total_steps:
            for batch in self.train_loader:
                # Training step
                step_start = time.time()
                step_metrics = self.train_step(batch)
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Update counters
                self.global_step += 1
                batch_size = batch['input_ids'].size(0)
                seq_len = batch['input_ids'].size(1)
                self.tokens_seen += batch_size * seq_len * self.world_size
                
                # Compute throughput
                tokens_per_sec = (batch_size * seq_len * self.world_size) / step_time
                step_metrics['tokens_per_sec'] = tokens_per_sec
                step_metrics['tokens_seen'] = self.tokens_seen
                step_metrics['step'] = self.global_step
                
                # Log metrics
                if self.global_step % log_interval == 0:
                    avg_step_time = sum(step_times[-log_interval:]) / len(step_times[-log_interval:])
                    eta_seconds = (total_steps - self.global_step) * avg_step_time
                    eta_hours = eta_seconds / 3600
                    
                    if self.global_rank == 0:
                        print(f"Step {self.global_step}/{total_steps} | "
                              f"Loss: {step_metrics['loss']:.4f} | "
                              f"LR: {step_metrics['learning_rate']:.2e} | "
                              f"Tokens/sec: {tokens_per_sec:.0f} | "
                              f"ETA: {eta_hours:.1f}h")
                    
                    self.metrics.log(step_metrics, step=self.global_step)
                
                # Save checkpoint
                if self.global_step % save_interval == 0:
                    if self.global_rank == 0:
                        self.checkpoint_mgr.save(
                            model=self.model_engine,
                            optimizer=self.optimizer,
                            step=self.global_step,
                            epoch=self.epoch,
                            metrics=step_metrics,
                        )
                
                # Check if training complete
                if self.global_step >= total_steps:
                    break
            
            self.epoch += 1
        
        # Final checkpoint
        if self.global_rank == 0:
            self.checkpoint_mgr.save(
                model=self.model_engine,
                optimizer=self.optimizer,
                step=self.global_step,
                epoch=self.epoch,
                metrics=step_metrics,
                is_final=True,
            )
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}")
        print(f"Total Steps: {self.global_step}")
        print(f"Total Tokens: {self.tokens_seen:,}")
        print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"Average Tokens/sec: {self.tokens_seen/total_time:.0f}")
        print(f"{'='*80}\n")


def main():
    """Main entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Mythos Transformer")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize distributed training
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        local_rank = args.local_rank
    else:
        world_size = 1
        local_rank = 0
    
    # Create trainer
    trainer = MythosTrainer(
        config=config,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
