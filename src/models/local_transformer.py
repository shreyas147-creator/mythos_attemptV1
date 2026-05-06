"""Small decoder-only transformer for local experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LocalTransformerConfig:
    vocab_size: int = 256
    context_size: int = 256
    hidden_dim: int = 384
    num_layers: int = 4
    num_heads: int = 6
    dropout: float = 0.1
    mlp_multiplier: int = 4

    def validate(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if self.context_size < 2:
            raise ValueError("context_size must be >= 2")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class LocalTransformerBlock(nn.Module):
    def __init__(self, config: LocalTransformerConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_dim)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim)
        self.attn = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * config.mlp_multiplier),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * config.mlp_multiplier, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = residual + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LocalTransformer(nn.Module):
    """Byte-level local model used for fast training and efficiency tests."""

    def __init__(self, config: Optional[LocalTransformerConfig] = None):
        super().__init__()
        self.config = config or LocalTransformerConfig()
        self.config.validate()

        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)
        self.position_embedding = nn.Embedding(self.config.context_size, self.config.hidden_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [LocalTransformerBlock(self.config) for _ in range(self.config.num_layers)]
        )
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.output = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | None]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.context_size:
            raise ValueError("sequence length exceeds configured context_size")
        if input_ids.min() < 0 or input_ids.max() >= self.config.vocab_size:
            raise ValueError("input_ids contain token ids outside vocab range")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, causal_mask)

        logits = self.output(self.norm(x))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 80,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
    ) -> torch.Tensor:
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        was_training = self.training
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.context_size :]
            logits = self(context)["logits"][:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and 0 < top_k < logits.size(-1):
                    values, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        if was_training:
            self.train()
        return generated


def count_local_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable
