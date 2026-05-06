"""Minimal local trainer for the byte-level LocalTransformer."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from src.data.local_text import get_local_dataloader, load_text_bytes, split_tokens
from src.models.local_transformer import (
    LocalTransformer,
    LocalTransformerConfig,
    count_local_parameters,
)


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith(("\"", "'")) and value.endswith(("\"", "'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = raw_line.split(" #", 1)[0].rstrip()
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if ":" not in content:
            raise ValueError(f"unsupported config line: {raw_line}")

        key, value = content.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return load_simple_yaml(text)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_bytes(token_ids: torch.Tensor) -> str:
    values = [int(token) for token in token_ids.detach().cpu().flatten()]
    return bytes(values).decode("utf-8", errors="replace")


def evaluate(model: LocalTransformer, loader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss = model(input_ids, labels=labels)["loss"]
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def save_checkpoint(
    output_dir: Path,
    model: LocalTransformer,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "local_latest.pt"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )
    with open(output_dir / "local_latest.json", "w", encoding="utf-8") as handle:
        json.dump({"step": step, "checkpoint": str(path)}, handle, indent=2)
    return path


def train(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.get("seed", 7))
    set_seed(seed)

    model_cfg = LocalTransformerConfig(**config.get("model", {}))
    model_cfg.validate()
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    device = pick_device(str(train_cfg.get("device", "auto")))
    output_dir = Path(str(train_cfg.get("output_dir", "runs/local_model")))
    batch_size = int(train_cfg.get("batch_size", 8))
    total_steps = int(train_cfg.get("total_steps", 200))
    eval_interval = int(train_cfg.get("eval_interval", 50))
    save_interval = int(train_cfg.get("save_interval", 100))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    tokens = load_text_bytes(
        data_cfg.get("data_path"),
        allow_sample_fallback=bool(data_cfg.get("allow_sample_fallback", True)),
    )
    train_tokens, val_tokens = split_tokens(
        tokens,
        validation_fraction=float(data_cfg.get("validation_fraction", 0.1)),
    )
    pin_memory = device.type == "cuda"
    train_loader = get_local_dataloader(
        train_tokens,
        context_size=model_cfg.context_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    val_loader = get_local_dataloader(
        val_tokens,
        context_size=model_cfg.context_size,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = LocalTransformer(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_params, trainable_params = count_local_parameters(model)

    model.train()
    start_time = time.time()
    last_loss = math.nan
    step = 0

    while step < total_steps:
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids, labels=labels)["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("training loss became invalid")
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            last_loss = float(loss.item())

            if eval_interval > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, max_batches=10)
                print(f"step={step} train_loss={last_loss:.4f} val_loss={val_loss:.4f}")

            if save_interval > 0 and step % save_interval == 0:
                save_checkpoint(output_dir, model, optimizer, config, step)

            if step >= total_steps:
                break

    checkpoint = save_checkpoint(output_dir, model, optimizer, config, step)
    elapsed = time.time() - start_time

    prompt = torch.tensor([[ord("M")]], dtype=torch.long, device=device)
    sample = decode_bytes(model.generate(prompt, max_new_tokens=80, temperature=0.8, top_k=50)[0])
    print(sample)

    return {
        "steps": step,
        "last_loss": last_loss,
        "params": total_params,
        "trainable_params": trainable_params,
        "device": device.type,
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local byte-level transformer")
    parser.add_argument("--config", default="model_families/local/config.yaml")
    args = parser.parse_args()
    stats = train(args.config)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
