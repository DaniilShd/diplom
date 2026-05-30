#!/usr/bin/env python3
import json
import logging
import time
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def train_ltdetr(data_yaml, out_dir, max_steps=5500, lr=1e-4, batch_size=8, seed=42, freeze_backbone=False, val_every_steps=500):
    from lightly_train import train_object_detection
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    precision = None
    if torch.cuda.is_available():
        precision = "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    
    model_args = {"lr": lr}
    if freeze_backbone:
        model_args["backbone_freeze"] = True
    
    train_config = {
        "out": str(out_dir),
        "model": "dinov3/vits16-ltdetr-coco",
        "data": data_config,
        "seed": seed,
        "batch_size": batch_size,
        "overwrite": True,
        "steps": max_steps,
        "model_args": model_args,
        "logger_args": {"val_every_num_steps": val_every_steps},
        "save_checkpoint_args": {"save_every_num_steps": val_every_steps},
    }
    
    if precision:
        train_config["precision"] = precision
    
    logger.info(f"Training: steps={max_steps}, lr={lr}, batch={batch_size}, freeze={freeze_backbone}")
    
    start = time.time()
    train_object_detection(**train_config)
    elapsed = time.time() - start
    
    model_path = None
    for p in [
        out_dir / "exported_models" / "exported_best.pt",
        out_dir / "exported_models" / "exported_last.pt",
    ]:
        if p.exists():
            model_path = str(p)
            break
    
    if not model_path:
        for p in out_dir.glob("**/exported_best.pt"):
            model_path = str(p)
            break
    
    result = {
        "model_path": model_path,
        "training_time_hours": elapsed / 3600,
        "max_steps": max_steps,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
    }
    
    result_path = out_dir / "result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Training completed in {elapsed/3600:.2f}h")
    
    return result