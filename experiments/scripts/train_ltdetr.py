#!/usr/bin/env python3
import json
import logging
import re
import time
from pathlib import Path

import torch
import yaml

logger = logging.getLogger(__name__)


def _parse_train_log(out_dir):
    import pandas as pd
    
    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        metrics_csv = next(out_dir.glob("**/metrics.csv"), None)
    
    if metrics_csv and metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
            if not df.empty:
                val_cols = [c for c in df.columns if 'val' in c.lower() and 'map' in c.lower()]
                if val_cols:
                    vals = df[val_cols[0]].values
                    best_idx = vals.argmax()
                    return {
                        'best_val_map50': float(vals.max()),
                        'best_val_step': int((best_idx + 1) * 500),
                        'final_val_map50': float(vals[-1]),
                        'num_val_rounds': len(vals),
                    }
        except Exception as e:
            logger.warning(f"Не удалось прочитать metrics.csv: {e}")
    
    log_path = out_dir / "train.log"
    if not log_path.exists():
        candidates = list(out_dir.glob("**/train.log"))
        log_path = candidates[0] if candidates else None
    
    if not log_path:
        return {}
    
    try:
        content = log_path.read_text()
        pattern = r'val[_\s/]*(?:metric/)?(?:map|mAP)50[_\s/]*[:=]\s*([0-9]*\.?[0-9]+)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            values = [float(m) for m in matches if m]
            if values:
                return {'best_val_map50': max(values), 'final_val_map50': values[-1], 'num_val_rounds': len(values)}
    except Exception:
        pass
    
    return {}


def _find_model_path(out_dir):
    candidates = [
        out_dir / "exported_models" / "exported_best.pt",
        out_dir / "exported_models" / "exported_last.pt",
    ]
    candidates.extend(out_dir.glob("**/exported_models/exported_best.pt"))
    candidates.extend(out_dir.glob("**/exported_models/exported_last.pt"))
    for c in candidates:
        if c.exists():
            return c
    
    ckpts = list(out_dir.glob("**/checkpoints/*.ckpt"))
    if ckpts:
        return max(ckpts, key=lambda p: p.stat().st_mtime)
    
    raise FileNotFoundError(f"Модель не найдена в {out_dir}")


def _count_dataset_images(data_yaml_path):
    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)
    
    train_path = data_config.get('train')
    if isinstance(train_path, str):
        train_path = Path(train_path)
        if not train_path.is_absolute():
            train_path = Path(data_config['path']) / train_path
    else:
        train_path = Path(data_config['path']) / 'train' / 'images'
    
    if train_path.exists():
        if train_path.name == 'images':
            images_dir = train_path
        elif (train_path / 'images').exists():
            images_dir = train_path / 'images'
        else:
            images_dir = train_path
    else:
        base_path = Path(data_config['path'])
        images_dir = base_path / 'train' / 'images'
    
    if not images_dir.exists():
        logger.warning(f"Директория с изображениями не найдена: {images_dir}")
        return 0
    
    return len([f for f in images_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])


def train_ltdetr(config, run_cfg, models_dir, extra_model_args=None):
    from lightly_train import train_object_detection, load_model
    from experiments.scripts.evaluate import evaluate_model

    data_yaml_path = Path(config['paths']['experiment_data']) / run_cfg['data_yaml']
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml не найден: {data_yaml_path}")

    fixed_epochs = config['training'].get('fixed_epochs')
    if fixed_epochs is None:
        raise ValueError("training.fixed_epochs must be set in config")
    
    n_images = _count_dataset_images(data_yaml_path)
    if n_images == 0:
        raise ValueError(f"Не найдено изображений в датасете: {data_yaml_path}")
    
    batch_size = config['training']['batch_size']
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, n_images // effective_batch)
    max_steps = steps_per_epoch * fixed_epochs
    
    logger.info(f"Датасет {run_cfg['dataset_name']}: {n_images} изображений, {max_steps} шагов ({fixed_epochs} эпох)")

    model_args = {"lr": config['training'].get('lr', 1e-4)}
    model_args["backbone_freeze"] = run_cfg.get('freeze_backbone', False)

    if extra_model_args:
        model_args.update(extra_model_args)

    precision = "bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "16-mixed"

    train_params = {
        "out": str(models_dir),
        "model": config['training']['model'],
        "data": str(data_yaml_path),
        "seed": run_cfg['seed'],
        "precision": precision,
        "steps": max_steps,
        "overwrite": True,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "model_args": model_args,
        "logger_args": {"val_every_num_steps": config['training'].get('val_every_steps', 500)},
        "save_checkpoint_args": {"save_best": True, "save_last": True},
    }

    logger.info(f"Обучение: {run_cfg['run_name']}, freeze_backbone={model_args['backbone_freeze']}")
    
    start = time.time()
    train_object_detection(**train_params)
    training_time = (time.time() - start) / 3600

    val_metrics = _parse_train_log(models_dir)
    model_path = _find_model_path(models_dir)
    model = load_model(str(model_path))

    with open(data_yaml_path) as f:
        data_config = yaml.safe_load(f)
    
    test_path = data_config.get('test', data_config.get('val'))
    if isinstance(test_path, str):
        test_path = Path(test_path)
        if not test_path.is_absolute():
            test_path = Path(data_config['path']) / test_path

    if (test_path / "images").exists():
        test_images = test_path / "images"
        test_labels = test_path / "labels"
    else:
        test_images = test_path
        test_labels = test_path.parent / "labels" if (test_path.parent / "labels").exists() else test_path.parent

    eval_conf = config['training'].get('eval_conf_threshold', 0.001)
    
    metrics = evaluate_model(
        model, test_images=test_images, test_labels=test_labels,
        num_classes=config['classes']['num_classes'],
        conf_threshold=eval_conf,
    )

    result = {
        'run_name': run_cfg['run_name'],
        'dataset_name': run_cfg['dataset_name'],
        'strategy_name': run_cfg['strategy_name'],
        'seed': run_cfg['seed'],
        'test_map50': metrics.get('mAP_50', 0),
        'test_map75': metrics.get('mAP_75', 0),
        'test_map50_95': metrics.get('mAP_50_95', 0),
        'val_map50': val_metrics.get('best_val_map50', 0),
        'best_val_step': val_metrics.get('best_val_step', 0),
        'training_time_hours': round(training_time, 3),
        'model_path': str(model_path),
        'status': 'completed',
        'n_epochs': fixed_epochs,
        'n_steps': max_steps,
        'n_images': n_images,
    }

    for k, v in metrics.items():
        if k.startswith('cls'):
            result[f'test_{k}'] = v

    logger.info(f"Готово: {run_cfg['run_name']}: mAP50={result['test_map50']:.4f}")
    
    with open(models_dir / "result.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result