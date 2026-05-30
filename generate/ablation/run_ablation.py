#!/usr/bin/env python3
import itertools
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path

import mlflow
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate.ablation.generate_synthetic import generate_synthetic_dataset
from generate.ablation.train_ltdetr import train_ltdetr
from generate.ablation.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_ablation_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_combinations(grid):
    keys = ['sd_defect_strength', 'sd_background_strength', 'high_freq_alpha', 'variants', 'balance_strategy']
    values = [grid[k] for k in keys]
    repeats = grid.get('repeats', 1)
    
    combos = []
    for i, combo in enumerate(itertools.product(*values)):
        params = dict(zip(keys, combo))
        for r in range(repeats):
            p = params.copy()
            p['run_id'] = f"abl_{i:03d}_r{r}"
            p['repeat'] = r
            combos.append(p)
    return combos


def count_dataset_images(directory):
    if not directory.exists():
        return 0
    return len([f for f in (directory / "images").glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])


def setup_dataset_dir(run_dir, real_train, real_val, real_test, synth_dir):
    ds_dir = run_dir / "dataset"
    
    for split, sources in [
        ('train', [synth_dir]),
        ('val', [real_val]),
        ('test', [real_test])
    ]:
        for sub in ['images', 'labels']:
            (ds_dir / split / sub).mkdir(parents=True, exist_ok=True)
        
        img_dst = ds_dir / split / 'images'
        lbl_dst = ds_dir / split / 'labels'
        
        for src in sources:
            s_img = src / 'images'
            s_lbl = src / 'labels'
            
            if not s_img.exists():
                continue
            
            for img_path in s_img.glob("*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                shutil.copy2(img_path, img_dst / img_path.name)
                
                lbl_path = s_lbl / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
    
    data_yaml = ds_dir / "data.yaml"
    with open(data_yaml, 'w') as f:
        yaml.dump({
            'format': 'yolo',
            'path': str(ds_dir),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'defect1', 1: 'defect2', 2: 'defect3', 3: 'defect4'}
        }, f)
    
    total_train_images = count_dataset_images(ds_dir / "train")
    
    return data_yaml, total_train_images


def main():
    config_path = Path(__file__).parent / "config.yaml"
    cfg = load_ablation_config(str(config_path))

    grid = cfg['grid']
    ltdetr_cfg = cfg['ltdetr']
    fixed = cfg['fixed_generation']
    paths = cfg['paths']
    
    real_train = Path(paths['real_train'])
    real_val = Path(paths['real_val'])
    real_test = Path(paths['real_test'])
    rle_csv = real_train / "train_rle.csv"
    results_base = Path(paths['results_dir'])
    results_base.mkdir(parents=True, exist_ok=True)
    
    combos = get_combinations(grid)
    logger.info(f"Total runs: {len(combos)}")
    
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])
    
    fixed_epochs = ltdetr_cfg.get('fixed_epochs')
    if fixed_epochs is None:
        raise ValueError("ltdetr.fixed_epochs must be set in config")
    
    all_results = []
    
    for idx, run_params in enumerate(combos):
        run_id = run_params['run_id']
        logger.info(f"Run {idx+1}/{len(combos)}: {run_id}")
        
        with mlflow.start_run(run_name=run_id):
            all_params = {
                **run_params,
                **{f"ltdetr_{k}": v for k, v in ltdetr_cfg.items() if k != 'fixed_epochs'},
                'fixed_epochs': fixed_epochs,
            }
            mlflow.log_params(all_params)
            mlflow.set_tag("run_type", "ablation")
            
            try:
                run_dir = results_base / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info("Step 1/4: Generating synthetic dataset")
                synth_dir = run_dir / "synthetic"
                total_synth = generate_synthetic_dataset(run_params, fixed, synth_dir, real_train, rle_csv)
                mlflow.log_metric("synthetic_images", total_synth)
                
                logger.info("Step 2/4: Preparing dataset")
                data_yaml, num_train_images = setup_dataset_dir(run_dir, real_train, real_val, real_test, synth_dir)
                
                batch_size = ltdetr_cfg['batch_size']
                steps_per_epoch = max(1, num_train_images // batch_size)
                max_steps = steps_per_epoch * fixed_epochs
                
                mlflow.log_metrics({
                    "num_train_images": num_train_images,
                    "computed_max_steps": max_steps,
                })
                
                logger.info("Step 3/4: Training LT-DETR")
                ltdetr_dir = run_dir / "ltdetr"
                train_result = train_ltdetr(
                    data_yaml=data_yaml,
                    out_dir=ltdetr_dir,
                    max_steps=max_steps,
                    val_every_steps=ltdetr_cfg.get('val_every_steps', 500),
                    lr=ltdetr_cfg['lr'],
                    batch_size=ltdetr_cfg['batch_size'],
                    freeze_backbone=ltdetr_cfg.get('freeze_backbone', False),
                    seed=ltdetr_cfg.get('seed', 42),
                )
                
                train_result['fixed_epochs'] = fixed_epochs
                train_result['computed_max_steps'] = max_steps
                
                train_log = ltdetr_dir / "train.log"
                if train_log.exists():
                    mlflow.log_artifact(str(train_log), "training_logs")
                
                result_json = ltdetr_dir / "result.json"
                if result_json.exists():
                    mlflow.log_artifact(str(result_json), "training_logs")
                
                logger.info("Step 4/4: Evaluating")
                metrics = {
                    'mAP_50': 0.0, 'mAP_75': 0.0, 'mAP_50_95': 0.0,
                    'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0,
                    'num_predictions': 0, 'num_ground_truth': 0
                }
                
                if train_result.get('model_path'):
                    try:
                        metrics = evaluate_model(
                            model_path=train_result['model_path'],
                            test_images=real_test / "images",
                            test_labels=real_test / "labels",
                        )
                    except Exception as e:
                        logger.error(f"Evaluation failed: {e}")
                
                metrics['training_time_h'] = train_result.get('training_time_hours', 0)
                metrics['computed_max_steps'] = max_steps
                metrics['effective_epochs'] = fixed_epochs
                
                mlflow.log_metrics(metrics)
                all_results.append({**run_params, **metrics})
                mlflow.set_tag("status", "completed")
                
            except Exception as e:
                logger.error(f"Run {run_id} failed: {e}")
                traceback.print_exc()
                
                mlflow.set_tag("status", "failed")
                mlflow.log_metric("failed", 1)
                
                all_results.append({
                    **run_params,
                    'mAP_50': 0.0, 'mAP_75': 0.0, 'mAP_50_95': 0.0,
                    'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0,
                    'training_time_h': 0, 'computed_max_steps': 0,
                    'effective_epochs': fixed_epochs, 'status': 'failed'
                })
            
            finally:
                torch.cuda.empty_cache()
    
    if all_results:
        summary_path = results_base / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        with mlflow.start_run(run_name="ablation_summary"):
            mlflow.log_artifact(str(summary_path), "summary")
            
            successful_runs = [r for r in all_results if r.get('status') != 'failed']
            
            if successful_runs:
                best = max(successful_runs, key=lambda x: x.get('mAP_50', 0))
                
                logger.info(f"Best run: {best['run_id']}")
                logger.info(f"  mAP_50={best.get('mAP_50', 0):.4f}, mAP_50_95={best.get('mAP_50_95', 0):.4f}")
                logger.info(f"  defect_strength={best.get('sd_defect_strength')}, bg_strength={best.get('sd_background_strength')}")
                logger.info(f"  variants={best.get('variants')}, balance={best.get('balance_strategy')}")
                
                mlflow.log_metrics({
                    'best_mAP_50': best.get('mAP_50', 0),
                    'best_mAP_50_95': best.get('mAP_50_95', 0),
                    'total_runs': len(all_results),
                    'successful_runs': len(successful_runs),
                })
            
            logger.info(f"Total: {len(all_results)} runs ({len(successful_runs)} successful)")


if __name__ == "__main__":
    main()