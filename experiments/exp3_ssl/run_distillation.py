#!/usr/bin/env python3
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.scripts.train_ltdetr import train_ltdetr
from experiments.scripts.statistical_analysis import run_statistical_analysis
from experiments.scripts.visualize import create_all_visualizations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "exp3_ssl"
STRATEGY_NAME = "ssl"


def _prepare_mlflow_metrics(result):
    metrics = {
        'test_map50': result.get('test_map50', 0),
        'test_map75': result.get('test_map75', 0),
        'test_map50_95': result.get('test_map50_95', 0),
        'val_map50': result.get('val_map50', 0),
        'training_time_hours': result.get('training_time_hours', 0),
        'n_epochs': result.get('n_epochs', 0),
        'n_images': result.get('n_images', 0),
    }
    for k, v in result.items():
        if k.startswith('test_cls'):
            metrics[k] = v
    return metrics


def ssl_pretrain_for_dataset(config, models_dir, ds_name, seed):
    from lightly_train import pretrain

    ssl_out = models_dir / f"ssl_pretrain_{ds_name}_seed{seed}"
    experiment_data = Path(config['paths']['experiment_data'])
    unlabeled_path = experiment_data / ds_name / "train" / "images"
    
    if not unlabeled_path.exists():
        raise FileNotFoundError(f"No images for SSL: {unlabeled_path}")
    
    n_images = len(list(unlabeled_path.glob("*")))
    
    backbone = config['ssl']['backbone']
    teacher = config['ssl']['teacher']
    
    ssl_epochs = config['ssl'].get('epochs_per_dataset', {}).get(ds_name, config['ssl'].get('epochs', 400))
    
    logger.info(f"SSL distillation for {ds_name}: {n_images} images, epochs={ssl_epochs}")
    
    pretrain(
        out=str(ssl_out),
        data=str(unlabeled_path),
        model=backbone,
        method=config['ssl']['method'],
        method_args={"teacher": teacher},
        epochs=ssl_epochs,
        batch_size=config['ssl']['batch_size'],
        seed=seed,
        overwrite=True,
    )

    backbone_path = ssl_out / "exported_models" / "exported_last.pt"
    if not backbone_path.exists():
        raise FileNotFoundError(f"SSL backbone not found: {backbone_path}")
    
    logger.info(f"Distilled backbone saved: {backbone_path}")
    return backbone_path


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    datasets = config['datasets']
    seeds = config['seeds']
    total_runs = len(datasets) * len(seeds)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(f"{config['mlflow']['experiment_name']}_{EXPERIMENT_NAME}")

    results_dir = Path(config['paths']['results_dir']) / EXPERIMENT_NAME
    models_dir = Path(config['paths']['models_dir']) / EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    completed_count = 0

    with mlflow.start_run(run_name=EXPERIMENT_NAME):
        mlflow.log_param("experiment", EXPERIMENT_NAME)
        mlflow.log_param("strategy", STRATEGY_NAME)
        mlflow.log_param("total_runs", total_runs)
        mlflow.log_param("ssl_method", config['ssl'].get('method', 'distillation'))

        for (ds_name, ds_cfg), seed in itertools.product(datasets.items(), seeds):
            run_cfg = {
                'run_name': f"{ds_name}_{STRATEGY_NAME}_seed{seed}",
                'dataset_name': ds_name,
                'data_yaml': ds_cfg['data_yaml'],
                'strategy_name': STRATEGY_NAME,
                'freeze_backbone': False,
                'seed': seed,
            }

            logger.info(f"Запуск: {run_cfg['run_name']}")

            with mlflow.start_run(run_name=run_cfg['run_name'], nested=True):
                mlflow.log_params(run_cfg)
                
                try:
                    backbone_path = ssl_pretrain_for_dataset(config, models_dir, ds_name, seed)
                    mlflow.log_param("ssl_backbone_path", str(backbone_path))
                    
                    result = train_ltdetr(
                        config, run_cfg, models_dir / run_cfg['run_name'],
                        extra_model_args={"backbone_weights": str(backbone_path)},
                    )
                    
                    all_results.append(result)
                    completed_count += 1
                    mlflow.log_metrics(_prepare_mlflow_metrics(result))
                    mlflow.set_tag("status", "completed")
                    
                except Exception as e:
                    logger.error(f"Ошибка в {run_cfg['run_name']}: {e}", exc_info=True)
                    mlflow.set_tag("status", "failed")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))

        logger.info(f"Завершено: {completed_count}/{total_runs}")


if __name__ == "__main__":
    main()