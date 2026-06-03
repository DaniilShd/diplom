#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import copy_real, copy_synthetic, augment_synthetic, augment_real, augment_real_controlled, merge_datasets, validate_resize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_augmentation_copies(config):
    paths = config['paths']
    output_dir = Path(paths['output_dir'])
    real_train = output_dir / "real" / "train"
    synth_train = output_dir / "synthetic" / "train"
    
    real_images = list(real_train.glob('images/*.jpg')) + list(real_train.glob('images/*.jpeg')) + list(real_train.glob('images/*.png'))
    synth_images = list(synth_train.glob('images/*.jpg')) + list(synth_train.glob('images/*.jpeg')) + list(synth_train.glob('images/*.png'))
    
    R = len(real_images)
    S = len(synth_images)
    k = S / R
    
    copies_3x = max(1, int(2 * k - 1))
    copies_massive = max(1, int(3 * k))
    
    return copies_3x, copies_massive, R, S, k


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    logger.info("Подготовка данных для эксперимента")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    mlflow_cfg = config['mlflow']
    mlflow.set_tracking_uri(mlflow_cfg['tracking_uri'])
    mlflow.set_experiment(mlflow_cfg['experiment_name'])
    
    with mlflow.start_run(run_name=config['experiment']['name']):
        mlflow.log_dict(config, "config.yaml")
        
        logger.info("Этап 1/7: Копирование реальных данных")
        real_paths = copy_real.copy_real_dataset(config)
        
        logger.info("Этап 2/7: Копирование синтетики")
        synth_path = copy_synthetic.copy_synthetic_dataset(config)
        
        copies_3x, copies_massive, R, S, k = compute_augmentation_copies(config)
        
        logger.info(f"Реальных: {R}, синтетики: {S}, k={k:.1f}")
        logger.info(f"Копий для 3x: {copies_3x}, для massive: {copies_massive}")
        
        mlflow.log_metrics({"R": R, "S": S, "k": k, "copies_3x": copies_3x, "copies_massive": copies_massive})
        
        logger.info("Этап 3/7: Аугментация синтетики")
        augment_synthetic.augment_synthetic_dataset(config)
        
        logger.info("Этап 4a/7: Базовая аугментация реальных")
        augment_real.augment_real_dataset(config)
        
        logger.info(f"Этап 4b/7: Аугментация реальных x{copies_3x}")
        augment_real_controlled.augment_real_controlled(config, copies_per_image=copies_3x, output_subdir="real_augmented_3x")
        
        logger.info(f"Этап 4c/7: Аугментация реальных x{copies_massive}")
        augment_real_controlled.augment_real_controlled(config, copies_per_image=copies_massive, output_subdir="real_augmented_massive")
        
        logger.info("Этап 5/7: Сборка датасетов")
        datasets = merge_datasets.merge_all_datasets(config)
        
        logger.info("Этап 6/7: Финальная проверка")
        stats = validate_resize.validate_and_fix_all_datasets(config)
        
        total_images = sum(s['total_images'] for s in stats.values())
        
        logger.info(f"Готово. Датасетов: {len(datasets)}, изображений: {total_images}")


if __name__ == "__main__":
    main()