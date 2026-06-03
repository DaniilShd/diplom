#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.dataset_utils import read_yolo_labels, write_yolo_labels, validate_yolo_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_and_fix_image(img_path, lbl_path, target_size=(640, 640), jpeg_quality=95):
    target_w, target_h = target_size
    result = {'path': str(img_path), 'was_resized': False, 'was_fixed': False, 'errors': []}
    
    image = cv2.imread(str(img_path))
    if image is None:
        result['errors'].append('Не удалось загрузить')
        return result
    
    h, w = image.shape[:2]
    
    if h != target_h or w != target_w:
        result['was_resized'] = True
        result['was_fixed'] = True
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    
    if lbl_path.exists():
        try:
            bboxes, class_labels = read_yolo_labels(lbl_path)
            
            fixed_bboxes = []
            fixed_labels = []
            
            for bbox, cls in zip(bboxes, class_labels):
                xc, yc, bw, bh = bbox
                xc = np.clip(xc, 0.0, 1.0)
                yc = np.clip(yc, 0.0, 1.0)
                bw = np.clip(bw, 0.001, 1.0)
                bh = np.clip(bh, 0.001, 1.0)
                
                if bw > 0.001 and bh > 0.001:
                    fixed_bboxes.append([xc, yc, bw, bh])
                    fixed_labels.append(cls)
            
            if len(fixed_bboxes) != len(bboxes):
                result['was_fixed'] = True
                write_yolo_labels(lbl_path, fixed_bboxes, fixed_labels)
                
        except Exception as e:
            result['errors'].append(f'Ошибка чтения лейбла: {e}')
            result['was_fixed'] = True
    
    return result


def validate_and_fix_all_datasets(config):
    paths = config['paths']
    target_size = tuple(config['image']['target_size'])
    jpeg_quality = config['image']['jpeg_quality']
    
    experiment_data = Path(paths['output_dir'])
    
    info_path = experiment_data / "datasets_info.yaml"
    if not info_path.exists():
        logger.error(f"Файл с информацией о датасетах не найден: {info_path}")
        return {}
    
    with open(info_path, 'r') as f:
        datasets_info = yaml.safe_load(f)
    
    logger.info(f"Проверка {len(datasets_info)} датасетов")
    
    stats = {}
    
    for name, yaml_path in datasets_info.items():
        dataset_dir = Path(yaml_path).parent
        
        dataset_stats = {'total_images': 0, 'resized': 0, 'fixed_labels': 0, 'errors': 0}
        
        all_image_paths = []
        for split in ['train', 'val', 'test']:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue
            all_image_paths.extend(list((split_dir / 'images').glob('*')))
        
        for img_path in tqdm(all_image_paths, desc=f"  {name}"):
            rel_path = img_path.parent.parent.name
            lbl_dir = dataset_dir / rel_path / 'labels'
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            
            result = check_and_fix_image(img_path, lbl_path, target_size, jpeg_quality)
            
            dataset_stats['total_images'] += 1
            if result['was_resized']:
                dataset_stats['resized'] += 1
            if result['was_fixed']:
                dataset_stats['fixed_labels'] += 1
            if result['errors']:
                dataset_stats['errors'] += 1
        
        is_valid, errors = validate_yolo_dataset(dataset_dir)
        dataset_stats['yolo_valid'] = is_valid
        
        if is_valid:
            logger.info(f"  {name}: {dataset_stats['total_images']} изобр., OK")
        else:
            logger.warning(f"  {name}: {dataset_stats['total_images']} изобр., {len(errors)} проблем")
        
        stats[name] = dataset_stats
    
    stats_path = experiment_data / "validation_stats.yaml"
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    logger.info(f"Проверка завершена. Всего изображений: {sum(s['total_images'] for s in stats.values())}")
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    validate_and_fix_all_datasets(config)