#!/usr/bin/env python3
import logging
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.augmentation import get_metal_augmentation
from utils.dataset_utils import read_yolo_labels, write_yolo_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def augment_real_dataset(config):
    paths = config['paths']
    aug_cfg = config['augmentation']
    jpeg_quality = config['image']['jpeg_quality']
    
    output_dir = Path(paths['output_dir']) / "real_augmented" / "train"
    real_train = Path(paths['output_dir']) / "real" / "train"
    synth_train = Path(paths['output_dir']) / "synthetic" / "train"
    
    real_images_dir = real_train / 'images'
    synth_images_dir = synth_train / 'images'
    
    if not synth_images_dir.exists():
        raise FileNotFoundError(f"Синтетика не найдена: {synth_images_dir}")
    if not real_images_dir.exists():
        raise FileNotFoundError(f"Реальные данные не найдены: {real_images_dir}")
    
    num_synth = len(list(synth_images_dir.glob('*')))
    target = num_synth
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
    
    source_images = list(real_images_dir.glob('*.jpg')) + \
                    list(real_images_dir.glob('*.jpeg')) + \
                    list(real_images_dir.glob('*.png'))
    
    logger.info(f"real_augmented: цель={target}")
    
    transform = get_metal_augmentation(aug_cfg)
    
    generated = 0
    version_counter = {}
    attempts = 0
    max_attempts = target * 5
    
    pbar = tqdm(total=target, desc="  Генерация")
    
    while generated < target and attempts < max_attempts:
        attempts += 1
        
        src_img = random.choice(source_images)
        lbl_path = real_train / 'labels' / f"{src_img.stem}.txt"
        
        image = cv2.imread(str(src_img))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_yolo_labels(lbl_path)
        
        if bboxes:
            bboxes = np.clip(bboxes, 1e-7, 1.0 - 1e-7).tolist()
        
        try:
            if bboxes:
                augmented = transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
            else:
                augmented = transform(image=image_rgb, bboxes=[], class_labels=[])
                aug_image = augmented['image']
                aug_bboxes, aug_labels = [], []
        except Exception:
            continue
        
        if not aug_bboxes and bboxes:
            continue
        
        img_key = src_img.stem
        version_counter[img_key] = version_counter.get(img_key, 0) + 1
        
        new_name = f"real_aug_v{version_counter[img_key]:02d}_{src_img.stem}"
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / 'images' / f"{new_name}.jpg"), aug_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        write_yolo_labels(output_dir / 'labels' / f"{new_name}.txt", aug_bboxes, aug_labels)
        
        generated += 1
        pbar.update(1)
    
    pbar.close()
    
    final_count = len(list((output_dir / 'images').glob('*')))
    logger.info(f"real_augmented: {final_count}/{target}")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    augment_real_dataset(config)