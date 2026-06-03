#!/usr/bin/env python3
import logging
import random
import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from config import Config
from scripts.sd_generator import SDDefectGenerator
from utils.blending import apply_multiscale_blend
from utils.color_correction import adaptive_color_correction
from utils.rle_utils import rle_to_defect_bboxes
from utils.scaling import scale_defect_and_mask

logger = logging.getLogger(__name__)


class PoissonDefectGenerator:
    def __init__(self, config):
        self.config = config
        
        try:
            self.sd_generator = SDDefectGenerator(config)
        except Exception as e:
            logger.error(f"Ошибка загрузки SD генератора: {e}")
            raise
        
        random.seed(config.generation.random_seed)
        np.random.seed(config.generation.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.generation.random_seed)
    
    def process_image_all_defects(self, img_path, all_bboxes, output_dir, variant, total_idx):
        if self.sd_generator is None:
            return None
        
        try:
            original = cv2.imread(str(img_path))
            if original is None:
                return None
            
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            img_h, img_w = original.shape[:2]
            
            original_pil = Image.fromarray(original)
            bg_seed = self.config.generation.random_seed + total_idx * 1000 + variant * 100
            background = self.sd_generator.generate_background(original_pil, seed=bg_seed)
            background = np.clip(background, 0, 255).astype(np.uint8)
            
            result = background.copy().astype(np.float32)
            yolo_annotations = []
            
            for bbox_idx, bbox_info in enumerate(all_bboxes):
                x, y, w, h = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h']
                comp_mask = bbox_info['component_mask']
                
                pad = int(max(w, h) * 0.3)
                pad = max(pad, 4)
                
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img_w, x + w + pad)
                y2 = min(img_h, y + h + pad)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                if crop_w < 16 or crop_h < 16:
                    continue
                
                crop_comp_mask = comp_mask[y1:y2, x1:x2].copy()
                
                if crop_comp_mask.sum() == 0:
                    continue
                
                background_crop = background[y1:y2, x1:x2].copy()
                
                scale_factor = random.choice(self.config.scaling.factors)
                scaled_bg_crop, scaled_comp_mask, updated_bbox = scale_defect_and_mask(
                    background_crop, crop_comp_mask, scale_factor,
                    bbox_info, x1, y1, img_w, img_h
                )
                
                background_crop_pil = Image.fromarray(np.clip(scaled_bg_crop, 0, 255).astype(np.uint8))
                
                seed = self.config.generation.random_seed + total_idx * 100 + variant * 10 + bbox_idx
                
                try:
                    generated_np = self.sd_generator.generate_defect(background_crop_pil, seed=seed)
                except Exception as e:
                    logger.error(f"Ошибка SD генерации дефекта: {e}")
                    continue
                
                if generated_np.shape[:2] != (crop_h, crop_w):
                    generated_np = cv2.resize(generated_np, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
                
                if self.config.sd_defect.color_correction_strength > 0:
                    generated_np = adaptive_color_correction(
                        generated_np, background_crop.astype(np.float32),
                        scaled_comp_mask, self.config.sd_defect.color_correction_strength
                    )
                
                blended = apply_multiscale_blend(
                    generated_np, background_crop.astype(np.float32), scaled_comp_mask
                )
                
                if blended.shape[:2] != (crop_h, crop_w):
                    blended = cv2.resize(blended, (crop_w, crop_h))
                
                result[y1:y2, x1:x2] = blended.astype(np.float32)
                
                yolo_annotations.append({
                    'class': updated_bbox['class'],
                    'x_center': updated_bbox['x_center'],
                    'y_center': updated_bbox['y_center'],
                    'width': updated_bbox['width'],
                    'height': updated_bbox['height']
                })
            
            if not yolo_annotations:
                return None
            
            stem = img_path.stem
            filename = f"syn_{total_idx:06d}_{stem}_v{variant}.png"
            
            result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
            blur = cv2.GaussianBlur(result_uint8, (0, 0), sigmaX=1.0)
            result_uint8 = cv2.addWeighted(result_uint8, 1.1, blur, -0.1, 0)
            result_uint8 = np.clip(result_uint8, 0, 255).astype(np.uint8)
            
            result_bgr = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / "images" / filename), result_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            with open(output_dir / "labels" / f"{Path(filename).stem}.txt", 'w') as f:
                for ann in yolo_annotations:
                    f.write(f"{ann['class']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
            
            return {"filename": filename, "num_defects": len(yolo_annotations)}
        
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            traceback.print_exc()
            return None
    
    def generate_dataset(self, input_dir, rle_csv, output_dir, variants=3, limit=None):
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        rle_df = pd.read_csv(rle_csv)
        groups = rle_df.groupby('ImageId')
        
        image_ids = list(groups.groups.keys())
        rng = random.Random(self.config.generation.random_seed)
        rng.shuffle(image_ids)
        
        all_images = {}
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img in input_dir.glob(ext):
                all_images[img.name] = img
                all_images[img.stem] = img
        
        stats = {'total': 0, 'errors': 0, 'defects': 0}
        
        for image_id in tqdm(image_ids, desc="Generating"):
            if limit and stats['total'] >= limit:
                break
            
            group = groups.get_group(image_id)
            img_path = all_images.get(image_id)
            if img_path is None:
                continue
            
            all_bboxes = []
            for _, row in group.iterrows():
                rle = row.get('EncodedPixels')
                if pd.isna(rle) or str(rle).strip().lower() in ['', 'nan']:
                    continue
                all_bboxes.extend(rle_to_defect_bboxes(str(rle), int(row['ClassId']) - 1))
            
            if not all_bboxes:
                continue
            
            for v in range(variants):
                if limit and stats['total'] >= limit:
                    break
                
                result = self.process_image_all_defects(img_path, all_bboxes, output_dir, v, stats['total'])
                
                if result:
                    stats['total'] += 1
                    stats['defects'] += result['num_defects']
                else:
                    stats['errors'] += 1
                
                if stats['total'] % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info(f"Сгенерировано: {stats['total']} изображений, {stats['defects']} дефектов, {stats['errors']} ошибок")
        return stats['total']