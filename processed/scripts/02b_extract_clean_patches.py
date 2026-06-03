#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_config, ensure_dir, print_section
from utils.rle_utils import rle_to_mask
from utils.clean_patch_utils import has_black_pixels, compute_clean_ratio
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()


def find_clean_patches():
    p = cfg['paths']
    pc = cfg['patch']
    
    out_dir = ensure_dir(p['clean_patches_dir'])
    
    df = pd.read_csv(p['train_csv'])
    
    masks = {}
    for img_id, group in tqdm(df.groupby('ImageId'), desc="Маски"):
        combined = np.zeros((256, 1600), dtype=np.uint8)
        for _, row in group.iterrows():
            combined = np.maximum(combined, rle_to_mask(row['EncodedPixels']))
        masks[img_id] = combined
    
    image_files = list(Path(p['train_images_dir']).glob("*.jpg"))
    
    x_positions = range(0, 1600 - pc['patch_size'] + 1, pc['stride'])
    y_positions = range(0, 256 - pc['patch_size'] + 1, pc['stride'])
    
    clean_patches = []
    rejected_black = 0
    rejected_defects = 0
    
    for img_path in tqdm(image_files, desc="Чистые патчи"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = masks.get(img_path.name, np.zeros((256, 1600), dtype=np.uint8))
        
        for y in y_positions:
            for x in x_positions:
                patch_img = img_rgb[y:y + pc['patch_size'], x:x + pc['patch_size']]
                patch_mask = mask[y:y + pc['patch_size'], x:x + pc['patch_size']]
                
                clean_ratio = compute_clean_ratio(patch_mask)
                if clean_ratio < 1.0:
                    rejected_defects += 1
                    continue
                
                if pc['reject_black'] and has_black_pixels(patch_img, pc['black_threshold']):
                    rejected_black += 1
                    continue
                
                if pc['resize_to'] != pc['patch_size']:
                    patch_img = cv2.resize(patch_img, (pc['resize_to'], pc['resize_to']), interpolation=cv2.INTER_CUBIC)
                
                name = f"{img_path.stem}_x{x}_y{y}_clean.{pc['save_format']}"
                cv2.imwrite(str(out_dir / name), cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 3] if pc['save_format'] == 'png' else [cv2.IMWRITE_JPEG_QUALITY, 95])
                clean_patches.append({'image': img_path.name, 'x': x, 'y': y, 'clean_ratio': clean_ratio})
    
    print(f"\nЧистых патчей: {len(clean_patches):,}")
    print(f"Отбраковано дефектных: {rejected_defects:,}")
    if pc['reject_black']:
        print(f"Отбраковано с чёрным: {rejected_black:,}")
    
    if clean_patches:
        pd.DataFrame(clean_patches).to_csv(out_dir / "clean_patches_info.csv", index=False)
        logger.info(f"Сохранено: {out_dir}")
    
    return clean_patches


def main():
    print_section("ИЗВЛЕЧЕНИЕ ЧИСТЫХ ПАТЧЕЙ")
    patches = find_clean_patches()
    if patches:
        logger.info(f"Найдено {len(patches):,} чистых патчей")


if __name__ == "__main__":
    main()