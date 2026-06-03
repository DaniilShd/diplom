#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import json
import shutil
import random
import ast
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from utils import load_config, ensure_dir, print_section
from utils.report_utils import save_figure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
cfg = load_config()

np.random.seed(cfg['split']['random_seed'])
random.seed(cfg['split']['random_seed'])


def load_metadata():
    p = cfg['paths']
    meta = pd.read_csv(Path(p['defect_patches_dir']) / 'patches_metadata.csv')
    with open(Path(p['defect_patches_dir']) / 'annotations.json') as f:
        ann = json.load(f)
    return meta, ann


def get_file_mappings():
    p = cfg['paths']
    src_img = Path(p['defect_patches_dir']) / p['yolo_images_subdir']
    src_lbl = Path(p['defect_patches_dir']) / p['yolo_labels_subdir']
    
    real_img = {}
    for ext in ['*.png', '*.jpg']:
        real_img.update({f.stem: f.name for f in src_img.glob(ext)})
    real_lbl = {f.stem: f.name for f in src_lbl.glob("*.txt")}
    
    return src_img, src_lbl, real_img, real_lbl


def parse_classes_from_metadata(meta):
    patches_by_class = defaultdict(list)
    
    for _, row in meta.iterrows():
        name = row['saved_as']
        val = row.get('classes_present', '[]')
        classes = set()
        
        if isinstance(val, str):
            try:
                classes = set(ast.literal_eval(val))
            except (ValueError, SyntaxError):
                classes = {int(x.strip()) for x in val.strip('[]').split(',') if x.strip()}
        else:
            classes = set(val if isinstance(val, list) else [val])
        
        for c in classes:
            patches_by_class[c].append(name)
    
    return patches_by_class


def select_balanced_no_leakage(patches_by_class):
    sp = cfg['split']
    multiplier = sp.get('balance_multiplier', 2.0)
    
    image_to_patches = defaultdict(list)
    for class_id, patches in patches_by_class.items():
        for patch in patches:
            orig_img = patch.split('_x')[0] if '_x' in patch else patch
            image_to_patches[orig_img].append(patch)
    
    min_cls = min(patches_by_class, key=lambda x: len(patches_by_class[x]))
    min_cnt = len(patches_by_class[min_cls])
    logger.info(f"Миноритарный класс: {min_cls}, {min_cnt} патчей")
    
    target_per_class = {}
    for cls in patches_by_class:
        target_per_class[cls] = min_cnt if cls == min_cls else min(int(min_cnt * multiplier), len(patches_by_class[cls]))
    
    all_images = list(image_to_patches.keys())
    random.shuffle(all_images)
    
    n_train = int(len(all_images) * sp['train_ratio'])
    n_val = int(len(all_images) * sp['val_ratio'])
    
    split_assignment = {}
    for i, img in enumerate(all_images):
        if i < n_train:
            split_assignment[img] = 'train'
        elif i < n_train + n_val:
            split_assignment[img] = 'val'
        else:
            split_assignment[img] = 'test'
    
    selected = {'train': [], 'val': [], 'test': []}
    sel_by_class = defaultdict(lambda: {'train': [], 'val': [], 'test': []})
    
    for cls, target in target_per_class.items():
        patches_by_img = defaultdict(list)
        for patch in patches_by_class[cls]:
            orig_img = patch.split('_x')[0] if '_x' in patch else patch
            patches_by_img[orig_img].append(patch)
        
        for split in ['train', 'val', 'test']:
            imgs_in_split = [img for img in patches_by_img.keys() if split_assignment.get(img) == split]
            
            patches_in_split = []
            for img in imgs_in_split:
                patches_in_split.extend(patches_by_img[img])
            
            target_count = int(target * {'train': 0.7, 'val': 0.15, 'test': 0.15}[split])
            
            if len(patches_in_split) < target_count:
                needed = target_count - len(patches_in_split)
                other_patches = [p for p in patches_by_class[cls] if p not in patches_in_split]
                random.shuffle(other_patches)
                patches_in_split.extend(other_patches[:needed])
            
            selected[split].extend(patches_in_split[:target_count])
            sel_by_class[cls][split] = patches_in_split[:target_count]
    
    for split in ['train', 'val', 'test']:
        selected[split] = list(set(selected[split]))
    
    return selected, sel_by_class, min_cls, min_cnt


def copy_patches(selected, src_img, src_lbl, real_img, real_lbl, out_dir):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    
    for sn in ['train', 'val', 'test']:
        for sub in ['images', 'labels']:
            ensure_dir(out_dir / sn / sub)
        
        for name in selected[sn]:
            base = name.replace('.png', '').replace('.jpg', '')
            if base in real_img:
                shutil.copy2(src_img / real_img[base], out_dir / sn / 'images' / real_img[base])
            if base in real_lbl:
                shutil.copy2(src_lbl / real_lbl[base], out_dir / sn / 'labels' / real_lbl[base])


def plot_balance_summary(patches_by_class, sel_by_class, rpt):
    all_ids = sorted(sel_by_class.keys())
    names_list = [cfg['classes']['names'].get(i, f"Cls_{i}") for i in all_ids]
    
    fig, axes = plt.subplots(1, 2, figsize=cfg['report']['figsize'])
    x = np.arange(len(all_ids))
    w = 0.35
    
    orig_cnt = [len(patches_by_class[i]) for i in all_ids]
    sel_cnt = [sum(len(sel_by_class[i][s]) for s in ['train', 'val', 'test']) for i in all_ids]
    train_cnt = [len(sel_by_class[i]['train']) for i in all_ids]
    val_cnt = [len(sel_by_class[i]['val']) for i in all_ids]
    test_cnt = [len(sel_by_class[i]['test']) for i in all_ids]
    
    axes[0].bar(x - w/2, orig_cnt, w, label='Исходное', color='#3498db', alpha=0.7)
    axes[0].bar(x + w/2, sel_cnt, w, label='Отобранное', color='#e74c3c', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names_list, rotation=45, ha='right')
    axes[0].set_title('До / После балансировки')
    axes[0].legend()
    
    axes[1].bar(x - w, train_cnt, w, label='Train', color='#2ecc71')
    axes[1].bar(x, val_cnt, w, label='Val', color='#f39c12')
    axes[1].bar(x + w, test_cnt, w, label='Test', color='#e74c3c')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names_list, rotation=45, ha='right')
    axes[1].set_title(f'Разбиение')
    axes[1].legend()
    
    plt.suptitle('Балансировка по патчам', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "balance_defect_split.png", rpt, cfg['report']['dpi'])
    plt.close()


def main():
    print_section("БАЛАНСИРОВКА ПАТЧЕЙ")
    
    p = cfg['paths']
    out = ensure_dir(p['balanced_defect_patches_dir'])
    rpt = ensure_dir(p['reports_dir'])
    
    meta, ann = load_metadata()
    src_img, src_lbl, real_img, real_lbl = get_file_mappings()
    
    patches_by_class = parse_classes_from_metadata(meta)
    
    selected, sel_by_class, min_cls, min_cnt = select_balanced_no_leakage(patches_by_class)
    
    total_sel = sum(len(v) for v in selected.values())
    logger.info(f"Отобрано: {total_sel} патчей (train={len(selected['train'])}, val={len(selected['val'])}, test={len(selected['test'])})")
    
    copy_patches(selected, src_img, src_lbl, real_img, real_lbl, out)
    plot_balance_summary(patches_by_class, sel_by_class, rpt)
    
    logger.info(f"Готово: {out}")


if __name__ == "__main__":
    main()