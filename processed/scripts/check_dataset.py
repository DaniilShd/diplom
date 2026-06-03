#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter


def check_dataset():
    base_path = Path("/app/data/processed/balanced_defect_patches_v2")
    
    print("Проверка датасета патчей")
    print(f"Путь: {base_path}\n")
    
    total_images = 0
    total_labels = 0
    split_stats = {}
    
    for split in ['train', 'val', 'test']:
        split_path = base_path / split
        images_dir = split_path / 'images'
        labels_dir = split_path / 'labels'
        
        n_images = len(list(images_dir.glob('*'))) if images_dir.exists() else 0
        n_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        split_stats[split] = {'images': n_images, 'labels': n_labels}
        total_images += n_images
        total_labels += n_labels
        
        print(f"  {split.upper()}: images={n_images}, labels={n_labels}")
        if n_images != n_labels:
            print(f"    НЕСООТВЕТСТВИЕ: images != labels")
    
    print(f"\n  ВСЕГО: images={total_images}, labels={total_labels}")
    
    print("\nПроверка дубликатов:")
    for split in ['train', 'val', 'test']:
        labels_dir = base_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        patches = [f.stem for f in labels_dir.glob('*.txt')]
        unique = set(patches)
        
        if len(patches) != len(unique):
            print(f"  {split}: {len(patches)} всего, {len(unique)} уникальных (дубликатов: {len(patches) - len(unique)})")
        else:
            print(f"  {split}: {len(patches)} (дубликатов нет)")
    
    print("\nПроверка утечки данных:")
    img_to_splits = defaultdict(set)
    
    for split in ['train', 'val', 'test']:
        labels_dir = base_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            patch_name = label_file.stem
            orig_img = patch_name.split('_x')[0] if '_x' in patch_name else patch_name
            img_to_splits[orig_img].add(split)
    
    leaked = {img: splits for img, splits in img_to_splits.items() if len(splits) > 1}
    
    if leaked:
        print(f"  Найдена утечка: {len(leaked)} изображений в нескольких сплитах")
    else:
        print(f"  Утечки нет. Все {len(img_to_splits)} изображений в одном сплите")
    
    print("\nАнализ классов:")
    class_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for split in ['train', 'val', 'test']:
        labels_dir = base_path / split / 'labels'
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file) as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_stats[class_id][split] += 1
    
    print(f"  {'Класс':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Всего':<10}")
    for class_id in sorted(class_stats.keys()):
        train_c = class_stats[class_id]['train']
        val_c = class_stats[class_id]['val']
        test_c = class_stats[class_id]['test']
        total = train_c + val_c + test_c
        print(f"  {class_id:<10} {train_c:<10} {val_c:<10} {test_c:<10} {total:<10}")
    
    print("\nПроверка RLE файлов:")
    for split in ['train', 'val', 'test']:
        rle_file = base_path / split / f"{split}_rle.csv"
        if rle_file.exists():
            df = pd.read_csv(rle_file)
            print(f"  {split}: {len(df)} записей, {df['ImageId'].nunique()} изображений")
    
    issues = []
    for split, stats in split_stats.items():
        if stats['images'] != stats['labels']:
            issues.append(f"{split}: images != labels")
    if leaked:
        issues.append(f"Data leakage: {len(leaked)} изображений")
    
    if issues:
        print(f"\nОбнаружены проблемы:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nВсе проверки пройдены")


if __name__ == "__main__":
    check_dataset()