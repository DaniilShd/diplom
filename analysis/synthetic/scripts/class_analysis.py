"""
Смотрим, насколько равномерно классы распределены в синтетике.
Если какой-то класс проседает — модель будет к нему слепа.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict

from config import AnalysisConfig


def parse_yolo_label(label_path):
    """Вытаскивает class_id из YOLO-разметки."""
    if not label_path.exists():
        return [], 0

    classes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.append(int(parts[0]))

    return classes, len(classes)


def analyze(config: AnalysisConfig):
    """Основной анализ: статистика, графики, JSON-отчёт."""
    class_names = config.class_analysis.class_names
    num_classes = config.class_analysis.num_classes

    labels_dir = config.paths.synthetic_dir / "labels"
    images_dir = config.paths.synthetic_dir / "images"
    output_dir = config.paths.output_dir

    print("=" * 60)
    print("Распределение классов в синтетике")
    print("=" * 60)

    # Список разметки
    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"Файлов разметки: {len(label_files)}")

    image_ids = set()
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_ids.update(f.stem for f in images_dir.glob(ext))
    print(f"Изображений: {len(image_ids)}")

    # Сбор статистики
    class_counter = Counter()
    images_per_class = defaultdict(set)
    bboxes_per_image = []
    classes_per_image = []
    empty = 0
    bbox_areas = []
    class_areas = defaultdict(list)

    for label_file in label_files:
        classes, n_boxes = parse_yolo_label(label_file)
        img_name = label_file.stem

        if n_boxes == 0:
            empty += 1
            continue

        bboxes_per_image.append(n_boxes)
        classes_per_image.append(len(set(classes)))

        for cls in classes:
            class_counter[cls] += 1
            images_per_class[cls].add(img_name)

        # Размеры рамок (опционально)
        if config.class_analysis.compute_bbox_statistics:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        area = w * h
                        bbox_areas.append(area)
                        class_areas[cls_id].append(area)

    total_bboxes = sum(class_counter.values())
    images_with_defects = len(label_files) - empty

    # --- Вывод таблицы ---
    print(f"\n{'Класс':<25} {'Рамок':<8} {'%':<8} {'Изобр.':<8} {'Рамок/изобр.':<12}")
    print("-" * 60)

    class_stats = {}
    for cls_id in range(num_classes):
        n = class_counter[cls_id]
        pct = (n / total_bboxes * 100) if total_bboxes > 0 else 0
        n_img = len(images_per_class[cls_id])
        avg = n / max(n_img, 1)

        name = class_names.get(cls_id, f"cls_{cls_id}")
        print(f"{name:<25} {n:<8} {pct:<7.1f}% {n_img:<8} {avg:<12.2f}")

        class_stats[name] = {
            "class_id": cls_id,
            "bbox_count": n,
            "percentage": round(pct, 2),
            "image_count": n_img,
            "avg_bboxes_per_image": round(avg, 2),
        }

    # --- Графики ---
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F9ED69', '#F08A5D'][:num_classes]
    counts = [class_counter[i] for i in range(num_classes)]
    img_counts = [len(images_per_class[i]) for i in range(num_classes)]

    fig, axes = plt.subplots(2, 2, figsize=config.visualization.figsize)

    axes[0, 0].bar(range(num_classes), counts, color=colors, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_title('Рамок на класс')
    axes[0, 0].set_xticks(range(num_classes))
    axes[0, 0].set_xticklabels([class_names[i] for i in range(num_classes)],
                               rotation=45, ha='right')

    axes[0, 1].bar(range(num_classes), img_counts, color=colors, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_title('Изображений с классом')
    axes[0, 1].set_xticks(range(num_classes))
    axes[0, 1].set_xticklabels([class_names[i] for i in range(num_classes)],
                               rotation=45, ha='right')

    if bboxes_per_image:
        max_b = max(bboxes_per_image)
        axes[1, 0].hist(bboxes_per_image, bins=range(1, max_b + 2),
                        color='#3498db', edgecolor='black', align='left')
        axes[1, 0].set_title('Рамок на изображение')

    if classes_per_image:
        max_c = max(classes_per_image)
        axes[1, 1].bar(range(1, max_c + 1),
                       [classes_per_image.count(i) for i in range(1, max_c + 1)],
                       color='#e74c3c', edgecolor='black')
        axes[1, 1].set_title('Классов на изображение')

    plt.suptitle('Распределение классов — синтетика', fontsize=14, fontweight='bold')
    plt.tight_layout()
    viz_dir = output_dir / config.paths.subdirs.get('visualizations', 'visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)
    for fmt in config.visualization.save_formats:
        plt.savefig(viz_dir / f'class_distribution.{fmt}',
                    dpi=config.visualization.dpi, bbox_inches='tight')
    plt.close()

    # Круговая диаграмма
    if sum(counts) > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(counts, labels=[class_names[i] for i in range(num_classes)],
               autopct='%1.1f%%', colors=colors, startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Распределение классов', fontsize=14, fontweight='bold')
        for fmt in config.visualization.save_formats:
            plt.savefig(viz_dir / f'class_pie.{fmt}',
                        dpi=config.visualization.dpi, bbox_inches='tight')
        plt.close()

    # --- JSON-отчёт ---
    bbox_stats = {}
    if bbox_areas:
        areas_arr = np.array(bbox_areas)
        bbox_stats = {
            "mean": float(np.mean(areas_arr)),
            "std": float(np.std(areas_arr)),
            "min": float(np.min(areas_arr)),
            "max": float(np.max(areas_arr)),
            "median": float(np.median(areas_arr)),
            "by_class": {}
        }
        for cls_id in range(num_classes):
            if cls_id in class_areas:
                a = np.array(class_areas[cls_id])
                bbox_stats["by_class"][class_names[cls_id]] = {
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a)),
                    "median": float(np.median(a)),
                }

    results = {
        "summary": {
            "total_images": len(image_ids),
            "total_labels": len(label_files),
            "images_with_defects": images_with_defects,
            "empty_images": empty,
            "total_bboxes": total_bboxes,
            "avg_bboxes_per_image": total_bboxes / max(images_with_defects, 1),
        },
        "class_stats": class_stats,
        "bbox_statistics": bbox_stats,
    }

    json_path = output_dir / "class_distribution.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nСохранено: {json_path}")
    print(f"Графики: {viz_dir}")

    return results