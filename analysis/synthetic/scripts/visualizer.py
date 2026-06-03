"""
Визуализация: попарные сравнения, сетки, карты различий, примеры по классам.
"""
import random
from pathlib import Path

import cv2
import numpy as np

from config import AnalysisConfig
from utils.io_utils import read_yolo_labels

COLORS = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
}


def draw_bboxes(img, label_path):
    """Рисует YOLO-рамки на изображении."""
    if not label_path.exists():
        return img

    h, w = img.shape[:2]
    data = read_yolo_labels(label_path)

    for cls, bbox in zip(data['classes'], data['bboxes']):
        xc, yc, bw, bh = bbox
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        color = COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"cls {cls}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def side_by_side(orig_path, synth_path, orig_label, synth_label, size=640):
    """Склеивает оригинал и синтетику в одно изображение."""
    orig = cv2.imread(str(orig_path))
    synth = cv2.imread(str(synth_path))
    if orig is None or synth is None:
        return None

    orig = cv2.resize(orig, (size, size))
    synth = cv2.resize(synth, (size, size))
    orig = draw_bboxes(orig, orig_label)
    synth = draw_bboxes(synth, synth_label)

    h, w = size, size
    result = np.ones((h + 40, w * 2 + 10, 3), dtype=np.uint8) * 255
    result[20:20 + h, 0:w] = orig
    result[20:20 + h, w + 10:w * 2 + 10] = synth
    cv2.putText(result, "ORIGINAL", (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(result, "SYNTHETIC", (w + 20, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return result


def make_grid(image_paths, labels_dir, rows=4, cols=5, size=640):
    """Собирает сетку из изображений с рамками."""
    grid_h = rows * size + (rows + 1) * 10
    grid_w = cols * size + (cols + 1) * 10
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    for idx, img_path in enumerate(image_paths):
        if idx >= rows * cols:
            break
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        label = labels_dir / f"{img_path.stem}.txt"
        img = draw_bboxes(img, label)

        r, c = idx // cols, idx % cols
        y1 = 10 + r * (size + 10)
        x1 = 10 + c * (size + 10)
        grid[y1:y1 + size, x1:x1 + size] = img

    return grid


def difference_maps(orig_dir, synth_dir, output_dir, num_samples=10):
    """Карты разницы: оригинал, синтетика, absdiff, heatmap."""
    diff_dir = output_dir / "visualizations" / "difference_maps"
    diff_dir.mkdir(parents=True, exist_ok=True)

    orig_images = {}
    for ext in ['*.png', '*.jpg']:
        for p in (orig_dir / "images").glob(ext):
            orig_images[p.stem] = p

    synth_images = []
    for ext in ['*.png', '*.jpg']:
        synth_images.extend((synth_dir / "images").glob(ext))

    pairs = []
    for synth_path in synth_images:
        stem = synth_path.stem.replace('syn_', '').split('_v')[0]
        for orig_stem, orig_path in orig_images.items():
            if stem in orig_stem or orig_stem in stem:
                pairs.append((orig_path, synth_path))
                break

    for i, (orig_path, synth_path) in enumerate(pairs[:num_samples]):
        orig = cv2.imread(str(orig_path))
        synth = cv2.imread(str(synth_path))
        if orig is None or synth is None:
            continue

        size = 640
        orig = cv2.resize(orig, (size, size))
        synth = cv2.resize(synth, (size, size))

        diff = cv2.absdiff(orig, synth)
        diff_enh = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

        h, w = size, size
        composite = np.ones((h + 40, w * 4 + 30, 3), dtype=np.uint8) * 255
        composite[20:20 + h, 0:w] = orig
        composite[20:20 + h, w + 10:w * 2 + 10] = synth
        composite[20:20 + h, w * 2 + 20:w * 3 + 20] = diff_enh
        composite[20:20 + h, w * 3 + 30:w * 4 + 30] = heatmap

        for j, label in enumerate(["ORIG", "SYNTH", "DIFF", "HEATMAP"]):
            cv2.putText(composite, label, (10 + j * (w + 10), 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imwrite(str(diff_dir / f"diff_{i:03d}.png"), composite)

    print(f"Difference maps: {len(pairs[:num_samples])}")


def class_grids(synth_dir, output_dir):
    """Сетки примеров для каждого класса."""
    class_dir = output_dir / "visualizations" / "by_class"
    class_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = synth_dir / "labels"
    images_dir = synth_dir / "images"

    class_images = {i: [] for i in range(4)}
    for label_file in labels_dir.glob("*.txt"):
        data = read_yolo_labels(label_file)
        for cls in set(data['classes']):
            for ext in ['.png', '.jpg']:
                img_path = images_dir / f"{label_file.stem}{ext}"
                if img_path.exists():
                    class_images[cls].append(img_path)
                    break

    for cls, paths in class_images.items():
        if not paths:
            continue
        samples = random.sample(paths, min(16, len(paths)))
        grid = make_grid(samples, labels_dir, rows=4, cols=4)
        cv2.imwrite(str(class_dir / f"class_{cls}.png"), grid)

    print("Class grids saved")


def run_visualization(config: AnalysisConfig):
    """Точка входа."""
    orig_dir = config.paths.original_dir
    synth_dir = config.paths.synthetic_dir
    out_dir = config.paths.output_dir
    n = config.visualization.random_samples

    # Попарные сравнения
    comp_dir = out_dir / "visualizations" / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    orig_images = {}
    for ext in ['*.png', '*.jpg']:
        for p in (orig_dir / "images").glob(ext):
            orig_images[p.stem] = p

    synth_images = []
    for ext in ['*.png', '*.jpg']:
        synth_images.extend((synth_dir / "images").glob(ext))

    pairs = []
    for synth_path in synth_images[:n]:
        stem = synth_path.stem.replace('syn_', '').split('_v')[0]
        for orig_stem, orig_path in orig_images.items():
            if stem in orig_stem or orig_stem in stem:
                pairs.append((orig_path, synth_path))
                break

    for i, (orig_path, synth_path) in enumerate(pairs[:n]):
        comp = side_by_side(orig_path, synth_path,
                            orig_dir / "labels" / f"{orig_path.stem}.txt",
                            synth_dir / "labels" / f"{synth_path.stem}.txt")
        if comp is not None:
            cv2.imwrite(str(comp_dir / f"pair_{i:03d}.png"), comp)

    print(f"Comparisons: {len(pairs[:n])}")

    # Сетки
    viz_dir = out_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    orig_grid = make_grid(list(orig_images.values())[:20], orig_dir / "labels")
    cv2.imwrite(str(viz_dir / "grid_original.png"), orig_grid)

    synth_grid = make_grid(synth_images[:20], synth_dir / "labels")
    cv2.imwrite(str(viz_dir / "grid_synthetic.png"), synth_grid)
    print("Grids saved")

    # Карты различий
    difference_maps(orig_dir, synth_dir, out_dir, min(n, 10))

    # Сетки по классам
    class_grids(synth_dir, out_dir)