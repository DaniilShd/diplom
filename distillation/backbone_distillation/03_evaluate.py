#!/usr/bin/env python3
# 03_evaluate.py
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm

from utils.dataset import load_yolo_gt
from utils.metrics import measure_fps, model_stats, predict_teacher, predict_student, evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("03_evaluate.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def analyze_dataset_difficulty(image_files, labels_dir, num_classes, img_size, class_names=None):
    stats = {
        'num_images': len(image_files),
        'total_objects': 0,
        'small': 0,
        'medium': 0,
        'large': 0,
        'objects_per_image': [],
        'objects_per_class': defaultdict(int),
        'areas': [],
    }
    
    for img_path in tqdm(image_files, desc="Анализ датасета"):
        gt = load_yolo_gt(img_path, labels_dir, num_classes, img_size)
        boxes = gt['boxes']
        labels = gt['labels']
        
        if len(boxes) == 0:
            stats['objects_per_image'].append(0)
            continue
        
        stats['total_objects'] += len(boxes)
        stats['objects_per_image'].append(len(boxes))
        
        for box, label in zip(boxes, labels):
            area = ((box[2] - box[0]) * (box[3] - box[1])).item()
            stats['areas'].append(area)
            stats['objects_per_class'][label.item()] += 1
            
            if area < 32**2:
                stats['small'] += 1
            elif area < 96**2:
                stats['medium'] += 1
            else:
                stats['large'] += 1
    
    total = stats['total_objects']
    stats['pct_small'] = stats['small'] / max(total, 1) * 100
    stats['pct_medium'] = stats['medium'] / max(total, 1) * 100
    stats['pct_large'] = stats['large'] / max(total, 1) * 100
    stats['avg_objects_per_image'] = np.mean(stats['objects_per_image'])
    stats['median_objects_per_image'] = np.median(stats['objects_per_image'])
    stats['max_objects_per_image'] = max(stats['objects_per_image']) if stats['objects_per_image'] else 0
    
    if stats['areas']:
        stats['avg_box_area'] = np.mean(stats['areas'])
        stats['median_box_area'] = np.median(stats['areas'])
        stats['min_box_area'] = np.min(stats['areas'])
        stats['max_box_area'] = np.max(stats['areas'])
    
    logger.info(f"Изображений: {stats['num_images']}, объектов: {stats['total_objects']}")
    logger.info(f"В среднем объектов: {stats['avg_objects_per_image']:.1f}")
    logger.info(f"Размеры: small={stats['pct_small']:.1f}%, medium={stats['pct_medium']:.1f}%, large={stats['pct_large']:.1f}%")
    
    if stats['pct_small'] > 50:
        logger.info("Преобладают мелкие объекты (>50%) - ожидайте низкие mAP@50:95")
    elif stats['pct_small'] > 30:
        logger.info("Значительная доля мелких объектов (30-50%)")
    
    return {k: v for k, v in stats.items() if k not in ['objects_per_image', 'areas']}


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    
    if not cfg_path.exists():
        logger.error(f"Конфиг не найден: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Устройство: {device}")

    num_classes = cfg["detection"]["num_classes"]
    img_size = tuple(cfg["detection"]["img_size"])
    class_names = list(cfg["detection"].get("class_names", {}).values())
    data_path = Path(cfg["detection"]["data_path"])

    test_imgs = data_path / "test" / "images"
    test_lbls = data_path / "test" / "labels"
    if not test_imgs.exists():
        logger.warning("test/ не найден, используется val/")
        test_imgs = data_path / "val" / "images"
        test_lbls = data_path / "val" / "labels"

    if not test_imgs.exists():
        logger.error("Нет изображений для оценки")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(f for f in test_imgs.glob("*") if f.suffix.lower() in exts)
    
    if not image_files:
        logger.error("Нет изображений в тестовой выборке")
        sys.exit(1)
    
    logger.info(f"Изображений для оценки: {len(image_files)}")

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    dataset_stats = analyze_dataset_difficulty(
        image_files, test_lbls, num_classes, img_size, class_names
    )
    
    stats_path = results_dir / "dataset_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    logger.info(f"Статистика датасета сохранена: {stats_path}")

    # Оценка учителя
    detector_path = cfg["teacher"].get("detector_path")
    if detector_path and Path(detector_path).exists():
        logger.info(f"Оценка учителя: {detector_path}")

        try:
            import lightly_train
            teacher_model = lightly_train.load_model(detector_path)
            teacher_model.eval()

            predict_fn = lambda p: predict_teacher(teacher_model, p, img_size)

            metrics = evaluate_model(predict_fn, image_files, test_lbls, num_classes, img_size, class_names)
            fps = measure_fps(teacher_model, image_files[0], img_size, device, cfg["fps"]["warmup"], cfg["fps"]["iterations"])
            stats = model_stats(teacher_model)

            result = {"model": "teacher_ltdetr", "type": "teacher", **metrics, **fps, **stats}
            all_results.append(result)
            logger.info(f"Учитель: mAP50:95={metrics['map50_95']:.4f}, mAP50={metrics['map50']:.4f}, FPS={fps['fps']:.1f}")
        except Exception as e:
            logger.error(f"Ошибка оценки учителя: {e}", exc_info=True)

    # Оценка учеников
    for name, group_cfg in cfg["students"].items():
        det_out = Path(cfg["paths"]["detection_output"]) / name
        
        ckpt = det_out / "model_final.pth"
        if not ckpt.exists():
            ckpt = det_out / "best_model.pth"
        if not ckpt.exists():
            logger.warning(f"Чекпоинт не найден: {name}")
            continue

        logger.info(f"Оценка: {name} ({group_cfg['type']})")

        try:
            backbone = resnet_fpn_backbone("resnet18", pretrained=False)
            model = FasterRCNN(backbone, num_classes=num_classes + 1)
            
            ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = ckpt_data.get("model_state_dict", ckpt_data)
            model.load_state_dict(sd)
            model.to(device).eval()

            predict_fn = lambda p, m=model: predict_student(m, p, img_size, device)

            metrics = evaluate_model(predict_fn, image_files, test_lbls, num_classes, img_size, class_names)
            fps = measure_fps(model, image_files[0], img_size, device, cfg["fps"]["warmup"], cfg["fps"]["iterations"])
            stats = model_stats(model)

            result = {"model": name, "type": group_cfg["type"], **metrics, **fps, **stats}
            all_results.append(result)
            logger.info(f"{name}: mAP50:95={metrics['map50_95']:.4f}, mAP50={metrics['map50']:.4f}, FPS={fps['fps']:.1f}")
        except Exception as e:
            logger.error(f"Ошибка оценки {name}: {e}", exc_info=True)

    out_path = results_dir / "evaluation.json"
    final_results = {
        "dataset_analysis": dataset_stats,
        "models": all_results,
        "evaluation_params": {
            "num_test_images": len(image_files),
            "img_size": img_size,
            "num_classes": num_classes,
            "class_names": class_names,
        }
    }
    out_path.write_text(json.dumps(final_results, indent=2))
    logger.info(f"Результаты сохранены: {out_path}")

    if all_results:
        logger.info(f"\n{'Модель':<35} {'mAP50:95':>10} {'mAP50':>8} {'mAP75':>8} {'FPS':>7} {'Params':>8} {'Size':>7}")
        logger.info("-" * 90)

        for r in sorted(all_results, key=lambda x: x.get("map50_95", 0), reverse=True):
            logger.info(
                f"{r['model']:<35} {r.get('map50_95',0):>10.4f} "
                f"{r.get('map50',0):>8.4f} {r.get('map75',0):>8.4f} "
                f"{r.get('fps',0):>7.1f} {r.get('params_M',0):>7.1f}M "
                f"{r.get('size_mb',0):>6.1f}MB"
            )

        teacher_r = next((r for r in all_results if r.get("type") == "teacher"), None)
        distilled_r = next((r for r in all_results if r.get("type") == "lightly_pretrained"), None)
        scratch_r = next((r for r in all_results if r.get("type") == "scratch"), None)
        imagenet_r = next((r for r in all_results if r.get("type") == "imagenet_pretrained"), None)

        if distilled_r and teacher_r:
            ratio = distilled_r["map50_95"] / max(teacher_r["map50_95"], 1e-6) * 100
            speedup = distilled_r["fps"] / max(teacher_r["fps"], 1e-6)
            logger.info(f"Студент сохраняет {ratio:.1f}% точности учителя, ускорение ×{speedup:.1f}")

        if scratch_r and imagenet_r and distilled_r:
            logger.info(f"Сравнение: Scratch={scratch_r.get('map50_95', 0):.4f}, ImageNet={imagenet_r.get('map50_95', 0):.4f}, Distilled={distilled_r.get('map50_95', 0):.4f}")


if __name__ == "__main__":
    main()