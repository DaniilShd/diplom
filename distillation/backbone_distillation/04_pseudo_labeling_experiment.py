#!/usr/bin/env python3
# 04_pseudo_experiment.py
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from tqdm import tqdm

import lightly_train
from utils.dataset import YOLODataset, collate_fn, load_yolo_gt
from utils.backbone_loader import load_lightly_backbone
from utils.metrics import measure_fps, model_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pseudo_experiment.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Конфиг не найден: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    cfg["pseudo_labeling"] = {
        "enabled": True,
        "raw_patches_dir": "/app/data/processed/defect_patches/images/train",
        "pseudo_count": 4000,
        "pseudo_conf": 0.7,
        "seed": 42,
    }
    
    cfg["paths"]["pseudo_output"] = "results/pseudo_experiment"
    
    return cfg


class PseudoLabelGenerator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = None
        
    def load_teacher(self):
        teacher_path = self.cfg["teacher"]["detector_path"]
        logger.info(f"Загрузка учителя: {teacher_path}")
        
        if not Path(teacher_path).exists():
            raise FileNotFoundError(f"Модель учителя не найдена: {teacher_path}")
        
        self.teacher = lightly_train.load_model(teacher_path)
        self.teacher.to(self.device)
        self.teacher.eval()
        
        for p in self.teacher.parameters():
            p.requires_grad = False
    
    def generate(self):
        pseudo_cfg = self.cfg["pseudo_labeling"]
        output_dir = Path(self.cfg["paths"]["pseudo_output"]) / "pseudo_dataset"
        img_dir = output_dir / "images"
        lbl_dir = output_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        patches_dir = Path(pseudo_cfg["raw_patches_dir"])
        if not patches_dir.exists():
            logger.error(f"Директория с патчами не найдена: {patches_dir}")
            sys.exit(1)
        
        all_patches = list(patches_dir.glob("*.jpg")) + list(patches_dir.glob("*.png"))
        logger.info(f"Найдено патчей: {len(all_patches)}")
        
        random.seed(pseudo_cfg["seed"])
        selected = random.sample(all_patches, min(pseudo_cfg["pseudo_count"], len(all_patches)))
        logger.info(f"Выбрано для псевдоразметки: {len(selected)}")
        
        if self.teacher is None:
            self.load_teacher()
        
        stats = {
            "total_processed": 0,
            "total_boxes": 0,
            "class_distribution": defaultdict(int),
            "failed_images": [],
        }
        
        for patch_path in tqdm(selected, desc="Pseudo-labeling"):
            try:
                results = self.teacher.predict(str(patch_path))
                
                boxes = results.get("bboxes", [])
                labels = results.get("labels", [])
                scores = results.get("scores", [])
                
                if len(scores) > 0:
                    mask = scores > pseudo_cfg["pseudo_conf"]
                    boxes = boxes[mask] if hasattr(boxes, '__getitem__') else [b for b, m in zip(boxes, mask) if m]
                    labels = labels[mask] if hasattr(labels, '__getitem__') else [l for l, m in zip(labels, mask) if m]
                
                if len(boxes) == 0:
                    continue
                
                img = Image.open(patch_path).convert("RGB")
                w, h = img.size
                
                yolo_lines = []
                for box, label in zip(boxes, labels):
                    if hasattr(box, 'tolist'):
                        box = box.tolist()
                    if hasattr(label, 'item'):
                        label = label.item()
                    
                    x1, y1, x2, y2 = box
                    cls_id = int(label)
                    
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    if 0 <= cx <= 1 and 0 <= cy <= 1 and bw > 0 and bh > 0:
                        yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        stats["class_distribution"][str(cls_id)] += 1
                
                if yolo_lines:
                    img.save(img_dir / patch_path.name)
                    with open(lbl_dir / f"{patch_path.stem}.txt", "w") as f:
                        f.write("\n".join(yolo_lines))
                    
                    stats["total_processed"] += 1
                    stats["total_boxes"] += len(yolo_lines)
                    
            except Exception as e:
                stats["failed_images"].append(str(patch_path.name))
        
        stats["failed_images"] = stats["failed_images"][:10]
        stats_path = output_dir / "pseudo_generation_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Псевдоразметка завершена: обработано {stats['total_processed']}, боксов {stats['total_boxes']}")
        
        return output_dir


class PseudoExperimentTrainer:
    def __init__(self, cfg, name, group_cfg, train_loader, val_loader, backbone_ckpt=None):
        self.cfg = cfg
        self.name = name
        self.group_cfg = group_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.backbone_ckpt = backbone_ckpt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["detection"]["num_classes"]
        
        self.model = self._build_model()
        self.model.to(self.device)
        
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=group_cfg["lr"],
            weight_decay=group_cfg.get("weight_decay", 5e-4)
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=group_cfg["epochs"], eta_min=1e-6
        )
        
        self.out_dir = Path(cfg["paths"]["pseudo_output"]) / "detectors" / name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_map = 0.0
        self.best_epoch = 0
        self.patience = group_cfg.get("patience", 15)
        self._patience_cnt = 0
    
    def _build_model(self):
        init_type = self.group_cfg["type"]
        logger.info(f"[{self.name}] init={init_type}")
        
        backbone = resnet_fpn_backbone("resnet18", pretrained=False)
        model = FasterRCNN(backbone, num_classes=self.num_classes + 1)
        
        if init_type == "imagenet_pretrained":
            pretrained_bb = resnet_fpn_backbone("resnet18", pretrained=True)
            model.backbone.load_state_dict(pretrained_bb.state_dict())
        elif init_type == "lightly_pretrained":
            if self.backbone_ckpt and Path(self.backbone_ckpt).exists():
                load_lightly_backbone(model, self.backbone_ckpt)
            else:
                logger.warning(f"Чекпоинт не найден: {self.backbone_ckpt}")
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Параметры: {total/1e6:.1f}M всего, {trainable/1e6:.1f}M trainable")
        
        return model
    
    def train(self):
        logger.info(f"Обучение: {self.name}")
        
        history = []
        best_state = None
        start_time = time.time()
        
        for epoch in range(1, self.group_cfg["epochs"] + 1):
            self.model.train()
            total_loss = 0.0
            
            for images, targets in self.train_loader:
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
                
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            
            metrics = self._validate()
            map50_95 = metrics["map"].item()
            map50 = metrics["map_50"].item()
            map75 = metrics["map_75"].item()
            
            is_best = map50_95 > self.best_map
            logger.info(
                f"Epoch {epoch:3d}/{self.group_cfg['epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"mAP50:95={map50_95:.4f} | "
                f"mAP50={map50:.4f} | "
                f"mAP75={map75:.4f}"
                f"{' ***' if is_best else ''}"
            )
            
            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "map50_95": map50_95,
                "map50": map50,
                "map75": map75,
                "lr": self.opt.param_groups[0]["lr"],
            })
            
            if is_best:
                self.best_map = map50_95
                self.best_epoch = epoch
                self._patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self._save("best_model.pth")
            else:
                self._patience_cnt += 1
                if self._patience_cnt >= self.patience:
                    logger.info(f"Early stopping на эпохе {epoch}")
                    break
            
            self.sched.step()
        
        if best_state:
            self.model.load_state_dict(best_state)
        self._save("model_final.pth")
        
        training_time = (time.time() - start_time) / 3600
        
        result = {
            "model_name": self.name,
            "init_type": self.group_cfg["type"],
            "best_map50_95": self.best_map,
            "best_map50": max(h["map50"] for h in history),
            "best_map75": max(h["map75"] for h in history),
            "best_epoch": self.best_epoch,
            "epochs_trained": len(history),
            "training_time_hours": round(training_time, 2),
        }
        
        (self.out_dir / "history.json").write_text(
            json.dumps({"history": history, "summary": result}, indent=2)
        )
        
        logger.info(f"Готово: {self.name} | mAP50:95={self.best_map:.4f} | {training_time:.1f}ч")
        
        return result
    
    def _validate(self):
        from torchmetrics.detection import MeanAveragePrecision
        
        metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
        self.model.eval()
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [i.to(self.device) for i in images]
                outputs = self.model(images)
                
                valid_preds = []
                valid_targets = []
                
                for out, t in zip(outputs, targets):
                    pred_dict = {
                        "boxes": out["boxes"].cpu(),
                        "scores": out["scores"].cpu(),
                        "labels": (out["labels"] - 1).cpu().clamp(min=0),
                    }
                    target_dict = {
                        "boxes": t["boxes"].cpu(),
                        "labels": (t["labels"] - 1).cpu().clamp(min=0),
                    }
                    valid_preds.append(pred_dict)
                    valid_targets.append(target_dict)
                
                if len(valid_preds) == len(valid_targets):
                    metric.update(valid_preds, valid_targets)
        
        return metric.compute()
    
    def _save(self, fname):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_map50_95": self.best_map,
            "best_epoch": self.best_epoch,
            "config": self.group_cfg,
        }, self.out_dir / fname)
    
    def get_model(self):
        return self.model


class PseudoExperimentEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["detection"]["num_classes"]
        self.img_size = tuple(cfg["detection"]["img_size"])
        self.class_names = list(cfg["detection"].get("class_names", {}).values())
    
    def evaluate_all(self, trained_models):
        data_path = Path(self.cfg["detection"]["data_path"])
        
        test_imgs = data_path / "test" / "images"
        test_lbls = data_path / "test" / "labels"
        if not test_imgs.exists():
            logger.warning("test/ не найден, используется val/")
            test_imgs = data_path / "val" / "images"
            test_lbls = data_path / "val" / "labels"
        
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted(f for f in test_imgs.glob("*") if f.suffix.lower() in exts)
        
        logger.info(f"Оценка на {len(image_files)} изображениях")
        
        results_dir = Path(self.cfg["paths"]["pseudo_output"]) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        dataset_stats = self._analyze_dataset(image_files, test_lbls)
        
        teacher_result = self._evaluate_teacher(image_files, test_lbls)
        if teacher_result:
            all_results.append(teacher_result)
        
        for name, trainer in trained_models.items():
            student_result = self._evaluate_student(name, trainer, image_files, test_lbls)
            if student_result:
                all_results.append(student_result)
        
        final_results = {
            "dataset_analysis": dataset_stats,
            "models": all_results,
            "evaluation_params": {
                "num_test_images": len(image_files),
                "img_size": self.img_size,
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "experiment_type": "pseudo_labeling",
            }
        }
        
        out_path = results_dir / "evaluation.json"
        with open(out_path, "w") as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Результаты сохранены: {out_path}")
        self._print_summary(all_results)
        
        return final_results
    
    def _evaluate_teacher(self, image_files, labels_dir):
        from utils.metrics import evaluate_model, predict_teacher
        
        teacher_path = self.cfg["teacher"]["detector_path"]
        if not teacher_path or not Path(teacher_path).exists():
            logger.warning("Учитель не найден для оценки")
            return None
        
        logger.info("Оценка учителя")
        
        try:
            teacher = lightly_train.load_model(teacher_path)
            teacher.eval()
            
            predict_fn = lambda p: predict_teacher(teacher, p, self.img_size)
            metrics = evaluate_model(predict_fn, image_files, labels_dir, self.num_classes, self.img_size, self.class_names)
            
            fps = measure_fps(teacher, image_files[0], self.img_size, self.device)
            stats = model_stats(teacher)
            
            result = {"model": "teacher_ltdetr", "type": "teacher", **metrics, **fps, **stats}
            logger.info(f"Учитель: mAP50:95={metrics['map50_95']:.4f}, mAP50={metrics['map50']:.4f}, FPS={fps['fps']:.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка оценки учителя: {e}", exc_info=True)
            return None
    
    def _evaluate_student(self, name, trainer, image_files, labels_dir):
        from utils.metrics import evaluate_model, predict_student
        
        logger.info(f"Оценка: {name}")
        
        try:
            model = trainer.get_model()
            model.eval()
            
            predict_fn = lambda p, m=model: predict_student(m, p, self.img_size, self.device)
            metrics = evaluate_model(predict_fn, image_files, labels_dir, self.num_classes, self.img_size, self.class_names)
            
            fps = measure_fps(model, image_files[0], self.img_size, self.device)
            stats = model_stats(model)
            
            result = {"model": name, "type": trainer.group_cfg["type"], **metrics, **fps, **stats}
            logger.info(f"{name}: mAP50:95={metrics['map50_95']:.4f}, mAP50={metrics['map50']:.4f}, FPS={fps['fps']:.1f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка оценки {name}: {e}", exc_info=True)
            return None
    
    def _analyze_dataset(self, image_files, labels_dir):
        stats = {
            'num_images': len(image_files),
            'total_objects': 0,
            'small': 0,
            'medium': 0,
            'large': 0,
            'objects_per_class': defaultdict(int),
        }
        
        all_areas = []
        
        for img_path in tqdm(image_files, desc="Анализ датасета"):
            gt = load_yolo_gt(img_path, labels_dir, self.num_classes, self.img_size)
            boxes = gt['boxes']
            labels = gt['labels']
            
            if len(boxes) == 0:
                continue
            
            stats['total_objects'] += len(boxes)
            
            for box, label in zip(boxes, labels):
                area = ((box[2] - box[0]) * (box[3] - box[1])).item()
                all_areas.append(area)
                stats['objects_per_class'][str(label.item())] += 1
                
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
        stats['avg_objects_per_image'] = total / max(len(image_files), 1)
        
        if all_areas:
            stats['avg_box_area'] = np.mean(all_areas)
            stats['median_box_area'] = np.median(all_areas)
            stats['min_box_area'] = np.min(all_areas)
            stats['max_box_area'] = np.max(all_areas)
        
        stats['objects_per_class'] = dict(stats['objects_per_class'])
        
        return stats
    
    def _print_summary(self, all_results):
        logger.info(f"\n{'Модель':<35} {'mAP50:95':>10} {'mAP50':>8} {'mAP75':>8} {'FPS':>7} {'Params':>8} {'Size':>7}")
        logger.info("-" * 90)
        
        for r in sorted(all_results, key=lambda x: x.get("map50_95", 0), reverse=True):
            logger.info(
                f"{r['model']:<35} {r.get('map50_95',0):>10.4f} "
                f"{r.get('map50',0):>8.4f} {r.get('map75',0):>8.4f} "
                f"{r.get('fps',0):>7.1f} {r.get('params_M',0):>7.1f}M "
                f"{r.get('size_mb',0):>6.1f}MB"
            )


def main():
    logger.info("Эксперимент с псевдоразметкой")
    
    cfg = load_config()
    
    Path(cfg["paths"]["pseudo_output"]).mkdir(parents=True, exist_ok=True)
    
    logger.info("Шаг 1: Генерация псевдоразметки")
    generator = PseudoLabelGenerator(cfg)
    pseudo_dir = generator.generate()
    
    logger.info("Шаг 2: Подготовка датасетов")
    data_path = Path(cfg["detection"]["data_path"])
    num_classes = cfg["detection"]["num_classes"]
    img_size = tuple(cfg["detection"]["img_size"])
    
    train_original = YOLODataset(
        data_path / "train" / "images",
        data_path / "train" / "labels",
        num_classes, img_size
    )
    
    val_dataset = YOLODataset(
        data_path / "val" / "images",
        data_path / "val" / "labels",
        num_classes, img_size
    )
    
    train_pseudo = YOLODataset(
        pseudo_dir / "images",
        pseudo_dir / "labels",
        num_classes, img_size
    )
    
    train_combined = ConcatDataset([train_original, train_pseudo])
    
    logger.info(f"Оригинальный train: {len(train_original)}, псевдо: {len(train_pseudo)}, объединенный: {len(train_combined)}, val: {len(val_dataset)}")
    
    batch_size = cfg["students"]["faster_rcnn_r18_scratch"]["batch"]
    
    train_loader = DataLoader(
        train_combined, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info("Шаг 3: Обучение моделей")
    
    pt_file = Path(cfg["paths"]["pretrain_output"]) / "pretrained_path.txt"
    backbone_ckpt = None
    
    if pt_file.exists():
        backbone_ckpt = pt_file.read_text().strip()
        if not Path(backbone_ckpt).exists():
            logger.warning(f"Файл бэкбона не найден: {backbone_ckpt}")
            backbone_ckpt = None
    
    trained_models = {}
    
    for name, group_cfg in cfg["students"].items():
        pseudo_name = f"{name}_pseudo"
        
        logger.info(f"Обучение: {pseudo_name}")
        
        try:
            trainer = PseudoExperimentTrainer(
                cfg, pseudo_name, group_cfg,
                train_loader, val_loader,
                backbone_ckpt
            )
            trainer.train()
            trained_models[pseudo_name] = trainer
            
        except Exception as e:
            logger.error(f"Ошибка обучения {pseudo_name}: {e}", exc_info=True)
    
    logger.info("Шаг 4: Оценка моделей")
    
    evaluator = PseudoExperimentEvaluator(cfg)
    evaluator.evaluate_all(trained_models)
    
    logger.info(f"Эксперимент завершён. Результаты: {cfg['paths']['pseudo_output']}")


if __name__ == "__main__":
    main()