#!/usr/bin/env python3
# 02_train_detectors.py
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from utils.dataset import YOLODataset, collate_fn
from utils.backbone_loader import load_lightly_backbone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("02_train_detectors.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg, name, group_cfg, backbone_ckpt=None):
        self.cfg = cfg
        self.name = name
        self.group_cfg = group_cfg
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

        self.train_dl, self.val_dl = self._dataloaders()
        self.out_dir = Path(cfg["paths"]["detection_output"]) / name
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
                n_loaded = load_lightly_backbone(model, self.backbone_ckpt)
                if n_loaded == 0:
                    logger.error("Дистиллированные веса не загружены, используется случайная инициализация")
            else:
                logger.warning(f"Чекпоинт не найден: {self.backbone_ckpt}")

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Параметры: {total/1e6:.1f}M всего, {trainable/1e6:.1f}M trainable")
        return model

    def _dataloaders(self):
        dp = Path(self.cfg["detection"]["data_path"])
        sz = tuple(self.cfg["detection"]["img_size"])
        nc = self.num_classes
        bs = self.group_cfg["batch"]

        train_ds = YOLODataset(dp/"train"/"images", dp/"train"/"labels", nc, sz)
        val_ds = YOLODataset(dp/"val"/"images", dp/"val"/"labels", nc, sz)

        train_dl = DataLoader(
            train_ds, bs, shuffle=True, num_workers=4,
            collate_fn=collate_fn, pin_memory=torch.cuda.is_available()
        )
        val_dl = DataLoader(
            val_ds, bs, shuffle=False, num_workers=2,
            collate_fn=collate_fn, pin_memory=torch.cuda.is_available()
        )
        return train_dl, val_dl

    def train(self):
        logger.info(f"Обучение: {self.name}")
        
        history = []
        best_state = None
        start_time = time.time()

        for epoch in range(1, self.group_cfg["epochs"] + 1):
            self.model.train()
            total_loss = 0.0
            
            for images, targets in self.train_dl:
                images = [i.to(self.device) for i in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dl)
            
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
                best_state = copy.deepcopy(self.model.state_dict())
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
            for images, targets in self.val_dl:
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


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    
    if not cfg_path.exists():
        logger.error(f"Конфиг не найден: {cfg_path}")
        sys.exit(1)
    
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pt_file = Path(cfg["paths"]["pretrain_output"]) / "pretrained_path.txt"
    backbone_ckpt = None
    
    if pt_file.exists():
        backbone_ckpt = pt_file.read_text().strip()
        if not Path(backbone_ckpt).exists():
            logger.warning(f"Файл не найден: {backbone_ckpt}")
            backbone_ckpt = None
        else:
            logger.info(f"Backbone: {backbone_ckpt}")
    else:
        logger.warning("pretrained_path.txt не найден, distilled группа = scratch")

    summary_path = Path(cfg["paths"]["detection_output"]) / "results.json"
    all_results = []
    trained = set()
    
    if summary_path.exists():
        try:
            all_results = json.loads(summary_path.read_text())
            trained = {r["model_name"] for r in all_results}
        except Exception:
            pass

    for name, group_cfg in cfg["students"].items():
        final_path = Path(cfg["paths"]["detection_output"]) / name / "model_final.pth"
        
        if name in trained or final_path.exists():
            logger.info(f"{name} уже обучена, пропускаем")
            continue

        logger.info(f"Обучение: {name}")
        
        try:
            trainer = Trainer(cfg, name, group_cfg, backbone_ckpt)
            result = trainer.train()
            all_results.append(result)
            summary_path.write_text(json.dumps(all_results, indent=2))
        except Exception as e:
            logger.error(f"Ошибка обучения {name}: {e}", exc_info=True)

    if all_results:
        logger.info(f"\n{'Модель':<35} {'mAP50:95':>10} {'mAP50':>8} {'mAP75':>8} {'Время':>8}")
        logger.info("-" * 70)
        
        for r in sorted(all_results, key=lambda x: x["best_map50_95"], reverse=True):
            logger.info(
                f"{r['model_name']:<35} "
                f"{r['best_map50_95']:>10.4f} "
                f"{r['best_map50']:>8.4f} "
                f"{r['best_map75']:>8.4f} "
                f"{r.get('training_time_hours', 0):>7.1f}ч"
            )


if __name__ == "__main__":
    main()