import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
import logging

logger = logging.getLogger(__name__)


def measure_fps(model, img_path, img_size, device, warmup=30, iterations=100):
    if hasattr(model, 'predict'):
        def predict_fn(p):
            return model.predict(str(p))
    else:
        def predict_fn(p):
            img = Image.open(p).convert("RGB").resize(img_size, Image.BILINEAR)
            tensor = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                return model(tensor)[0]
    
    for _ in range(warmup):
        try:
            predict_fn(img_path)
        except Exception:
            pass
    
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        try:
            predict_fn(img_path)
        except Exception:
            pass
        times.append(time.perf_counter() - t0)
    
    if not times:
        return {"fps": 0.0, "latency_ms": 0.0}
    
    avg_lat = np.mean(times) * 1000
    return {"fps": round(1000/avg_lat, 1), "latency_ms": round(avg_lat, 2)}


def model_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    tmp = Path("/tmp/_eval_model.pth")
    torch.save(model.state_dict(), tmp)
    size_mb = tmp.stat().st_size / (1024**2)
    tmp.unlink(missing_ok=True)
    
    return {
        "params_M": round(total/1e6, 1),
        "trainable_M": round(trainable/1e6, 1),
        "size_mb": round(size_mb, 1)
    }


def predict_teacher(model, img_path, img_size, conf=0.25):
    if hasattr(model, 'predict'):
        try:
            results = model.predict(str(img_path))
            boxes = results["bboxes"]
            scores = results["scores"]
            labels = results["labels"]
            
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu()
                scores = scores.cpu()
                labels = labels.cpu()
            
            keep = scores > conf
            return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}
        except Exception as e:
            logger.error(f"predict() failed: {e}")
    
    return {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}


def predict_student(model, img_path, img_size, device, conf=0.25):
    try:
        img = Image.open(img_path).convert("RGB").resize(img_size, Image.BILINEAR)
        tensor = torch.from_numpy(
            np.array(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Ошибка загрузки {img_path}: {e}")
        return {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}

    with torch.no_grad():
        out = model(tensor)[0]

    keep = out["scores"] > conf
    return {
        "boxes": out["boxes"][keep].cpu(),
        "scores": out["scores"][keep].cpu(),
        "labels": (out["labels"][keep] - 1).cpu().clamp(min=0),
    }


def evaluate_model(predict_fn, image_files, labels_dir, num_classes, img_size, class_names=None):
    from utils.dataset import load_yolo_gt
    
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", class_metrics=True)
    
    n_processed = 0
    for img_path in image_files:
        pred = predict_fn(img_path)
        gt = load_yolo_gt(img_path, labels_dir, num_classes, img_size)
        
        if len(pred["boxes"]) > 0 or len(gt["boxes"]) > 0:
            metric.update([pred], [gt])
            n_processed += 1

    logger.info(f"Обработано изображений с объектами: {n_processed}/{len(image_files)}")
    
    result = metric.compute()

    out = {
        "map50_95": float(result["map"].item()),
        "map50": float(result["map_50"].item()),
        "map75": float(result["map_75"].item()),
    }

    if "map_per_class" in result and result["map_per_class"].numel() > 0:
        per_cls = result["map_per_class"].tolist()
        names = class_names or [f"cls{i}" for i in range(len(per_cls))]
        for i, (name, ap) in enumerate(zip(names, per_cls)):
            out[f"AP50_{name}"] = round(float(ap), 4)

    return out