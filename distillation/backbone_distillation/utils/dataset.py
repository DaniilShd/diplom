import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, num_classes, img_size=(640, 640)):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = num_classes
        self.img_size = img_size

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.files = sorted(
            f for f in self.images_dir.glob("*") 
            if f.suffix.lower() in exts
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize(self.img_size, Image.BILINEAR)
            tensor = torch.from_numpy(
                np.array(img, dtype=np.float32) / 255.0
            ).permute(2, 0, 1)
        except Exception:
            return torch.zeros(3, *self.img_size), {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            }

        boxes, labels = self._load_yolo(img_path, orig_w, orig_h)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
        }
        return tensor, target

    def _load_yolo(self, img_path, ow, oh):
        boxes, labels = [], []
        lbl_path = self.labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            return boxes, labels

        sx, sy = self.img_size[1] / ow, self.img_size[0] / oh
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    if cls >= self.num_classes:
                        continue
                    xc, yc, w, h = map(float, parts[1:5])
                    x1 = max(0.0, (xc - w/2) * ow * sx)
                    y1 = max(0.0, (yc - h/2) * oh * sy)
                    x2 = min(float(self.img_size[1]), (xc + w/2) * ow * sx)
                    y2 = min(float(self.img_size[0]), (yc + h/2) * oh * sy)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)
                except (ValueError, IndexError):
                    continue
        return boxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


def load_yolo_gt(img_path, labels_dir, num_classes, img_size=(640, 640)):
    lbl_path = Path(labels_dir) / f"{Path(img_path).stem}.txt"
    boxes, labels = [], []
    
    if not lbl_path.exists():
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(0, dtype=torch.int64),
        }

    try:
        w0, h0 = Image.open(img_path).size
    except Exception:
        w0, h0 = img_size[1], img_size[0]
    
    sx, sy = img_size[1] / w0, img_size[0] / h0

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                if cls >= num_classes:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                x1 = max(0.0, (xc - w/2) * w0 * sx)
                y1 = max(0.0, (yc - h/2) * h0 * sy)
                x2 = min(float(img_size[1]), (xc + w/2) * w0 * sx)
                y2 = min(float(img_size[0]), (yc + h/2) * h0 * sy)
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
            except (ValueError, IndexError):
                continue

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
    }