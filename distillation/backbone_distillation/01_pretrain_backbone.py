#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

import torch
import torch.serialization
import yaml
import lightly_train


torch.serialization.add_safe_globals([torch.utils.data.DataLoader])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("01_pretrain_backbone.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def count_images(directory):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(
        len(list(directory.rglob(f"*{e}")))
        for e in exts | {e.upper() for e in exts}
    )


def main():
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"Конфиг не найден: {config_path}")
        sys.exit(1)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pretrain_cfg = cfg["pretrain"]
    teacher_cfg = cfg["teacher"]
    paths_cfg = cfg["paths"]

    out_dir = Path(paths_cfg["pretrain_output"]) / "resnet18_distilled"
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_path = out_dir / "exported_models" / "exported_last.pt"
    if exported_path.exists():
        logger.info(f"Модель уже существует: {exported_path}")
        cache_file = Path(paths_cfg["pretrain_output"]) / "pretrained_path.txt"
        cache_file.write_text(str(exported_path))
        return

    unlabeled_path = Path(pretrain_cfg["unlabeled_data"])
    
    if not unlabeled_path.exists() or count_images(unlabeled_path) == 0:
        fallback = Path(cfg["detection"]["data_path"]) / "train" / "images"
        if fallback.exists() and count_images(fallback) > 0:
            logger.warning(f"unlabeled_data не найдены, используется {fallback}")
            unlabeled_path = fallback
        else:
            logger.error("Нет изображений для дистилляции")
            sys.exit(1)

    n_images = count_images(unlabeled_path)
    logger.info(f"Изображений: {n_images}")
    if n_images < 1000:
        logger.warning("Мало изображений (<1000)")

    teacher_weights = Path(teacher_cfg["teacher_weights"])
    if not teacher_weights.exists():
        logger.error(f"Файл учителя не найден: {teacher_weights}")
        sys.exit(1)

    method_args = {
        "teacher": teacher_cfg["base_model"],
        "teacher_weights": str(teacher_weights),
    }

    logger.info(f"Учитель: {method_args['teacher']}")
    logger.info(f"Метод: {pretrain_cfg['method']}, эпох: {pretrain_cfg['epochs']}, batch: {pretrain_cfg['batch_size']}")

    try:
        lightly_train.pretrain(
            out=str(out_dir),
            data=str(unlabeled_path),
            model="torchvision/resnet18",
            method=pretrain_cfg["method"],
            method_args=method_args,
            epochs=pretrain_cfg["epochs"],
            batch_size=pretrain_cfg["batch_size"],
            transform_args={"image_size": tuple(pretrain_cfg["image_size"])},
            overwrite=True,
        )
    except KeyboardInterrupt:
        logger.warning("Дистилляция прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка дистилляции: {e}", exc_info=True)
        sys.exit(1)

    if not exported_path.exists():
        logger.error(f"Модель не найдена: {exported_path}")
        sys.exit(1)

    logger.info(f"Готово: {exported_path} ({exported_path.stat().st_size / (1024**2):.1f} MB)")

    cache_file = Path(paths_cfg["pretrain_output"]) / "pretrained_path.txt"
    cache_file.write_text(str(exported_path))


if __name__ == "__main__":
    main()