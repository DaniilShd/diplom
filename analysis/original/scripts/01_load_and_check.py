#!/usr/bin/env python3
"""
Первым делом проверяем, что данные на месте и нет расхождений
между разметкой (CSV) и файлами на диске.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import logging

from utils import ensure_dir, print_section, load_config

cfg = load_config(Path(__file__).parent.parent / "config.yaml")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_check():
    print_section("ЗАГРУЗКА И ПРОВЕРКА ДАННЫХ")

    p = cfg['paths']

    # --- CSV ---
    df = pd.read_csv(p['train_csv'])
    logger.info(
        f"CSV загружен: {len(df):,} строк, "
        f"{df['ImageId'].nunique():,} уникальных изображений"
    )

    dup = df['ImageId'].duplicated().sum()
    if dup:
        logger.warning(f"Дубликатов ImageId в CSV: {dup}")

    n_empty = df['EncodedPixels'].isna().sum()
    n_filled = len(df) - n_empty
    logger.info(f"Разметка: заполнено {n_filled:,}, пустых {n_empty:,}")

    # --- Файлы на диске ---
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))
    logger.info(f"Изображений в папке: {len(image_files):,}")

    # --- Сверка ---
    csv_ids = set(df['ImageId'].unique())
    disk_ids = {f.name for f in image_files}

    missing_disk = csv_ids - disk_ids
    missing_csv = disk_ids - csv_ids

    if missing_disk:
        logger.warning(
            f"Есть в CSV, но нет на диске: {len(missing_disk)} шт."
        )
        for fname in sorted(missing_disk)[:5]:
            logger.warning(f"  - {fname}")
        if len(missing_disk) > 5:
            logger.warning(f"  ... и ещё {len(missing_disk) - 5}")

    if missing_csv:
        logger.warning(
            f"Есть на диске, но нет в CSV: {len(missing_csv)} шт."
        )
        for fname in sorted(missing_csv)[:5]:
            logger.warning(f"  - {fname}")
        if len(missing_csv) > 5:
            logger.warning(f"  ... и ещё {len(missing_csv) - 5}")

    if not missing_disk and not missing_csv:
        logger.info("CSV и диск полностью совпадают")

    # --- Сохраняем сводку ---
    stats = {
        'total_rows': len(df),
        'unique_images': df['ImageId'].nunique(),
        'rle_filled': n_filled,
        'rle_empty': n_empty,
        'disk_images': len(image_files),
        'missing_disk': len(missing_disk),
        'missing_csv': len(missing_csv),
    }

    report_dir = ensure_dir(p['reports_dir'])
    out_path = report_dir / "01_data_integrity.csv"
    pd.DataFrame([stats]).to_csv(out_path, index=False)
    logger.info(f"Сводка сохранена: {out_path}")


if __name__ == "__main__":
    load_and_check()