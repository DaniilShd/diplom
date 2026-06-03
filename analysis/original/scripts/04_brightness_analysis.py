#!/usr/bin/env python3
"""
Оценка качества освещения на снимках.

Проблемы, которые ищем:
- Пересветы — пиксели, выбитые в 255, теряют информацию о текстуре
- Недосветы — тёмные области, где дефекты просто не видны
- Неравномерность засветки — край изображения темнее/светлее центра

Для неравномерности бьём кадр на блоки и считаем разброс средних —
если по краям темнее, чем в центре, модель может привязаться
к положению дефекта в кадре, а не к самому дефекту.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging

from utils import load_config, ensure_dir, print_section, load_images_batch
from utils.report_utils import save_figure

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cfg = load_config()


def analyze_brightness(img: np.ndarray, bc: dict) -> dict:
    """
    Собирает метрики по яркости для одного изображения.
    bc — параметры из конфига: пороги пересвета/недосвета, размер блока.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Доля проблемных пикселей
    overexposed = (gray > bc['overexposure_threshold']).sum() / gray.size
    underexposed = (gray < bc['underexposure_threshold']).sum() / gray.size

    # Разбиваем на блоки — смотрим, насколько равномерно освещено поле
    h, w = gray.shape
    bs = bc['local_block_size']
    n_h, n_w = h // bs, w // bs

    block_means = []
    for i in range(n_h):
        y0, y1 = i * bs, (i + 1) * bs
        for j in range(n_w):
            x0, x1 = j * bs, (j + 1) * bs
            block_means.append(gray[y0:y1, x0:x1].mean())

    return {
        'mean_brightness': gray.mean(),
        'std_brightness': gray.std(),
        'overexposed_ratio': overexposed,
        'underexposed_ratio': underexposed,
        'contrast': gray.std() / gray.mean() if gray.mean() > 0 else 0,
        'local_std': np.std(block_means),
        'local_range': np.max(block_means) - np.min(block_means),
    }


def brightness_analysis():
    p = cfg['paths']
    bc = cfg['brightness']
    rpt = ensure_dir(p['reports_dir'])

    print_section("АНАЛИЗ ЯРКОСТИ И ЗАСВЕТОВ")

    # --- Загрузка ---
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))

    n_sample = min(500, len(image_files))
    logger.info(f"Анализируем яркость на {n_sample} изображениях...")
    images = load_images_batch(image_files, max_images=n_sample)

    brightness_data = [analyze_brightness(img, bc) for img in images]

    # --- Графики ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    M = [d['mean_brightness'] for d in brightness_data]
    axes[0, 0].hist(M, bins=40, alpha=0.7, color='#FFD700')
    axes[0, 0].set_title('Средняя яркость')
    axes[0, 0].set_xlabel('0–255')

    stds = [d['std_brightness'] for d in brightness_data]
    axes[0, 1].hist(stds, bins=40, alpha=0.7, color='#FF6B6B')
    axes[0, 1].set_title('Std яркости (глобальный контраст)')

    over_pct = [d['overexposed_ratio'] * 100 for d in brightness_data]
    axes[0, 2].hist(over_pct, bins=40, alpha=0.7, color='#FF4500')
    axes[0, 2].set_title(f'Пересветы (>{bc["overexposure_threshold"]})')
    axes[0, 2].set_xlabel('% пикселей')

    under_pct = [d['underexposed_ratio'] * 100 for d in brightness_data]
    axes[1, 0].hist(under_pct, bins=40, alpha=0.7, color='#4169E1')
    axes[1, 0].set_title(f'Недосветы (<{bc["underexposure_threshold"]})')
    axes[1, 0].set_xlabel('% пикселей')

    loc_std = [d['local_std'] for d in brightness_data]
    axes[1, 1].hist(loc_std, bins=40, alpha=0.7, color='#4ECDC4')
    axes[1, 1].set_title('Локальная вариация яркости')
    axes[1, 1].set_xlabel('std по блокам')

    loc_range = [d['local_range'] for d in brightness_data]
    axes[1, 2].hist(loc_range, bins=40, alpha=0.7, color='#96CEB4')
    axes[1, 2].set_title('Размах локальной яркости')
    axes[1, 2].set_xlabel('max − min по блокам')

    plt.suptitle('Анализ яркости: засветы, недосветы, неравномерность',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "04_brightness_analysis.png", rpt, cfg['report']['dpi'])
    plt.close(fig)

    # --- Вывод с оценкой ---
    print(f"\nЯркостные характеристики по {n_sample} изображениям:")
    print(f"  Средняя яркость:       {np.mean(M):.1f} ± {np.std(M):.1f}")
    print(f"  Пересветы (в среднем): {np.mean(over_pct):.2f}% "
          f"(макс {np.max(over_pct):.2f}%)")
    print(f"  Недосветы (в среднем): {np.mean(under_pct):.2f}% "
          f"(макс {np.max(under_pct):.2f}%)")
    print(f"  Локальная вариация:    {np.mean(loc_std):.1f} ± {np.std(loc_std):.1f}")

    # Быстрая диагностика
    if np.mean(over_pct) > 1.0:
        print(f"Есть пересветы — часть текстуры может быть потеряна")
    if np.mean(under_pct) > 5.0:
        print(f"Много тёмных областей — дефекты могут быть не видны")
    if np.mean(loc_std) > 15:
        print(f"Сильная неравномерность засветки — "
              f"модель может цепляться за положение, а не за дефект")
    if (np.mean(over_pct) <= 1.0 and np.mean(under_pct) <= 5.0
            and np.mean(loc_std) <= 15):
        print(f"  ✓ Освещение выглядит приемлемым")

    logger.info(f"График сохранён: {rpt / '04_brightness_analysis.png'}")


if __name__ == "__main__":
    brightness_analysis()