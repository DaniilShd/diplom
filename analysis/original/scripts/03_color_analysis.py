#!/usr/bin/env python3
"""
Цветовой анализ датасета: переходим из RGB в LAB и HSV,
потому что RGB для анализа цвета практически бесполезен —
каналы сильно скоррелированы и не соответствуют человеческому восприятию.

LAB: Lightness + зелёный↔красный (a*) + синий↔жёлтый (b*)
HSV: Hue (тон) + Saturation (насыщенность) + Value (яркость)

Интересует, насколько датасет разнообразен по цвету.
Если все образцы серые и одинаковые — модель может не научиться
отличать дефекты от цветовых артефактов.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import logging

from utils import load_config, ensure_dir, print_section, load_images_batch
from utils.color_utils import compute_color_stats
from utils.report_utils import save_figure

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cfg = load_config(Path(__file__).parent.parent / "config.yaml")


def color_analysis():
    p = cfg['paths']
    rpt = ensure_dir(p['reports_dir'])

    print_section("ЦВЕТОВОЙ АНАЛИЗ")

    # --- Загрузка ---
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))

    n_sample = min(500, len(image_files))
    logger.info(f"Считаем цветовые признаки для {n_sample} изображений...")
    images = load_images_batch(image_files, max_images=n_sample)

    color_feat = [compute_color_stats(img) for img in images]

    # --- Распределения по каналам ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    L = [f['lab_ch0_mean'] for f in color_feat]
    axes[0, 0].hist(L, bins=40, alpha=0.7, color='#FF6B6B')
    axes[0, 0].set_title('LAB L* (светлота)')
    axes[0, 0].axvline(np.mean(L), color='red', ls='--')

    a = [f['lab_ch1_mean'] for f in color_feat]
    axes[0, 1].hist(a, bins=40, alpha=0.7, color='#4ECDC4')
    axes[0, 1].set_title('LAB a* (зелёный ↔ красный)')
    axes[0, 1].axvline(0, color='black', ls='-', alpha=0.5)

    b = [f['lab_ch2_mean'] for f in color_feat]
    axes[0, 2].hist(b, bins=40, alpha=0.7, color='#45B7D1')
    axes[0, 2].set_title('LAB b* (синий ↔ жёлтый)')
    axes[0, 2].axvline(0, color='black', ls='-', alpha=0.5)

    # Hue — циклическая штука, распределение может быть обманчивым
    H = [f['hsv_ch0_mean'] for f in color_feat]
    axes[1, 0].hist(H, bins=50, alpha=0.7, color='#96CEB4')
    axes[1, 0].set_title('HSV Hue (цветовой тон)')

    # Насыщенность: у чистого металла — низкая, у ржавчины/окислов — высокая
    S = [f['hsv_ch1_mean'] for f in color_feat]
    axes[1, 1].hist(S, bins=40, alpha=0.7, color='#FFA07A')
    axes[1, 1].set_title('HSV Saturation (насыщенность)')

    # Проекция a*b* — главный график для оценки цветового разброса
    # цвет точек кодирует яркость L
    axes[1, 2].scatter(a, b, c=L, cmap='viridis', alpha=0.5, s=10)
    axes[1, 2].set_xlabel('a* (зелёный → красный)')
    axes[1, 2].set_ylabel('b* (синий → жёлтый)')
    axes[1, 2].set_title('Цветовое распределение (a*b*)')
    axes[1, 2].axhline(0, color='gray', alpha=0.3)
    axes[1, 2].axvline(0, color='gray', alpha=0.3)

    plt.suptitle('Цветовой анализ: разнообразие оттенков металла',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "03_color_diversity.png", rpt, cfg['report']['dpi'])
    plt.close(fig)

    # --- Численная сводка ---
    # Для a* и b* делим на |mean| с защитой от нуля,
    # потому что среднее может быть близко к нулю (нейтральный цвет)
    print(f"\nЦветовая вариативность:")
    print(f"  Яркость L*:      CV = {np.std(L) / np.mean(L):.3f}")
    print(f"  a* (зел-красн):  CV = {np.std(a) / max(abs(np.mean(a)), 1e-6):.3f}")
    print(f"  b* (син-жёлт):   CV = {np.std(b) / max(abs(np.mean(b)), 1e-6):.3f}")
    print(f"  Тон Hue:         σ  = {np.std(H):.1f}° (разброс тонов)")
    print(f"  Насыщенность S:  CV = {np.std(S) / max(np.mean(S), 1e-6):.3f}")

    # Интерпретация для себя
    if np.mean(S) < 0.1:
        print(f"  → Насыщенность низкая — датасет почти чёрно-белый")
    else:
        print(f"  → Насыщенность заметная — есть цветовые артефакты")

    logger.info(f"График сохранён: {rpt / '03_color_diversity.png'}")


if __name__ == "__main__":
    color_analysis()