#!/usr/bin/env python3
"""
Считаем текстурные признаки по трём методикам: GLCM, LBP и Габор.
Плюс градиенты — как простой способ оценить «резкость» текстуры.
Строим гистограммы, смотрим на разброс — если он маленький,
значит датасет однородный и модель может переобучиться на фон.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import logging

from utils import load_config, ensure_dir, print_section, load_images_batch
from utils.texture_utils import (
    compute_glcm_features, compute_lbp_features, compute_gabor_features,
)
from utils.image_utils import compute_gradient_magnitude
from utils.report_utils import save_figure

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cfg = load_config(Path(__file__).parent.parent / "config.yaml")


def texture_analysis():
    p = cfg['paths']
    tc = cfg['texture']
    rpt = ensure_dir(p['reports_dir'])

    print_section("АНАЛИЗ ТЕКСТУР")

    # --- Загрузка ---
    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))

    n_sample = min(500, len(image_files))
    logger.info(f"Загружаем {n_sample} изображений для анализа текстур...")
    images = load_images_batch(image_files, max_images=n_sample)

    # --- Признаки ---
    glcm_feat = []
    lbp_feat = []
    gabor_feat = []
    grad_means = []

    for i, img in enumerate(images):
        gray = np.mean(img, axis=2).astype(np.uint8)

        glcm_feat.append(
            compute_glcm_features(gray, tc['glcm_distances'], tc['glcm_angles'])
        )
        lbp_feat.append(
            compute_lbp_features(gray, tc['lbp_radius'], tc['lbp_points'])
        )
        gabor_feat.append(
            compute_gabor_features(gray, tc['gabor_frequencies'], tc['gabor_angles'])
        )
        grad_means.append(compute_gradient_magnitude(img).mean())

        if (i + 1) % 100 == 0:
            logger.info(f"  обработано {i + 1}/{n_sample}")

    # --- Визуализация ---
    fig, axes = plt.subplots(2, 2, figsize=cfg['report']['figsize'])

    # GLCM contrast — самый интерпретируемый признак из матрицы смежности
    contrast = [f['glcm_contrast_mean'] for f in glcm_feat]
    axes[0, 0].hist(contrast, bins=30, alpha=0.7, color='#FF6B6B')
    axes[0, 0].set_title('GLCM Contrast')
    axes[0, 0].axvline(np.mean(contrast), color='red', ls='--',
                       label=f'μ={np.mean(contrast):.1f}')
    axes[0, 0].legend(fontsize=8)

    # Энтропия LBP — мера хаотичности текстуры
    lbp_ent = [f['lbp_entropy'] for f in lbp_feat]
    axes[0, 1].hist(lbp_ent, bins=30, alpha=0.7, color='#4ECDC4')
    axes[0, 1].set_title('LBP Entropy')
    axes[0, 1].axvline(np.mean(lbp_ent), color='red', ls='--',
                       label=f'μ={np.mean(lbp_ent):.3f}')
    axes[0, 1].legend(fontsize=8)

    # Средний градиент — быстро и понятно
    axes[1, 0].hist(grad_means, bins=30, alpha=0.7, color='#45B7D1')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axvline(np.mean(grad_means), color='red', ls='--',
                       label=f'μ={np.mean(grad_means):.2f}')
    axes[1, 0].legend(fontsize=8)

    # Отклик одного фильтра Габора — направление 90°, частота 0.3
    # именно эта комбинация хорошо цепляет вертикальные дефекты на металле
    gabor_key = 'gabor_t90_f0.3_mean'
    gabor_vals = [f.get(gabor_key, 0) for f in gabor_feat]
    axes[1, 1].hist(gabor_vals, bins=30, alpha=0.7, color='#96CEB4')
    axes[1, 1].set_title('Gabor Response (θ=90°, f=0.3)')
    axes[1, 1].axvline(np.mean(gabor_vals), color='red', ls='--',
                       label=f'μ={np.mean(gabor_vals):.4f}')
    axes[1, 1].legend(fontsize=8)

    plt.suptitle('Разнообразие текстур в датасете', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "02_texture_diversity.png", rpt, cfg['report']['dpi'])
    plt.close(fig)  # чтобы не висело в памяти

    # --- CV как мера разнообразия ---
    print(f"\nКоэффициент вариации текстурных признаков (чем выше, тем разнообразнее):")
    print(f"  GLCM contrast:   CV = {np.std(contrast) / np.mean(contrast):.3f}")
    print(f"  LBP entropy:     CV = {np.std(lbp_ent) / np.mean(lbp_ent):.3f}")
    print(f"  Gradient:        CV = {np.std(grad_means) / np.mean(grad_means):.3f}")
    print(f"  Gabor response:  CV = {np.std(gabor_vals) / np.mean(gabor_vals):.3f}")

    logger.info(f"График сохранён: {rpt / '02_texture_diversity.png'}")


if __name__ == "__main__":
    texture_analysis()