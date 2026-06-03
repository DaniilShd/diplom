#!/usr/bin/env python3
"""
Кластеризация изображений по глобальным признакам — смотрим,
разбивается ли датасет на визуально разные группы
(условия съёмки, текстура фона, засветка).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging

from utils import load_config, ensure_dir, print_section, load_images_batch
from utils.report_utils import save_figure

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cfg = load_config()


def extract_features(img):
    """Собирает всё в кучу: гистограмму, яркость, спектр, цвет."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / hist.sum()

    # Низкие частоты FFT — грубая структура фона
    fft = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = gray.shape
    low_energy = fft[h//2 - 10:h//2 + 10, w//2 - 10:w//2 + 10].mean()
    total_energy = fft.mean()

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab_mean = lab.mean(axis=(0, 1))
    lab_std = lab.std(axis=(0, 1))

    return np.concatenate([
        hist[:16],
        [gray.mean(), gray.std()],
        [low_energy / total_energy],
        lab_mean,
        lab_std,
    ])


def main():
    p = cfg['paths']
    cc = cfg['cluster']
    rpt = ensure_dir(p['reports_dir'])

    print_section("КЛАСТЕРИЗАЦИЯ ИЗОБРАЖЕНИЙ")

    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))

    n_sample = min(cc['sample_size'], len(image_files))
    images = load_images_batch(image_files, max_images=n_sample)
    logger.info(f"Извлечение признаков для {n_sample} изображений...")

    features = np.array([extract_features(img) for img in images])

    # Сжимаем до 2D для визуализации
    pca = PCA(n_components=2, random_state=cc['random_seed'])
    xy = pca.fit_transform(features)

    # Кластеризуем
    n_clusters = cc['n_clusters']
    logger.info(f"KMeans на {n_clusters} кластеров...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=cc['random_seed'], n_init=10)
    labels = kmeans.fit_predict(features)

    # --- Графики ---
    fig, axes = plt.subplots(1, 2, figsize=cfg['report']['figsize'])
    cmap = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    for i in range(n_clusters):
        mask = labels == i
        axes[0].scatter(xy[mask, 0], xy[mask, 1],
                        c=[cmap[i]], label=f'Кластер {i + 1}',
                        alpha=0.6, s=15)
    axes[0].set_title(f'PCA ({n_clusters} кластеров)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].legend(fontsize=7, ncol=2)

    sizes = [(labels == i).sum() for i in range(n_clusters)]
    axes[1].bar(range(1, n_clusters + 1), sizes, color=cmap)
    axes[1].set_title('Размеры кластеров')
    axes[1].set_xlabel('Кластер')
    axes[1].set_ylabel('Изображений')

    plt.suptitle('Кластеризация: группы неоднородности', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "06_cluster_analysis.png", rpt, cfg['report']['dpi'])

    print(f"\nКластеров: {n_clusters}")
    print(f"  Размеры: от {min(sizes)} до {max(sizes)}")
    print(f"  CV: {np.std(sizes) / np.mean(sizes):.3f}")

    logger.info(f"Сохранено: {rpt / '06_cluster_analysis.png'}")


if __name__ == "__main__":
    main()