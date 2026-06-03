#!/usr/bin/env python3
"""
Спектральный анализ через FFT: раскладываем изображения по частотным
кольцам и смотрим, где энергия — на низких частотах (фон) или высоких (детали).
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


def compute_fft_features(gray, n_bands=8):
    """Энергия по частотным кольцам + отношение низкие/высокие."""
    fft = np.fft.fft2(gray)
    magnitude = np.abs(np.fft.fftshift(fft))

    h, w = gray.shape
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx)
    band_w = max_r / n_bands

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

    features = {}
    for i in range(n_bands):
        r_in = i * band_w
        r_out = (i + 1) * band_w
        mask = (r >= r_in) & (r < r_out)
        features[f'band_{i}'] = float(magnitude[mask].sum())

    mid = n_bands // 2
    low = sum(features[f'band_{i}'] for i in range(mid))
    high = sum(features[f'band_{i}'] for i in range(mid, n_bands))
    features['low_high'] = low / high if high > 0 else 0

    return features


def main():
    p = cfg['paths']
    sc = cfg['spectral']
    rpt = ensure_dir(p['reports_dir'])

    print_section("СПЕКТРАЛЬНЫЙ АНАЛИЗ (FFT)")

    images_dir = Path(p['train_images_dir'])
    image_files = []
    for ext in cfg['image']['extensions']:
        image_files.extend(images_dir.glob(f"*{ext}"))

    n_sample = min(300, len(image_files))
    images = load_images_batch(image_files, max_images=n_sample)

    fft_feat = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fft_feat.append(compute_fft_features(gray, sc['fft_bands']))

    # --- Графики ---
    fig, axes = plt.subplots(2, 2, figsize=cfg['report']['figsize'])

    n_bands = sc['fft_bands']
    band_means = [np.mean([f[f'band_{i}'] for f in fft_feat]) for i in range(n_bands)]
    band_stds = [np.std([f[f'band_{i}'] for f in fft_feat]) for i in range(n_bands)]

    axes[0, 0].bar(range(n_bands), band_means, yerr=band_stds,
                   color='#FF6B6B', capsize=3)
    axes[0, 0].set_title('Энергия по частотным полосам')
    axes[0, 0].set_xlabel('0 — низкие, {} — высокие'.format(n_bands - 1))

    lh = [f['low_high'] for f in fft_feat]
    axes[0, 1].hist(lh, bins=30, alpha=0.7, color='#4ECDC4')
    axes[0, 1].set_title('Low / High ratio')
    axes[0, 1].axvline(np.mean(lh), color='red', ls='--',
                       label=f'μ={np.mean(lh):.2f}')
    axes[0, 1].legend()

    for i in range(min(4, len(images))):
        gray = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        log_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
        row, col = i // 2, i % 2
        axes[1, row].imshow(log_mag, cmap='inferno', aspect='auto')
        axes[1, row].set_title(f'Спектр {i + 1}')
        axes[1, row].axis('off')

    plt.suptitle('Спектральный анализ', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, "05_spectral_analysis.png", rpt, cfg['report']['dpi'])

    print(f"\nLow/High ratio: {np.mean(lh):.2f} ± {np.std(lh):.2f}")
    print(f"CV: {np.std(lh) / np.mean(lh):.3f}")

    logger.info(f"Сохранено: {rpt / '05_spectral_analysis.png'}")


if __name__ == "__main__":
    main()