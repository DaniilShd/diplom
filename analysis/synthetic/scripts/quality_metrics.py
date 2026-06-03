"""
Попиксельные метрики: PSNR, SSIM, гистограммы, границы, спектр.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from config import AnalysisConfig


def compute_psnr(img1, img2):
    return psnr(img1, img2, data_range=255)


def compute_ssim(img1, img2, win_size=7):
    w = min(win_size, min(img1.shape[0], img1.shape[1]) // 2 * 2 + 1)
    return ssim(img1, img2, data_range=255, channel_axis=2, win_size=w)


def histogram_similarity(img1, img2):
    """Корреляция, Бхаттачарья, хи-квадрат по каждому каналу."""
    result = {}
    for ch, name in enumerate(['B', 'G', 'R']):
        h1 = cv2.normalize(cv2.calcHist([img1], [ch], None, [256], [0, 256]), None).flatten()
        h2 = cv2.normalize(cv2.calcHist([img2], [ch], None, [256], [0, 256]), None).flatten()
        result[name] = {
            "corr": float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)),
            "bhatta": float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)),
            "chi2": float(cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)),
        }
    return result


def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges_canny = cv2.Canny(gray, 50, 150)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return {
        "canny_density": float(np.sum(edges_canny > 0) / gray.size),
        "sobel_mean": float(np.mean(sobel)),
        "sobel_std": float(np.std(sobel)),
    }


def texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray8 = (gray / 256).astype(np.uint8)
    try:
        glcm = graycomatrix(gray8, [1, 3, 5], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        return {
            "contrast": float(np.mean(graycoprops(glcm, 'contrast'))),
            "homogeneity": float(np.mean(graycoprops(glcm, 'homogeneity'))),
            "energy": float(np.mean(graycoprops(glcm, 'energy'))),
            "correlation": float(np.mean(graycoprops(glcm, 'correlation'))),
        }
    except Exception:
        return {"contrast": 0, "homogeneity": 0, "energy": 0, "correlation": 0}


def frequency_spectrum(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_low = min(h, w) // 8
    r_high = min(h, w) // 3
    total = np.sum(mag)
    return {
        "low_freq": float(np.sum(mag[dist <= r_low]) / total),
        "mid_freq": float(np.sum(mag[(dist > r_low) & (dist <= r_high)]) / total),
        "high_freq": float(np.sum(mag[dist > r_high]) / total),
        "centroid": float(np.sum(dist * mag) / total),
    }


def find_pairs(original_dir, synthetic_dir, max_pairs):
    orig = {}
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for p in (original_dir / "images").glob(ext):
            orig[p.stem] = p

    synth = {}
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for p in (synthetic_dir / "images").glob(ext):
            synth[p.stem] = p

    pairs = []
    for o_stem, o_path in orig.items():
        for s_stem, s_path in synth.items():
            if o_stem in s_stem:
                pairs.append((o_path, s_path, o_stem))
                break

    # Добиваем случайными, если не хватает
    if len(pairs) < max_pairs and synth:
        random.seed(42)
        o_list = list(orig.values())
        s_list = list(synth.values())
        while len(pairs) < max_pairs and o_list and s_list:
            pairs.append((random.choice(o_list), random.choice(s_list), "random"))

    return pairs[:max_pairs]


def run_quality_analysis(config: AnalysisConfig):
    cfg = config.quality
    out = config.paths.output_dir

    print("Quality metrics: searching pairs...")
    pairs = find_pairs(config.paths.original_dir, config.paths.synthetic_dir,
                       min(cfg.num_samples_fid, 100))
    print(f"Pairs found: {len(pairs)}")

    psnr_vals = []
    ssim_vals = []
    edge_orig, edge_synth = defaultdict(list), defaultdict(list)
    tex_orig, tex_synth = defaultdict(list), defaultdict(list)
    hist_sim = defaultdict(list)
    freq_orig, freq_synth = defaultdict(list), defaultdict(list)
    per_image = []

    for orig_path, synth_path, name in tqdm(pairs, desc="Metrics"):
        orig_img = cv2.imread(str(orig_path))
        synth_img = cv2.imread(str(synth_path))
        if orig_img is None or synth_img is None:
            continue

        orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        synth_rgb = cv2.cvtColor(synth_img, cv2.COLOR_BGR2RGB)
        if orig_rgb.shape != synth_rgb.shape:
            synth_rgb = cv2.resize(synth_rgb, (orig_rgb.shape[1], orig_rgb.shape[0]))

        if cfg.compute_psnr:
            psnr_vals.append(compute_psnr(orig_rgb, synth_rgb))
        if cfg.compute_ssim:
            ssim_vals.append(compute_ssim(orig_rgb, synth_rgb, cfg.ssim_window_size))

        if config.additional_analyses.compute_color_histograms:
            for ch_name, metrics in histogram_similarity(orig_rgb, synth_rgb).items():
                for m_name, v in metrics.items():
                    hist_sim[f"{ch_name}_{m_name}"].append(v)

        if config.additional_analyses.compute_edge_density:
            e1 = edge_density(orig_rgb)
            e2 = edge_density(synth_rgb)
            for k in e1:
                edge_orig[k].append(e1[k])
                edge_synth[k].append(e2[k])

        if config.additional_analyses.compute_texture_analysis:
            t1 = texture_features(orig_rgb)
            t2 = texture_features(synth_rgb)
            for k in t1:
                tex_orig[k].append(t1[k])
                tex_synth[k].append(t2[k])

        if config.additional_analyses.compute_frequency_analysis:
            f1 = frequency_spectrum(orig_rgb)
            f2 = frequency_spectrum(synth_rgb)
            for k in f1:
                freq_orig[k].append(f1[k])
                freq_synth[k].append(f2[k])

        per_image.append({
            "name": name,
            "psnr": float(psnr_vals[-1]) if psnr_vals else None,
            "ssim": float(ssim_vals[-1]) if ssim_vals else None,
        })

    # Сборка результатов
    results = {"per_image": per_image}

    if psnr_vals:
        results["psnr"] = {
            "mean": float(np.mean(psnr_vals)),
            "std": float(np.std(psnr_vals)),
            "min": float(np.min(psnr_vals)),
            "max": float(np.max(psnr_vals)),
            "median": float(np.median(psnr_vals)),
        }
    if ssim_vals:
        results["ssim"] = {
            "mean": float(np.mean(ssim_vals)),
            "std": float(np.std(ssim_vals)),
            "min": float(np.min(ssim_vals)),
            "max": float(np.max(ssim_vals)),
            "median": float(np.median(ssim_vals)),
        }

    # JSON
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / "quality_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved: {json_path}")

    # Графики
    viz_dir = out / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    if psnr_vals:
        axes[0, 0].hist(psnr_vals, bins=30, color='#2196F3', alpha=0.8, edgecolor='black')
        axes[0, 0].axvline(np.mean(psnr_vals), color='r', ls='--',
                           label=f'mean: {np.mean(psnr_vals):.2f} dB')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_title('PSNR')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    if ssim_vals:
        axes[0, 1].hist(ssim_vals, bins=30, color='#4CAF50', alpha=0.8, edgecolor='black')
        axes[0, 1].axvline(np.mean(ssim_vals), color='r', ls='--',
                           label=f'mean: {np.mean(ssim_vals):.4f}')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_title('SSIM')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    if psnr_vals and ssim_vals and len(psnr_vals) == len(ssim_vals):
        axes[1, 0].scatter(psnr_vals, ssim_vals, alpha=0.5, c='#673AB7', s=20)
        axes[1, 0].set_xlabel('PSNR (dB)')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('PSNR vs SSIM')
        axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis('off')
    txt = "Quality metrics summary\n\n"
    if psnr_vals:
        txt += f"PSNR: mean={np.mean(psnr_vals):.2f} std={np.std(psnr_vals):.2f} "
        txt += f"min={np.min(psnr_vals):.2f} max={np.max(psnr_vals):.2f}\n\n"
    if ssim_vals:
        txt += f"SSIM: mean={np.mean(ssim_vals):.4f} std={np.std(ssim_vals):.4f} "
        txt += f"min={np.min(ssim_vals):.4f} max={np.max(ssim_vals):.4f}\n"
    axes[1, 1].text(0.1, 0.5, txt, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Quality: original vs synthetic', fontsize=14, fontweight='bold')
    plt.tight_layout()
    for fmt in config.visualization.save_formats:
        plt.savefig(viz_dir / f'quality_metrics.{fmt}',
                    dpi=config.visualization.dpi, bbox_inches='tight')
    plt.close()
    print(f"Plots saved: {viz_dir}")

    return results