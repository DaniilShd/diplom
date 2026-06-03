"""
Сравниваем оригинальные и синтетические изображения через эмбеддинги DINOv2.
Если распределения сильно разные — синтетика не поможет, а навредит.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import AnalysisConfig


class ImageDataset(Dataset):
    """Грузит картинки и прогоняет через препроцессор DINOv2."""

    def __init__(self, image_paths, processor, image_size=256):
        self.paths = image_paths
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        except Exception:
            img = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        inputs = self.processor(images=img, return_tensors="pt")
        return inputs.pixel_values.squeeze(0), path.name


def extract_features(image_dir, processor, model, device, config):
    """Извлекает DINOv2-эмбеддинги из папки с изображениями."""
    images_dir = image_dir / "images"
    paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        paths.extend(sorted(images_dir.glob(ext)))

    n_sample = config.dinov2.num_samples
    if len(paths) > n_sample:
        rng = np.random.RandomState(config.dinov2.random_seed)
        paths = rng.choice(paths, n_sample, replace=False).tolist()

    dataset = ImageDataset(paths, processor, config.dinov2.image_size)
    loader = DataLoader(dataset, batch_size=config.dinov2.batch_size,
                        shuffle=False, num_workers=4)

    features, names = [], []
    with torch.no_grad():
        for batch, batch_names in tqdm(loader, desc="  Извлечение признаков"):
            output = model(batch.to(device))
            features.append(output.pooler_output.cpu().numpy())
            names.extend(batch_names)

    return np.concatenate(features, axis=0), names


def compute_emd(feat1, feat2):
    """Earth Mover's Distance поканально."""
    emd_vals = []
    for i in range(feat1.shape[1]):
        emd_vals.append(wasserstein_distance(feat1[:, i], feat2[:, i]))
    emd_vals = np.array(emd_vals)
    return {
        "mean": float(np.mean(emd_vals)),
        "std": float(np.std(emd_vals)),
        "max": float(np.max(emd_vals)),
        "min": float(np.min(emd_vals)),
        "median": float(np.median(emd_vals)),
        "p95": float(np.percentile(emd_vals, 95)),
    }


def compute_similarity(feat1, feat2, config):
    """Метрики схожести: косинус центроидов, 1-NN, gap ratio."""
    n_test = min(config.dinov2.nn_test_samples, len(feat1), len(feat2))
    rng = np.random.RandomState(config.dinov2.random_seed)

    c1 = np.mean(feat1, axis=0)
    c2 = np.mean(feat2, axis=0)
    centroid_dist = np.linalg.norm(c1 - c2)
    cosine = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)

    # 1-NN тест: если домены неразличимы, точность ~0.5
    idx1 = rng.choice(len(feat1), n_test, replace=False)
    idx2 = rng.choice(len(feat2), n_test, replace=False)
    combined = np.concatenate([feat1[idx1], feat2[idx2]])
    labels = np.array([0] * n_test + [1] * n_test)

    correct = 0
    for i in range(2 * n_test):
        dist = cdist(combined[i:i+1], combined, metric='cosine')[0]
        dist[i] = np.inf
        if labels[i] == labels[np.argmin(dist)]:
            correct += 1
    nn_acc = correct / (2 * n_test)

    # Внутри- и меж-доменные расстояния
    d_11 = cdist(feat1[idx1], feat1[idx1], metric='cosine')
    d_22 = cdist(feat2[idx2], feat2[idx2], metric='cosine')
    d_12 = cdist(feat1[idx1], feat2[idx2], metric='cosine')
    np.fill_diagonal(d_11, np.inf)
    np.fill_diagonal(d_22, np.inf)

    intra = (np.mean(np.min(d_11, axis=1)) + np.mean(np.min(d_22, axis=1))) / 2
    inter = (np.mean(np.min(d_12, axis=1)) + np.mean(np.min(d_12, axis=0))) / 2
    gap_ratio = inter / (intra + 1e-8)
    overlap = np.clip(2.0 * (1.0 - nn_acc), 0.0, 1.0)

    return {
        "centroid_distance": float(centroid_dist),
        "cosine_similarity": float(cosine),
        "intra_distance": float(intra),
        "inter_distance": float(inter),
        "gap_ratio": float(gap_ratio),
        "nn_accuracy": float(nn_acc),
        "overlap_score": float(overlap),
    }


def plot_embeddings(feat_list, names_list, domain_labels, output_dir, config):
    """PCA и t-SNE визуализации с эллипсами."""
    viz_dir = output_dir / config.paths.subdirs.get('visualizations', 'visualizations')
    viz_dir.mkdir(parents=True, exist_ok=True)
    vc = config.visualization
    colors = {'original': '#2196F3', 'synthetic': '#FF9800'}
    domain_ru = {'original': 'Оригинал', 'synthetic': 'Синтетика'}

    # Объединяем и стандартизируем
    all_feat = np.concatenate(feat_list, axis=0)
    all_feat = StandardScaler().fit_transform(all_feat)
    all_names = []
    for name, feat in zip(names_list, feat_list):
        all_names.extend([name] * len(feat))

    # --- PCA ---
    pca = PCA(n_components=2)
    xy = pca.fit_transform(all_feat)
    var = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=vc.figsize)

    for name in set(all_names):
        mask = np.array([n == name for n in all_names])
        data = xy[mask]
        color = colors.get(name, '#999')
        label = domain_ru.get(name, name)
        axes[0].scatter(data[:, 0], data[:, 1], c=color, label=label,
                        alpha=0.5, s=10, edgecolors='none')

        if len(data) > 2 and vc.pca.get('show_ellipses', True):
            try:
                cov = np.cov(data.T)
                mean = np.mean(data, axis=0)
                eigvals, eigvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                w, h = 2 * np.sqrt(eigvals) * 2.447
                axes[0].add_patch(Ellipse(xy=mean, width=w, height=h, angle=angle,
                                          facecolor='none', edgecolor=color,
                                          linewidth=2, linestyle='--', alpha=0.8))
            except Exception:
                pass

    axes[0].set_xlabel(f'PC1 ({var[0]:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({var[1]:.1f}%)')
    axes[0].set_title('DINOv2 — PCA')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Гистограмма попарных расстояний
    names_unique = list(set(all_names))
    if len(names_unique) >= 2:
        d1 = xy[np.array([n == names_unique[0] for n in all_names])][:100]
        d2 = xy[np.array([n == names_unique[1] for n in all_names])][:100]
        n = min(len(d1), len(d2))
        dists = np.linalg.norm(d1[:n] - d2[:n], axis=1)
        axes[1].hist(dists, bins=30, alpha=0.7, color='#673AB7', edgecolor='black')
        axes[1].axvline(np.mean(dists), color='r', ls='--', label=f'среднее: {np.mean(dists):.3f}')
        axes[1].set_xlabel('Попарное расстояние в PCA')
        axes[1].set_title(f'{domain_ru.get(names_unique[0])} vs {domain_ru.get(names_unique[1])}')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in vc.save_formats:
        plt.savefig(viz_dir / f'pca_domain_gap.{fmt}', dpi=vc.dpi, bbox_inches='tight')
    plt.close()

    # --- t-SNE ---
    if vc.tsne.get('enabled', False):
        n_tsne = min(len(all_feat), vc.tsne.get('max_points', 300))
        idx = np.random.choice(len(all_feat), n_tsne, replace=False)
        tsne = TSNE(n_components=2, perplexity=vc.tsne.get('perplexity', 30),
                    max_iter=vc.tsne.get('max_iter', 1000),
                    random_state=config.dinov2.random_seed)
        xy_tsne = tsne.fit_transform(all_feat[idx])
        names_tsne = [all_names[i] for i in idx]

        fig, ax = plt.subplots(figsize=(10, 8))
        for name in set(names_tsne):
            mask = np.array([n == name for n in names_tsne])
            ax.scatter(xy_tsne[mask, 0], xy_tsne[mask, 1],
                       c=colors.get(name, '#999'), label=domain_ru.get(name, name),
                       alpha=0.6, s=15, edgecolors='none')
        ax.set_title('DINOv2 — t-SNE')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        for fmt in vc.save_formats:
            plt.savefig(viz_dir / f'tsne_domain_gap.{fmt}', dpi=vc.dpi, bbox_inches='tight')
        plt.close()

    print(f"Графики сохранены: {viz_dir}")


def generate_report(results, config, output_dir):
    """Текстовый отчёт с интерпретацией."""
    t = config.thresholds
    sim = results['domain_gap']
    emd = results['emd_analysis']
    overlap = sim['overlap_score']
    nn_acc = sim['nn_accuracy']

    if overlap > t.domain_overlap['excellent'] and nn_acc < 0.55:
        quality, recommendation = "Отлично", "использовать без ограничений"
    elif overlap > t.domain_overlap['good'] and nn_acc < 0.65:
        quality, recommendation = "Хорошо", "использовать с весом 0.3–0.5"
    elif overlap > t.domain_overlap['satisfactory'] and nn_acc < 0.75:
        quality, recommendation = "Удовлетворительно", "уменьшить аугментацию до 0.05–0.10"
    elif overlap > t.domain_overlap['poor']:
        quality, recommendation = "Плохо", "уменьшить аугментацию до 0.02–0.05"
    else:
        quality, recommendation = "Критически", "пересмотреть параметры генерации"

    report = f"""
================================================================================
                    АНАЛИЗ РАЗРЫВА ДОМЕНОВ (DOMAIN GAP)
================================================================================
Модель: {config.dinov2.model_name}
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}

1. СХОЖЕСТЬ ДОМЕНОВ
   Косинус центроидов:    {sim['cosine_similarity']:.4f}
   Gap ratio:             {sim['gap_ratio']:.4f}  (1.0 = идентичны)
   Overlap score:         {overlap:.4f}  (1.0 = полное перекрытие)
   1-NN accuracy:         {nn_acc:.4f}  (0.5 = неразличимы)

2. EMD ПОКАНАЛЬНО
   Среднее:  {emd['mean']:.6f}
   Медиана:  {emd['median']:.6f}
   P95:      {emd['p95']:.6f}

3. ОЦЕНКА КАЧЕСТВА
   Качество синтетики: {quality}
   Рекомендация: {recommendation}

================================================================================
"""
    print(report)
    with open(output_dir / "domain_gap_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)


def run_domain_gap_analysis(config: AnalysisConfig):
    """Точка входа: загрузка модели, извлечение признаков, анализ."""
    device = config.dinov2.device
    model_name = config.dinov2.model_name

    print("=" * 60)
    print("Domain gap анализ")
    print("=" * 60)

    print(f"\nЗагрузка {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Извлечение признаков
    orig_feat, _ = extract_features(config.paths.original_dir, processor, model, device, config)
    synth_feat, _ = extract_features(config.paths.synthetic_dir, processor, model, device, config)

    # Метрики
    print("\nСчитаем метрики схожести...")
    sim = compute_similarity(orig_feat, synth_feat, config)

    print("Считаем EMD...")
    emd = compute_emd(orig_feat, synth_feat)

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "domain_gap": sim,
        "emd_analysis": emd,
    }

    # Визуализация
    output_dir = config.paths.output_dir
    plot_embeddings([orig_feat, synth_feat], ['original', 'synthetic'],
                    ['Оригинал', 'Синтетика'], output_dir, config)

    # JSON
    json_path = output_dir / "domain_gap_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON: {json_path}")

    # Текстовый отчёт
    generate_report(results, config, output_dir)