#!/usr/bin/env python3
"""
Запускает анализы синтетики: domain gap и распределение классов.
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from config import AnalysisConfig
from domain_gap import run_domain_gap_analysis
from class_analysis import run_class_analysis


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(description="Анализ синтетических данных")
    parser.add_argument("--config", type=str, default="analysis/synthetic/config.yaml")
    parser.add_argument("--original_dir", type=str, default=None)
    parser.add_argument("--synthetic_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_domain_gap", action="store_true")
    parser.add_argument("--skip_class_analysis", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    config = AnalysisConfig.from_yaml(config_path)

    if args.original_dir:
        config.paths.original_dir = Path(args.original_dir)
    if args.synthetic_dir:
        config.paths.synthetic_dir = Path(args.synthetic_dir)
    if args.num_samples:
        config.dinov2.num_samples = args.num_samples

    output_dir = Path(args.output_dir) if args.output_dir else config.setup_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Original: {config.paths.original_dir}")
    print(f"Synthetic: {config.paths.synthetic_dir}")
    print(f"Output: {output_dir}")

    for name, d in [("original", config.paths.original_dir),
                    ("synthetic", config.paths.synthetic_dir)]:
        if not (d / "images").exists():
            print(f"Images not found: {d / 'images'}")
            sys.exit(1)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "original_dir": str(config.paths.original_dir),
        "synthetic_dir": str(config.paths.synthetic_dir),
        "analyses": {},
    }

    if not args.skip_domain_gap:
        print("\n--- Domain gap ---")
        try:
            result = run_domain_gap_analysis(config)
            dg = result.get('domain_gap', {})
            summary['analyses']['domain_gap'] = {
                "status": "ok",
                "overlap": dg.get('overlap_score'),
                "nn_accuracy": dg.get('nn_accuracy'),
                "cosine": dg.get('cosine_similarity'),
                "gap_ratio": dg.get('gap_ratio'),
                "mean_emd": result.get('emd_analysis', {}).get('mean'),
            }
        except Exception as e:
            print(f"Domain gap failed: {e}")
            summary['analyses']['domain_gap'] = {"status": "error", "error": str(e)}

    if not args.skip_class_analysis:
        print("\n--- Class distribution ---")
        try:
            result = run_class_analysis(config)
            cs = result.get('summary', {})
            summary['analyses']['class_distribution'] = {
                "status": "ok",
                "total_bboxes": cs.get('total_bboxes'),
                "images_with_defects": cs.get('images_with_defects'),
                "empty_images": cs.get('empty_images'),
                "classes": result.get('class_stats', {}),
            }
        except Exception as e:
            print(f"Class analysis failed: {e}")
            summary['analyses']['class_distribution'] = {"status": "error", "error": str(e)}

    json_path = output_dir / "analysis_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nSaved: {json_path}")

    txt_path = output_dir / "analysis_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Analysis summary\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"{summary['timestamp']}\n\n")

        dg = summary['analyses'].get('domain_gap', {})
        if dg.get('status') == 'ok':
            f.write("Domain gap:\n")
            f.write(f"  overlap: {dg['overlap']:.4f}\n")
            f.write(f"  1-NN accuracy: {dg['nn_accuracy']:.4f}\n")
            f.write(f"  cosine similarity: {dg['cosine']:.4f}\n")
            f.write(f"  gap ratio: {dg['gap_ratio']:.4f}\n")
            f.write(f"  mean EMD: {dg['mean_emd']:.6f}\n\n")
        elif dg:
            f.write(f"Domain gap: ERROR — {dg.get('error')}\n\n")

        cd = summary['analyses'].get('class_distribution', {})
        if cd.get('status') == 'ok':
            f.write("Class distribution:\n")
            f.write(f"  total bboxes: {cd['total_bboxes']}\n")
            f.write(f"  images with defects: {cd['images_with_defects']}\n")
            f.write(f"  empty images: {cd['empty_images']}\n")
            for cls_name, stats in cd.get('classes', {}).items():
                f.write(f"  {cls_name}: {stats['bbox_count']} bboxes ({stats['percentage']}%)\n")
        elif cd:
            f.write(f"Class distribution: ERROR — {cd.get('error')}\n")

    print(f"Saved: {txt_path}")


if __name__ == "__main__":
    sys.exit(main())