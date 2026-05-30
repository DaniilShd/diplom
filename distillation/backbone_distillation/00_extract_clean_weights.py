#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import yaml
from collections import OrderedDict


def extract_clean_state_dict(checkpoint_path, output_dir):
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    
    if "train_model" not in ckpt:
        print("Ключ 'train_model' не найден")
        return None
    
    model_state = ckpt["train_model"]
    clean_state_dict = OrderedDict()
    prefix = "model.backbone.dinov3."
    
    for key, value in model_state.items():
        if key.startswith(prefix):
            clean_key = key[len(prefix):]
            clean_state_dict[clean_key] = value
    
    print(f"Ключей: {len(model_state)} -> {len(clean_state_dict)}")
    
    if len(clean_state_dict) == 0:
        print("Не найдено ключей с префиксом 'model.backbone.dinov3.'")
        return None
    
    output_path = output_dir / "teacher_clean_state_dict.pt"
    torch.save(clean_state_dict, str(output_path))
    print(f"Сохранён: {output_path} ({output_path.stat().st_size / (1024**2):.1f} MB)")
    
    return output_path


def main():
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir / "config.yaml"
    
    if not config_path.exists():
        print("config.yaml не найден")
        sys.exit(1)
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    checkpoint_path = cfg["teacher"]["teacher_weights"]
    output_path = extract_clean_state_dict(checkpoint_path, script_dir)
    
    if output_path:
        print("Готово. Обновите config.yaml: teacher_weights: 'teacher_clean_state_dict.pt'")
    else:
        print("Не удалось извлечь веса")
        sys.exit(1)


if __name__ == "__main__":
    main()