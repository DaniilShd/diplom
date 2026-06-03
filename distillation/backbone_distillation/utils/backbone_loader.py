import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _strip_prefix(key):
    prefixes = [
        "model.backbone.body.", "model.backbone.",
        "student_model.backbone.body.", "student_model.backbone.",
        "backbone.body.", "backbone.", "body.",
        "student_model.", "module.", "model.",
    ]
    for p in prefixes:
        if key.startswith(p):
            return key[len(p):]
    return key


def load_lightly_backbone(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    else:
        state_dict = ckpt
    
    body = model.backbone.body
    body_sd = body.state_dict()
    
    mapped = {}
    skipped_shape = []
    
    for k, v in state_dict.items():
        clean = _strip_prefix(k)
        if clean in body_sd:
            if v.shape == body_sd[clean].shape:
                mapped[clean] = v
            else:
                skipped_shape.append((clean, v.shape, body_sd[clean].shape))
    
    body.load_state_dict({**body_sd, **mapped}, strict=False)
    
    if len(mapped) == 0:
        logger.error(f"Ни один ключ не совпал при загрузке {ckpt_path}")
    elif len(mapped) < len(body_sd) * 0.5:
        logger.warning(f"Загружено только {len(mapped)}/{len(body_sd)} ключей")
    
    if skipped_shape:
        logger.warning(f"Пропущено {len(skipped_shape)} ключей из-за несовпадения формы")
    
    return len(mapped)