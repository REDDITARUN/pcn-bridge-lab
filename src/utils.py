import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, value = item.split("=", 1)
        set_nested(cfg, key.split("."), parse_value(value))
    return cfg


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def set_nested(cfg: Dict[str, Any], keys: list[str], value: Any) -> None:
    cur = cfg
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(base_dir: str, name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / f"{stamp}-{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def get_peak_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def exact_match(pred: str, target: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(target))


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().replace(".", "").split())


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
