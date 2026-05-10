import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(file)
        return json.load(file)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
