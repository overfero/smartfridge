"""
SmartFridge config loader.

Gunakan load_config() untuk mendapatkan konfigurasi lengkap sebagai SimpleNamespace berlapis.
Seluruh parameter runtime dibaca dari:
  - smartfridge/cfg/default.yaml           → konfigurasi utama
  - smartfridge/cfg/models/<name>.yaml     → parameter model
  - smartfridge/cfg/trackers/<name>.yaml   → parameter tracker
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

_CFG_DIR = Path(__file__).resolve().parent.parent / "cfg"


def _to_ns(value: object) -> object:
    """Rekursif: dict → SimpleNamespace, list tetap list, nilai primitif tetap."""
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_ns(v) for v in value]
    return value


def _postprocess(cfg: SimpleNamespace) -> None:
    """Normalisasi tipe setelah parsing YAML."""
    lines = cfg.lines
    lines.crossings = [
        (tuple(p1), tuple(p2)) for p1, p2 in lines.crossings
    ]
    lines.colors = [tuple(c) for c in lines.colors]
    cfg.ui.box_color = tuple(cfg.ui.box_color)
    cfg.ui.text_color = tuple(cfg.ui.text_color)


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """Muat dan gabungkan default.yaml + model yaml + tracker yaml.

    Args:
        path: Path ke file YAML utama. Default: smartfridge/cfg/default.yaml.

    Returns:
        SimpleNamespace berlapis dengan sections:
        camera, lines, counter, capture, ui, outputs, video, model, tracker.
    """
    cfg_path = Path(path) if path else _CFG_DIR / "default.yaml"
    with open(cfg_path) as f:
        raw: dict = yaml.safe_load(f)

    # Gabungkan model config
    model_name: str = raw["model"]
    model_yaml = _CFG_DIR / "models" / f"{model_name}.yaml"
    with open(model_yaml) as f:
        model_raw: dict = yaml.safe_load(f)
    raw["model"] = {"name": model_name, **model_raw}

    # Gabungkan tracker config
    tracker_name: str = raw["tracker"]
    tracker_yaml = _CFG_DIR / "trackers" / f"{tracker_name}.yaml"
    with open(tracker_yaml) as f:
        tracker_raw: dict = yaml.safe_load(f)
    raw["tracker"] = {"name": tracker_name, **tracker_raw}

    cfg = _to_ns(raw)
    _postprocess(cfg)
    return cfg
