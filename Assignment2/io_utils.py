"""Shared paths and JSON helper (numpy-safe)."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"


def _default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_json(name, obj):
    """Write ``RESULTS_DIR/<name>.json`` (RESULTS_DIR looked up at call time)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    return path
