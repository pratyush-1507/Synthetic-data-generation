"""Loads configs/pipeline.yaml and exposes the project paths used by every script."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"

ASSETS = ROOT / "assets" / "models"
RAW = ASSETS / "raw"
NORM = ASSETS / "normalized"
DEFORMED = ASSETS / "deformed"
LICENSE_CSV = ASSETS / "LICENSES.csv"
SYNTHETIC = ROOT / "data" / "synthetic"
REAL = ROOT / "data" / "real"
FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
TRAIN = ROOT / "train"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def class_names(cfg):
    """Class names ordered by COCO id."""
    return [name for name, c in sorted(cfg["classes"].items(), key=lambda kv: kv[1]["id"])]


def category_ids(cfg):
    return {name: c["id"] for name, c in cfg["classes"].items()}
