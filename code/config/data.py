"""Shared dataset selection for inference and standalone visualization entries."""

from pathlib import Path

import yaml

from ..data.formats import dataset_root
from .paths import CONFIG_ROOT, PROJECT_ROOT

DEFAULT_DATA_CONFIG = CONFIG_ROOT / "data.yaml"
DATA_KEYS = frozenset(("dataset_type", "in_traj_folder"))


def validate_data_settings(settings):
    if not isinstance(settings["dataset_type"], str):
        raise TypeError("dataset_type must be PanoData or SelfData")
    root = dataset_root(settings["dataset_type"])
    value = settings["in_traj_folder"]
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("in_traj_folder must be a nonempty path to one trajectory")
    folder = (PROJECT_ROOT / Path(value).expanduser()).resolve()
    if root not in folder.parents:
        raise ValueError(f"Input trajectory must be under {root}")
    settings["in_traj_folder"] = folder


def load_data_settings(path=DEFAULT_DATA_CONFIG):
    try:
        settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid data config {path}: {exc}") from exc
    if not isinstance(settings, dict):
        raise TypeError("Data config must be a YAML mapping")
    unknown = settings.keys() - DATA_KEYS
    missing = DATA_KEYS - settings.keys()
    if unknown or missing:
        raise ValueError(
            f"Data config keys: missing={sorted(missing)}, unknown={sorted(unknown, key=str)}"
        )
    validate_data_settings(settings)
    return settings


def validate_algorithm_data_keys(settings):
    duplicated = settings.keys() & DATA_KEYS
    if duplicated:
        raise ValueError(
            f"Move {sorted(duplicated)} from the algorithm config to data.yaml or --data-config"
        )
