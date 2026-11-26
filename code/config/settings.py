"""Load and validate settings for the shared-surface RF reconstruction."""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from ..dataset_io.formats import dataset_root
from .data import load_data_settings, validate_algorithm_data_keys
from .evaluation import EVALUATION_DEFAULTS, validate_evaluation_settings

DEFAULTS = {
    **EVALUATION_DEFAULTS,
    "dataset_type": None,
    "platform_motion": "moving",
    "in_traj_folder": None,
    "geometry_json": None,
    "temporal_window": 5,
    "aperture": 301,
    "backend": "auto",
    "coarse_oversample": 4,
    "surface_candidates": 5,
    "surface_temporal_weight": 0.6,
    "surface_spatial_weight": 4.0,
    "surface_bp_iterations": 12,
    "surface_support_db": 9.0,
    "surface_min_views": 2,
    "min_range": 0.3,
    "max_range": 9.6,
    "keep_background": False,
    "oversample": 16,
    "surface_refine_radius_m": 0.075,
    "surface_refine_steps": 9,
    "frame_num": 2,
    "out_folder": None,
    "iterations": 80,
    "batch_size": 256,
    "learning_rate": 0.03,
    "range_bound_m": 0.075,
    "patch_shape": [3, 6],
    "geometry_weight": 0.05,
    "range_prior_weight": 0.02,
    "seed": 0,
    "temporal_fusion": "multilook",
    "min_coherence": 0.15,
    "dynamic_range_db": 20,
    "processed_traj_folder": None,
}


_PATH_SETTINGS = (
    "in_traj_folder",
    "out_folder",
    "geometry_json",
    "processed_traj_folder",
)


def load_imaging_settings(
    config_path: Path, data_config_path: Path, overrides: Mapping[str, object]
) -> SimpleNamespace:
    """Merge algorithm settings, dataset selection and command-line overrides."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Config must be a YAML mapping")
    unknown = config.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown, key=str)}")
    validate_algorithm_data_keys(config)

    settings = DEFAULTS | load_data_settings(data_config_path) | config
    settings.update(
        (key, value)
        for key, value in overrides.items()
        if key in DEFAULTS and value is not None
    )
    settings["data_config"] = data_config_path.resolve()
    _validate_imaging_settings(settings)
    root = dataset_root(settings["dataset_type"])
    for key in _PATH_SETTINGS:
        if settings[key] is not None:
            settings[key] = Path(settings[key]).expanduser().resolve()
    if root not in settings["in_traj_folder"].parents:
        raise ValueError(f"Input trajectory must be under {root}")
    return SimpleNamespace(**settings)


def _validate_imaging_settings(settings):
    for key in ("frame_num", "seed"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in (
        "iterations",
        "temporal_window",
        "batch_size",
        "aperture",
        "oversample",
        "coarse_oversample",
        "surface_candidates",
        "surface_bp_iterations",
        "surface_min_views",
        "surface_refine_steps",
    ):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        settings["aperture"] < 3
        or settings["aperture"] % 2 == 0
        or settings["temporal_window"] % 2 == 0
    ):
        raise ValueError("aperture must be odd and >=3; temporal_window must be odd")
    if settings["oversample"] < 2 or not 2 <= settings["surface_candidates"] <= 16:
        raise ValueError(
            "oversample must be >=2; surface_candidates must be between 2 and 16"
        )
    if (
        settings["surface_refine_steps"] < 3
        or settings["surface_refine_steps"] % 2 == 0
    ):
        raise ValueError("surface_refine_steps must be odd and >=3")
    for key in (
        "learning_rate",
        "range_bound_m",
        "surface_refine_radius_m",
        "min_range",
        "max_range",
        "surface_support_db",
        "geometry_weight",
        "range_prior_weight",
    ):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("surface_temporal_weight", "surface_spatial_weight"):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if (
        settings["surface_temporal_weight"] > 1
        or not settings["min_range"] < settings["max_range"]
    ):
        raise ValueError("Invalid temporal weight or range interval")
    shape = settings["patch_shape"]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or any(type(v) is not int or v < 1 for v in shape)
    ):
        raise ValueError("patch_shape must contain two positive integers")
    if shape[0] > 64 or shape[1] > 127:
        raise ValueError(
            "Local charts must fit within 64 elevation and 127 azimuth cells"
        )
    if type(settings["keep_background"]) is not bool:
        raise ValueError("keep_background must be boolean")
    if settings["backend"] not in ("auto", "cpu", "cuda") or settings[
        "platform_motion"
    ] not in ("static", "moving"):
        raise ValueError("Invalid backend or platform_motion")
    if settings["temporal_fusion"] not in ("multilook", "coherent"):
        raise ValueError("temporal_fusion must be multilook or coherent")
    if (
        not 0 <= settings["min_coherence"] <= 1
        or not np.isfinite(settings["dynamic_range_db"])
        or settings["dynamic_range_db"] <= 0
    ):
        raise ValueError("Invalid normal confidence thresholds")
    validate_evaluation_settings(settings)
