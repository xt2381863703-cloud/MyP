"""Static SelfData evaluation settings; no sensor data is accessed here."""

import numpy as np

EVALUATION_DEFAULTS = {
    "lidar_translation_m": [0.0, 0.06, -0.4],
    "lidar_normal_radius_m": 0.25,
    "lidar_normal_neighbors": 32,
    "lidar_normal_min_neighbors": 6,
    "lidar_slice_tolerance_m": 0.1,
}


def validate_evaluation_settings(settings):
    translation = np.asarray(settings["lidar_translation_m"], dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("lidar_translation_m must contain three finite values")
    for key in ("lidar_normal_radius_m", "lidar_slice_tolerance_m"):
        value = settings[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("lidar_normal_neighbors", "lidar_normal_min_neighbors"):
        if type(settings[key]) is not int or settings[key] < 3:
            raise ValueError(f"{key} must be an integer >= 3")
    if settings["lidar_normal_min_neighbors"] > settings["lidar_normal_neighbors"]:
        raise ValueError("lidar_normal_min_neighbors exceeds lidar_normal_neighbors")
