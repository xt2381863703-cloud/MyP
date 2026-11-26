"""Post-finalization errors for a shared surface, its RF normals and baseline."""

import numpy as np

from .normals import evaluate_normal_image


def _statistics(values):
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p90": float(np.quantile(values, 0.9)) if len(values) else None,
    }


def compare(prediction, reference, args, *, pose_refinement_enabled=True):
    """GT-only masks; neither confidence nor error magnitude selects score pixels."""
    depth = reference["range_image_m"]
    mask = np.isfinite(depth) & (depth >= args.min_range) & (depth < args.max_range)
    if "range_evaluation_mask" in reference:
        mask &= reference["range_evaluation_mask"]
    mask[:, -1] = False
    truth_points = reference["points_m"].reshape(*depth.shape, 3)
    position_mask = mask & np.isfinite(truth_points).all(axis=-1)
    arrays = {"range_mask": mask, "position_mask": position_mask}
    trajectory = "corrected trajectory" if pose_refinement_enabled else "input trajectory"
    evaluation = {
        "range": {},
        "position": {},
        "normal": {},
        "mask_uses_rf_confidence": False,
        "range_mask": "same finite measured LiDAR range in the configured interval, retaining glass regions and excluding the duplicate panorama endpoint",
        "position_mask": "same range mask with finite measured XYZ; paired panorama pixels, not prediction-selected nearest neighbors",
        "scope": (
            "PanoRadar baseline uses native FFT peaks from the target circle; "
            f"current range, points and echo normals share one finalized surface and {trajectory}"
        ),
    }
    for name, range_key, point_key in (
        ("panoradar", "panoradar_range_m", "panoradar_points_m"),
        ("current", "predicted_range_m", "points_m"),
    ):
        range_error = np.full(depth.shape, np.nan, dtype=np.float32)
        range_error[mask] = np.abs(prediction[range_key][mask] - depth[mask])
        point_error = np.full(depth.shape, np.nan, dtype=np.float32)
        point_error[position_mask] = np.linalg.norm(
            prediction[point_key][position_mask] - truth_points[position_mask], axis=-1
        )
        arrays[f"range_error_{name}"] = range_error
        arrays[f"position_error_{name}"] = point_error
        evaluation["range"][name] = _statistics(range_error[mask])
        evaluation["position"][name] = _statistics(point_error[position_mask])
    metrics, error, normal_mask = evaluate_normal_image(
        prediction["pred_rhs"],
        reference,
        prediction["normal_defined"],
        prediction["normal_valid"],
    )
    evaluation["normal"]["filtered"] = metrics
    arrays["normal_error_filtered"] = error
    arrays["normal_mask_filtered"] = normal_mask
    return evaluation, arrays
