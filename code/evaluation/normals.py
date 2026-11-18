"""Post-inference per-pixel normal and range metrics."""

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SHAPE = (64, 512)
SCORE_KEYS = ("sn_angle_mean", "sn_angle_median", "sn_angle_80", "sn_angle_90")


def evaluate_normal_image(
    pred_rhs, reference, defined=None, confident=None, predicted_range=None
):
    """The exact mask, cosine and quantiles in SnEvaluator.process."""
    pred_rhs = np.asarray(pred_rhs, dtype=np.float32)
    if pred_rhs.shape != (3, 64, 512):
        raise ValueError("Prediction must have shape (3,64,512)")
    finite = np.isfinite(pred_rhs).all(axis=0)
    pred_rhs = np.where(finite[None], pred_rhs, 0)
    pred = torch.tensor(pred_rhs.tolist(), dtype=torch.float32)
    target = torch.tensor(reference["training_rhs"].tolist(), dtype=torch.float32)
    depth = torch.tensor(reference["training_depth"][0].tolist(), dtype=torch.float32)
    mask = torch.logical_and(target[0] > -10, depth < 0.96)
    if "normal_evaluation_mask" in reference:
        mask = torch.tensor(reference["normal_evaluation_mask"].tolist(), dtype=torch.bool)
    cosine = F.cosine_similarity(pred, target, dim=0, eps=1e-4).clamp(-1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine[mask]))
    mask_np = np.asarray(mask.tolist(), dtype=bool)
    error = np.full(IMAGE_SHAPE, np.nan, np.float32)
    error[mask_np] = np.asarray(angles.tolist(), dtype=np.float32)
    predicted = finite & (np.linalg.norm(pred_rhs, axis=0) > 1e-12)
    if defined is not None:
        predicted &= np.asarray(defined, bool)
    confidence = predicted if confident is None else predicted & confident
    count = int(mask.sum())
    metrics = {key: None for key in SCORE_KEYS}
    if count:
        metrics = {
            "sn_angle_mean": float(angles.mean()),
            "sn_angle_median": float(angles.quantile(0.5)),
            "sn_angle_80": float(angles.quantile(0.8)),
            "sn_angle_90": float(angles.quantile(0.9)),
        }
    metrics.update(
        {
            "evaluation_pixels": count,
            "defined_prediction_pixels": int((mask_np & predicted).sum()),
            "confident_prediction_pixels": int((mask_np & confidence).sum()),
            "prediction_coverage": float((mask_np & predicted).sum() / count)
            if count
            else None,
            "invalid_prediction_pixels": int((mask_np & ~predicted).sum()),
            "primary_mask": "measured SelfData LiDAR cells with a defined local PCA normal; configured range; no duplicate endpoint"
            if "normal_evaluation_mask" in reference
            else "gt_rhs[0]>-10 AND gt_depth<0.96, exactly SnEvaluator",
            "evaluation_method": "SelfData LiDAR PCA" if "normal_evaluation_mask" in reference else "SnEvaluator",
            "prediction_confidence_filters_primary_score": False,
            "invalid_prediction_policy": "zero RHS; F.cosine_similarity gives 0 and angle is 90 deg",
        }
    )
    if count:
        metrics["angle_rmse_deg"] = float(torch.sqrt(torch.mean(angles**2)))
        for threshold in (5, 10, 20):
            metrics[f"fraction_within_{threshold}_deg"] = float(
                (angles <= threshold).float().mean()
            )
    if predicted_range is not None:
        gt_range = reference["range_image_m"]
        range_mask = reference.get("range_evaluation_mask", mask_np)
        positive = range_mask & np.isfinite(gt_range) & (gt_range > 0)
        distance_error = np.abs(predicted_range[positive] - gt_range[positive])
        metrics["range_mae_m"] = (
            float(distance_error.mean()) if len(distance_error) else None
        )
        metrics["range_evaluation_pixels"] = int(positive.sum())
        metrics["range_median_abs_error_m"] = (
            float(np.median(distance_error)) if len(distance_error) else None
        )
    return metrics, error, mask_np
