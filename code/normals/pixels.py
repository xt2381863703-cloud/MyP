"""Pack RF-only full-panorama normal predictions."""

import numpy as np

from ..imaging.coordinates import TRAINING_XYZ_TO_RADAR, normal_to_rhs


def pack_normal_image(estimated, inverse, points):
    """Use the same pixel and orientation conventions for either RF estimator."""
    image_shape = points.shape[:2]
    field = {
        key: value[inverse].reshape((*image_shape, *value.shape[1:]))
        for key, value in estimated.items()
    }
    # Fixed convention conversion, independent of labels and prediction quality.
    outward = -field["normal"]
    defined = field["defined"] & np.isfinite(outward).all(axis=-1)
    outward = np.where(defined[..., None], outward, 0)
    training_xyz = outward * TRAINING_XYZ_TO_RADAR
    rhs = normal_to_rhs(training_xyz).transpose(2, 0, 1).astype(np.float32)
    field.update(
        {
            "points_m": points,
            "normal_comparison": outward,
            "pred_rhs": rhs,
            "predicted_range_m": np.linalg.norm(points, axis=-1),
        }
    )
    return field
