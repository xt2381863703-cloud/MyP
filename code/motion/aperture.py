"""Select distinct real chirps facing a candidate point."""

import numpy as np


def motion_aperture_indices(points, origins, facing, count):
    """Choose distinct actual observations facing each point from its new origin.

    Unlike cyclic array padding, a split interval at a panorama seam refers to
    real, distinct timestamps/poses. FOV is the existing 90 degrees.
    """
    if count < 3 or count % 2 != 1 or count > len(origins):
        raise ValueError("Invalid moving aperture count")
    ray = np.asarray(points)[:, None] - np.asarray(origins)[None]
    norm = np.linalg.norm(ray, axis=-1)
    cosine = np.einsum("bnj,nj->bn", ray, facing) / np.maximum(norm, 1e-12)
    indices = np.sort(np.argpartition(-cosine, count - 1, axis=1)[:, :count], axis=1)
    valid = np.take_along_axis(cosine, indices, axis=1) >= np.cos(np.pi / 4)
    return indices, valid
