"""Fixed panoramic normal coordinates; no sensor data or reference packages."""

from functools import lru_cache

import numpy as np

TRAINING_XYZ_TO_RADAR = np.array([1.0, -1.0, -1.0])


def panoramic_angles(azimuth_columns):
    """Return azimuth and elevation grid axes in radians."""
    azimuth = np.linspace(0, -2 * np.pi, azimuth_columns)[None, :]
    elevation = np.linspace(np.pi / 4, -np.pi / 4, 64)[:, None]
    return azimuth, elevation


@lru_cache(maxsize=2)
def normal_coordinate_frames(azimuth_columns=512):
    """Return the training RHS-to-XYZ basis and its inverse at each pixel."""
    theta, phi = panoramic_angles(azimuth_columns)
    azimuth = np.zeros((1, azimuth_columns, 3, 3))
    azimuth[..., 0, 0] = azimuth[..., 1, 1] = np.cos(theta)
    azimuth[..., 1, 0] = -np.sin(theta)
    azimuth[..., 0, 1] = np.sin(theta)
    azimuth[..., 2, 2] = 1
    elevation = np.zeros((64, 1, 3, 3))
    elevation[..., 0, 0] = elevation[..., 2, 2] = np.cos(phi)
    elevation[..., 2, 0] = -np.sin(phi)
    elevation[..., 0, 2] = np.sin(phi)
    elevation[..., 1, 1] = 1
    basis = azimuth @ elevation
    inverse = np.linalg.inv(basis)
    basis.setflags(write=False)
    inverse.setflags(write=False)
    return basis, inverse


def normal_to_rhs(normal):
    """Convert XYZ normals into the fixed panoramic basis, without masking."""
    return (normal_coordinate_frames(normal.shape[-2])[1] @ normal[..., None])[..., 0]


def rhs_to_normal(rhs):
    """Convert RHS labels to XYZ, retaining the invalid-label sentinel."""
    normal = (normal_coordinate_frames(rhs.shape[-2])[0] @ rhs[..., None])[..., 0]
    normal[np.all(rhs < -10, axis=-1)] = -1e3
    return normal


def normal_color_mapping(normal):
    """Encode XYZ directions in the RGB convention used by training labels."""
    with np.errstate(invalid="ignore", divide="ignore"):
        color = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    color[..., :2] = np.round((color[..., :2] + 1) * 127.5)
    color[..., 2] = np.round((color[..., 2] + 1) * 63.5) + 127
    color[np.isnan(color)] = 0
    return color.astype(np.uint8)
