"""Independent reconstruction of training labels, for post-inference scoring.

Projection, filtering and coordinate conventions follow PanoRadar's published
label pipeline. No PanoRadar module or reference source is loaded at runtime.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import correlate1d
from torchvision.transforms.functional import resize

from ..imaging.coordinates import (
    TRAINING_XYZ_TO_RADAR,
    normal_color_mapping,
    normal_coordinate_frames,
    normal_to_rhs,
    rhs_to_normal,
)

REFERENCE_METHOD = "panoradar_original_training_range_gradient"


def _project_range_image(points, transform):
    """Rasterize organized LiDAR beams with clockwise azimuth and last-hit wins."""
    points = np.asarray(points)
    transform = np.asarray(transform)
    if points.ndim != 3 or points.shape[0] != 64 or points.shape[-1] != 3:
        raise ValueError("LiDAR points must have shape (64,azimuth,3)")
    if points.shape[1] == 0 or not np.isfinite(points).all():
        raise ValueError("LiDAR points must be nonempty and finite")
    if transform.shape != (2, 2) or not np.isfinite(transform).all():
        raise ValueError("LiDAR alignment must be a finite (2,2) matrix")
    rotation = np.eye(3)
    rotation[:2, :2] = transform
    aligned = points @ rotation[None]
    ranges = np.linalg.norm(aligned, axis=-1)
    angle = np.arctan2(aligned[..., 1], aligned[..., 0])
    angle[angle < 0] += 2 * np.pi
    width = points.shape[1]
    columns = np.rint(angle / (2 * np.pi / width)).astype(np.int32)
    image = np.zeros((64, width), np.float32)
    for row in range(64):
        valid = (columns[row] >= 0) & (columns[row] < width)
        image[row, columns[row, valid]] = ranges[row, valid]
    return image[:, ::-1]


def _training_rhs(depth):
    """Compute filtered gradient labels, including RGB quantization and cables."""
    filtered = cv2.medianBlur(depth, 3)
    filtered[filtered <= 0] = np.nan
    theta_step = 2 * np.pi / 511
    phi_step = (np.pi / 2) / 63
    gradients = []
    for axis, step, mode in ((1, theta_step, "wrap"), (0, phi_step, "constant")):
        left = (
            correlate1d(filtered, [1.0, -1.0, 0.0], axis=axis, mode=mode, cval=np.nan)
            / step
        )
        right = (
            correlate1d(filtered, [0.0, 1.0, -1.0], axis=axis, mode=mode, cval=np.nan)
            / step
        )
        gradients.append(np.where(np.abs(left) < np.abs(right), left, right))
    cosine = np.cos(np.linspace(np.pi / 4, -np.pi / 4, 64))[:, None]
    rhs = np.stack(
        (
            np.ones_like(filtered),
            1 / (filtered * cosine) * gradients[0],
            1 / filtered * gradients[1],
        ),
        axis=-1,
    )
    normal = (normal_coordinate_frames()[0] @ rhs[..., None])[..., 0]
    color = cv2.bilateralFilter(normal_color_mapping(normal), 15, 50, 50)
    normal = color.astype(np.float32)
    normal[..., :2] /= 127.5
    normal[..., 2] = (normal[..., 2] - 127) / 63.5
    normal -= 1
    invalid = np.all(color == 0, axis=-1)
    normal[invalid] = -1e3
    lower = invalid[44:63].astype(np.uint8) * 255
    lower = cv2.dilate(lower, np.ones((3, 7), np.uint8))
    lower = cv2.dilate(lower, np.ones((1, 7), np.uint8))
    invalid[44:63] = lower.astype(bool)
    rhs = normal_to_rhs(normal)
    rhs[invalid] = -1e3
    return rhs.astype(np.float32)


def range_image_from_raw(points, transform):
    """The LiDAR branch of process_raw_data.process_data, for identity checking.

    List-based tensor conversion avoids the installed Torch 2.0/NumPy ABI bridge;
    the arithmetic, bilinear resize and invalid-pixel replacement are unchanged.
    """
    image = _project_range_image(points, transform)
    tensor = torch.tensor(image.tolist(), dtype=torch.float32) / 10
    tensor[tensor == 0] = -1e3
    depth = np.asarray(
        resize(tensor.unsqueeze(0), (64, 512), antialias=False).tolist(),
        dtype=np.float32,
    )
    median = cv2.medianBlur(depth[0], 3)[None]
    failed = depth < 0
    depth[failed] = median[failed]
    return depth


def training_normal_reference(depth, glass):
    """Reproduce RfFourTasksMapper's pre-augmentation normal label exactly."""
    depth = np.asarray(depth, dtype=np.float32).reshape(64, 512).copy()
    glass = np.asarray(glass, dtype=bool).reshape(64, 512)
    masked_depth = depth.copy()
    masked_depth[glass] = -1e3
    rhs = _training_rhs(masked_depth)
    training_xyz = rhs_to_normal(rhs)
    label_valid = rhs[..., 0] > -10
    evaluation_mask = label_valid & (masked_depth < 0.96)
    physical = training_xyz * TRAINING_XYZ_TO_RADAR
    length = np.linalg.norm(physical, axis=-1)
    defined = label_valid & np.isfinite(physical).all(axis=-1) & (length > 1e-12)
    normal = np.full_like(physical, np.nan)
    normal[defined] = physical[defined] / length[defined, None]
    rays = normal_coordinate_frames()[0][..., :, 0] * TRAINING_XYZ_TO_RADAR
    position_valid = np.isfinite(depth) & (depth > 0)
    points = rays * (depth * 10)[..., None]
    points[~position_valid] = np.nan
    # Scoring must use the exact original mask, including any boundary pixels
    # filled by its filters. Position validity never changes pixel scoring.
    valid = evaluation_mask
    return {
        "points_m": points.reshape(-1, 3),
        "normal": normal.reshape(-1, 3),
        "defined": defined.ravel(),
        "valid": valid.ravel(),
        "position_valid": position_valid.ravel(),
        "pixel_row_col": np.stack(np.indices((64, 512)), axis=-1).reshape(-1, 2),
        "range_image_m": depth * 10,
        "glass_mask": glass,
        "training_rhs": rhs.transpose(2, 0, 1),
        "training_depth": masked_depth[None],
        "training_normal_xyz": training_xyz,
        "training_label_valid": label_valid,
        "original_evaluation_mask": evaluation_mask,
        "training_xyz_to_radar": TRAINING_XYZ_TO_RADAR,
    }


class TrainingLidarReference:
    """Match raw frames to processed labels by their full range images."""

    def __init__(self, raw_folder, processed_folder):
        self.raw_folder = Path(raw_folder)
        self.processed_folder = Path(processed_folder)
        self.paths = sorted((self.processed_folder / "lidar_npy").glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(
                f"No original training LiDAR labels in {self.processed_folder / 'lidar_npy'}"
            )
        self.depths = [np.load(path, allow_pickle=False) for path in self.paths]
        if any(
            depth.shape != (1, 64, 512) or depth.dtype != np.float32
            for depth in self.depths
        ):
            raise ValueError("Original training depth files must be float32 (1,64,512)")
        self.depth_index = {}
        for i, depth in enumerate(self.depths):
            self.depth_index.setdefault(
                hashlib.sha256(depth.tobytes()).digest(), []
            ).append(i)
        self.raw = np.load(self.raw_folder / "lidar_data.npy", mmap_mode="r")
        with np.load(self.raw_folder / "data_params.npz") as params:
            self.separators = params["lidar_sep"].copy()
            self.transform = params["transform"].copy()
        self.timestamps = np.loadtxt(self.raw_folder / "lidar_ts.txt", ndmin=1)

    def load(self, frame_num):
        if not 0 <= frame_num < len(self.separators) - 1:
            raise ValueError("LiDAR frame is outside the synchronized frame range")
        index = int(self.separators[frame_num])
        if not 0 <= index < len(self.raw):
            raise ValueError("Synchronized LiDAR frame does not exist")
        reconstructed = range_image_from_raw(
            np.asarray(self.raw[index]), self.transform
        )
        matches = self.depth_index.get(
            hashlib.sha256(reconstructed.tobytes()).digest(), []
        )
        matches = [i for i in matches if np.array_equal(reconstructed, self.depths[i])]
        if len(matches) != 1:
            raise ValueError(
                f"Frame {frame_num}: found {len(matches)} exact processed depth matches; cannot safely select a glass mask"
            )
        chosen = matches[0]
        depth_path = self.paths[chosen]
        glass_path = self.processed_folder / "glass_npy" / depth_path.name
        glass = np.load(glass_path, allow_pickle=False)
        if glass.shape != (1, 64, 512) or glass.dtype != bool:
            raise ValueError(f"Invalid original training glass mask: {glass_path}")
        reference = training_normal_reference(self.depths[chosen], glass)
        metadata = {
            "method": REFERENCE_METHOD,
            "raw_lidar_index": index,
            "timestamp_s": float(self.timestamps[index]),
            "depth_file": str(depth_path),
            "glass_mask_file": str(glass_path),
            "frame_mapping": "exact equality of raw-reconstructed and processed (1,64,512) depth arrays",
            "points": len(reference["points_m"]),
            "valid_normals": int(reference["valid"].sum()),
            "pipeline": "depth[glass]=-1000; get_smooth_surface_normal(sn,depth); sn.get_rhs_from_normal(normal,True).astype(float32)",
            "label_source": "code.evaluation.lidar: independent implementation of the published training-label convention",
            "evaluation_mask": "rhs[0]>-10 and masked depth<0.96, exactly as SnEvaluator",
            "training_depth_units": "range_m / 10",
            "coordinates": "original training XYZ rotated into RF XYZ with diag(1,-1,-1)",
            "rf_orientation": "negate RF point-to-antenna normals to original outward label convention",
            "angular_metric": "SnEvaluator: RHS float32 cosine_similarity(dim=0,eps=1e-4), acos, per-frame quantiles",
            "rf_influence": "none: label generation and frame verification are independent of RF",
        }
        return reference, metadata

    def load_pair(self, frame_num):
        """Compute both real training-label policies after RF finalization."""
        reference, metadata = self.load(frame_num)
        depth = np.load(metadata["depth_file"], allow_pickle=False)
        unmasked = training_normal_reference(depth, np.zeros((64, 512), dtype=bool))
        metadata.update(
            glass_labels_available=True,
            glass_pixels=int(reference["glass_mask"].sum()),
            masked_unmasked_identical=bool(
                np.array_equal(reference["training_rhs"], unmasked["training_rhs"])
            ),
            unmasked_reference="same measured depth and original normal-label processing, without applying the glass mask; returns through glass are not glass-surface truth",
        )
        return reference, unmasked, metadata
