"""Post-inference references from synchronized SelfData 32-line LiDAR scans."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..config.evaluation import validate_evaluation_settings
from ..imaging.coordinates import TRAINING_XYZ_TO_RADAR, normal_to_rhs


class SelfLidarReference:
    def __init__(self, raw_folder, min_range, max_range, *, azimuth_columns=512, **settings):
        validate_evaluation_settings(settings)
        if type(azimuth_columns) is not int or azimuth_columns not in (512, 1024):
            raise ValueError("SelfData expects 512 or 1024 panorama columns")
        self.folder = Path(raw_folder)
        self.minimum, self.maximum = min_range, max_range
        self.azimuth_columns = azimuth_columns
        self.settings = settings
        with np.load(self.folder / "data_params.npz", allow_pickle=False) as data:
            self.separators = data["lidar_sep"].copy()
            self.radar_sep = data["radar_sep"].copy()
            # SURadar _load_data_params transposes this matrix before row-vector use.
            xy = np.asarray(data["transform"], dtype=float).T
        if (
            xy.shape != (2, 2)
            or not np.isfinite(xy).all()
            or not np.allclose(xy @ xy.T, np.eye(2), atol=1e-5)
            or not np.isclose(np.linalg.det(xy), 1, atol=1e-5)
        ):
            raise ValueError("SelfData LiDAR transform must be a proper (2,2) rotation")
        self.rotation = np.eye(3)
        self.rotation[:2, :2] = xy
        self.translation = np.asarray(settings["lidar_translation_m"], dtype=float)
        # SelfData stores structured scans in an object array. Pickle is confined
        # to this evaluation-only loader for the user's existing local dataset.
        with np.load(self.folder / "point_cloud_raw.npz", allow_pickle=True) as data:
            self.clouds = data["points"]
            self.timestamps = data["timestamps"].copy()
        with np.load(self.folder / "radar.npz", allow_pickle=False) as data:
            self.radar_timestamps = data["timestamps"].copy()
        if (
            self.separators.shape != self.radar_sep.shape
            or not np.issubdtype(self.separators.dtype, np.integer)
            or self.separators[0] < 0
            or self.separators[-1] > len(self.clouds)
            or np.any(np.diff(self.separators) < 1)
        ):
            raise ValueError("Invalid synchronized SelfData lidar_sep")
        if (
            self.timestamps.shape != (len(self.clouds),)
            or not np.isfinite(self.timestamps).all()
            or np.any(np.diff(self.timestamps) <= 0)
        ):
            raise ValueError("Invalid SelfData LiDAR timestamps")

    def points(self, frame_num):
        if not 0 <= frame_num < len(self.separators) - 1:
            raise ValueError("SelfData LiDAR frame is out of range")
        index = int(self.separators[frame_num])
        raw = self.clouds[index]
        if (
            raw.ndim != 1
            or raw.dtype.names is None
            or not {"x", "y", "z"} <= set(raw.dtype.names)
        ):
            raise ValueError("SelfData LiDAR scan needs structured x/y/z fields")
        points = np.column_stack([raw[key] for key in ("x", "y", "z")]).astype(float)
        valid = np.isfinite(points).all(axis=-1) & (
            np.linalg.norm(points, axis=-1) > 1e-6
        )
        points = points[valid] @ self.rotation + self.translation
        points = points[np.linalg.norm(points, axis=-1) > 1e-6]
        if not len(points):
            raise ValueError("SelfData LiDAR scan has no finite nonzero points")
        metadata = {
            "method": "selfdata_synchronized_raw_lidar",
            "raw_lidar_index": index,
            "timestamp_s": float(self.timestamps[index]),
            "radar_start_timestamp_s": float(
                self.radar_timestamps[self.radar_sep[frame_num]]
            ),
            "lidar_minus_radar_start_s": float(
                self.timestamps[index]
                - self.radar_timestamps[self.radar_sep[frame_num]]
            ),
            "frame_mapping": "data_params.npz lidar_sep[frame_num]; no prediction-dependent alignment",
            "points": len(points),
            "coordinates": "row-vector XYZ @ block_diag(transform.T,1) + lidar_translation_m",
            "lidar_rotation_row_matrix": self.rotation.tolist(),
            "lidar_translation_m": self.translation.tolist(),
            "glass_labels_available": False,
            "rf_influence": "none; all reference masks and geometry depend only on LiDAR and fixed settings",
            "limitations": "single synchronized spinning scan without per-point deskew; sparse measured references, no glass exclusion",
        }
        return points, metadata

    def load(self, frame_num):
        points, metadata = self.points(frame_num)
        ranges = np.linalg.norm(points, axis=-1)
        elevation = np.arcsin(np.clip(points[:, 2] / ranges, -1, 1))
        allowed = (
            (ranges >= self.minimum)
            & (ranges < self.maximum)
            & (np.abs(elevation) <= np.pi / 4)
        )
        if not allowed.any():
            raise ValueError("No SelfData LiDAR points in the evaluation range/FoV")
        selected = np.flatnonzero(allowed)
        unique_columns = self.azimuth_columns - 1
        row = np.rint((np.pi / 4 - elevation[selected]) * 63 / (np.pi / 2)).astype(int)
        column = (
            np.rint(
                np.mod(-np.arctan2(points[selected, 1], points[selected, 0]), 2 * np.pi)
                * unique_columns
                / (2 * np.pi)
            ).astype(int)
            % unique_columns
        )
        cell = row * unique_columns + column
        order = np.lexsort((ranges[selected], cell))
        _, first = np.unique(cell[order], return_index=True)
        chosen = selected[order[first]]
        cells = cell[order[first]]
        count = 64 * unique_columns
        position = np.full((count, 3), np.nan)
        depth = np.full(count, np.nan)
        normal = np.full((count, 3), np.nan)
        position[cells], depth[cells] = points[chosen], ranges[chosen]
        radius = self.settings["lidar_normal_radius_m"]
        neighbors = self.settings["lidar_normal_neighbors"]
        distances, ids = cKDTree(points).query(
            points[chosen], k=neighbors, distance_upper_bound=radius, workers=-1
        )
        valid = np.isfinite(distances)
        samples = points[np.minimum(ids, len(points) - 1)]
        sizes = valid.sum(axis=1)
        center = np.where(valid[..., None], samples, 0).sum(axis=1) / np.maximum(
            sizes[:, None], 1
        )
        delta = np.where(valid[..., None], samples - center[:, None], 0)
        covariance = np.einsum("nki,nkj->nij", delta, delta) / np.maximum(
            sizes[:, None, None], 1
        )
        values, vectors = np.linalg.eigh(covariance)
        normals = vectors[:, :, 0]
        normals *= np.where(np.einsum("ij,ij->i", normals, points[chosen]) < 0, -1, 1)[
            :, None
        ]
        defined = (sizes >= self.settings["lidar_normal_min_neighbors"]) & (
            values[:, 1] > 1e-10
        )
        normal[cells[defined]] = normals[defined]

        def panorama(value):
            value = value.reshape(64, unique_columns, *value.shape[1:])
            return np.concatenate((value, value[:, :1]), axis=1)

        position, depth, normal = map(panorama, (position, depth, normal))
        range_mask = np.isfinite(depth)
        normal_mask = range_mask & np.isfinite(normal).all(axis=-1)
        # The panorama endpoint is a duplicate, not a second independent sample.
        range_mask[:, -1] = False
        normal_mask[:, -1] = False
        training_xyz = np.where(
            normal_mask[..., None], normal * TRAINING_XYZ_TO_RADAR, -1e3
        ).astype(np.float32)
        rhs = normal_to_rhs(training_xyz).astype(np.float32)
        rhs[~normal_mask] = -1e3
        reference = {
            "points_m": position.reshape(-1, 3),
            "normal": normal.reshape(-1, 3),
            "defined": normal_mask.ravel(),
            "valid": normal_mask.ravel(),
            "position_valid": range_mask.ravel(),
            "pixel_row_col": np.stack(np.indices(depth.shape), axis=-1).reshape(-1, 2),
            "range_image_m": depth.astype(np.float32),
            "training_rhs": rhs.transpose(2, 0, 1),
            "training_depth": np.where(range_mask, depth / 10, -1e3)[None].astype(
                np.float32
            ),
            "training_normal_xyz": training_xyz,
            "normal_evaluation_mask": normal_mask,
            "range_evaluation_mask": range_mask,
            "training_xyz_to_radar": TRAINING_XYZ_TO_RADAR,
        }
        metadata.update(
            method="selfdata_raw_lidar_local_pca",
            valid_ranges=int(range_mask.sum()),
            valid_normals=int(normal_mask.sum()),
            normal_radius_m=radius,
            normal_max_neighbors=neighbors,
            normal_min_neighbors=self.settings["lidar_normal_min_neighbors"],
            normal_method="local LiDAR PCA, outward orientation; no planarity/curvature selection",
            projection=f"nearest measured range in each 64x{unique_columns} angular cell; duplicated endpoint excluded from scoring; no interpolation or hole filling",
        )
        return reference, metadata

    def load_pair(self, frame_num):
        reference, metadata = self.load(frame_num)
        metadata.update(
            glass_pixels=None,
            masked_unmasked_identical=True,
            unmasked_reference="SelfData has no per-pixel glass labels; both references retain all measured cells with a defined local PCA normal; no glass mask is inferred",
        )
        return reference, reference, metadata
