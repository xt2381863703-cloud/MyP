"""Timestamped platform poses and per-chirp rotating radar registration."""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation, Slerp

from ..dataset_io.formats import require_motion_output
from ..dataset_io.trajectory import RawTrajectory
from ..imaging.core import Geometry


class PoseSeries:
    """Timestamped world_from_platform transforms, in RF axes and meters."""

    def __init__(self, timestamps, transforms, source):
        self.timestamps = np.asarray(timestamps, dtype=float)
        self.transforms = np.asarray(transforms, dtype=float)
        if (
            self.timestamps.ndim != 1
            or len(self.timestamps) < 2
            or np.any(np.diff(self.timestamps) <= 0)
            or self.transforms.shape != (len(self.timestamps), 4, 4)
            or not np.isfinite(self.timestamps).all()
            or not np.isfinite(self.transforms).all()
            or not np.allclose(self.transforms[:, 3], [0, 0, 0, 1])
        ):
            raise ValueError(
                "Poses need increasing timestamps_s and finite (N,4,4) world_from_platform"
            )
        rotation = self.transforms[:, :3, :3]
        if not np.allclose(
            rotation @ rotation.swapaxes(-1, -2), np.eye(3), atol=1e-6
        ) or not np.allclose(np.linalg.det(rotation), 1):
            raise ValueError("Pose rotations must be proper orthonormal matrices")
        self.slerp = Slerp(self.timestamps, Rotation.from_matrix(rotation))
        self.source = source

    def at(self, timestamps):
        ts = np.atleast_1d(timestamps).astype(float)
        if np.any(ts < self.timestamps[0]) or np.any(ts > self.timestamps[-1]):
            raise ValueError(
                "Motion poses do not cover radar samples; extrapolation is disabled"
            )
        transforms = np.broadcast_to(np.eye(4), (len(ts), 4, 4)).copy()
        transforms[:, :3, :3] = self.slerp(ts).as_matrix()
        for axis in range(3):
            transforms[:, axis, 3] = np.interp(
                ts, self.timestamps, self.transforms[:, axis, 3]
            )
        return transforms


def estimated_pose_series(folder, separators, timestamps, dataset_type="PanoData"):
    """Load precomputed RF velocity and IMU angular velocity, then integrate.

    Estimate sign, direction correction and Gaussian smoothing reproduce
    panoradar_SP.dataset.get_motion_params(need_smooth=True). Only _esti and
    omega_imu keys are loaded, even if the file also stores LiDAR motion GT.
    Translation assumes constant velocity in each rotation's starting axes;
    yaw is integrated from the per-rotation IMU angular velocity. This is an
    SE(2) approximation, not calibrated sub-wavelength SE(3) ground truth.
    """
    path = require_motion_output(folder, dataset_type)
    with np.load(path, allow_pickle=False) as data:
        speed = data["v_esti"].astype(float)
        direction = data["theta_v_esti"].astype(float)
        omega = data["omega_imu"].astype(float)
    count = len(separators) - 1
    if any(
        a.shape != (count,) or not np.isfinite(a).all()
        for a in (speed, direction, omega)
    ):
        raise ValueError("Motion estimates must cover every radar rotation")
    direction[speed < 0] += np.pi
    if dataset_type == "SelfData":
        direction = (direction + np.pi) % (2 * np.pi) - np.pi
    else:
        direction %= 2 * np.pi
        direction[direction > 5] -= 2 * np.pi
        direction -= np.deg2rad(5.5)
    speed, direction, omega = [
        gaussian_filter(a, sigma=1.0, truncate=4)
        for a in (np.abs(speed), direction, omega)
    ]
    # The final separator can equal raw length. Its timestamp is the next
    # sampling instant, rather than a duplicate last sample.
    delta_t = float(np.median(np.diff(timestamps)))
    boundary_ts = np.interp(
        separators,
        np.arange(len(timestamps) + 1),
        np.r_[timestamps, timestamps[-1] + delta_t],
    )
    dt = np.diff(boundary_ts)
    yaw = np.r_[0, np.cumsum(omega * dt)]
    velocity = np.column_stack(
        (
            speed * np.cos(direction + yaw[:-1]),
            speed * np.sin(direction + yaw[:-1]),
            np.zeros(count),
        )
    )
    position = np.vstack((np.zeros(3), np.cumsum(velocity * dt[:, None], axis=0)))
    transforms = np.broadcast_to(np.eye(4), (count + 1, 4, 4)).copy()
    transforms[:, :3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    transforms[:, :3, 3] = position
    return PoseSeries(
        boundary_ts,
        transforms,
        f"{path.resolve()}: RF v_esti/theta_v_esti + omega_imu; approximate SE(2)",
    )


class MotionTrajectory(RawTrajectory):
    """Rotating radar with fixed or timestamped platform poses and real chirps."""

    def __init__(
        self,
        folder,
        subtract_background=True,
        platform_motion="moving",
        dataset_type="PanoData",
    ):
        if platform_motion not in ("static", "moving"):
            raise ValueError("platform_motion must be static or moving")
        super().__init__(folder, subtract_background, dataset_type, platform_motion)
        self.platform_motion = platform_motion
        if (
            len(self.timestamps) not in (len(self.raw), len(self.raw) + 1)
            or not np.isfinite(self.timestamps).all()
            or np.any(np.diff(self.timestamps) <= 0)
        ):
            raise ValueError(
                "Radar timestamps must be finite, increasing and cover every chirp, "
                "with at most one terminal boundary timestamp"
            )
        # Some trajectories also store the boundary immediately after the last
        # chirp. Keep it for frame-boundary timing; RF reads still stop at raw N.
        if platform_motion == "static":
            # Only the platform stays fixed. Motor phase and the rotating
            # antenna geometry below still vary with every actual chirp.
            self.poses = PoseSeries(
                self.timestamps[[0, -1]],
                np.repeat(np.eye(4)[None], 2, axis=0),
                "fixed platform; identity poses; radar motor rotation retained",
            )
        else:
            self.poses = estimated_pose_series(
                self.folder, self.separators, self.timestamps, dataset_type
            )

    def registered_frame(self, number, target, aperture, geometry_json=None):
        # Extend with actual adjacent samples; never wrap first/last chirps.
        metadata = self.frame_metadata(number)
        start, end = self.frame_bounds(number)
        lo = max(int(self.separators[0]), start - aperture)
        hi = min(int(self.separators[-1]), end + aperture, len(self.raw))
        frame = self.read_chirps(lo, hi)
        raw_ids = np.arange(lo, hi)
        rotations = np.clip(
            np.searchsorted(self.separators, raw_ids, side="right") - 1,
            0,
            self.frame_count - 1,
        )
        # Motor phase is defined in platform axes. External body rotation is
        # supplied separately by world_from_platform at EACH chirp timestamp.
        phase = (
            -2
            * np.pi
            * (raw_ids - self.separators[rotations])
            / np.diff(self.separators)[rotations]
        )
        local = Geometry.rotating(phase, self.radius, geometry_json, self.virtual_spacing_m)
        if self.dataset_type == "SelfData" and geometry_json is None:
            local.model = "suradar_effective_phase_centers"
        anchor = self.poses.at(self.timestamps[int(self.separators[target])])[0]
        poses = np.linalg.inv(anchor)[None] @ self.poses.at(self.timestamps[lo:hi])

        def transform(points):
            return (
                np.einsum("nij,ncj->nci", poses[:, :3, :3], points)
                + poses[:, None, :3, 3]
            )

        geometry = Geometry(
            transform(local.phase_centers_m),
            None if local.tx_m is None else transform(local.tx_m),
            None if local.rx_m is None else transform(local.rx_m),
            f"motion_registered_{local.model}",
        )
        facing = np.einsum(
            "nij,nj->ni",
            poses[:, :3, :3],
            np.column_stack((np.cos(phase), np.sin(phase), np.zeros(len(phase)))),
        )
        metadata.update(
            platform_motion=self.platform_motion,
            registered_raw_indices=[lo, hi],
            registration="per-chirp pose in target rotation-start RF coordinates",
            pose_source=self.poses.source,
            origin_in_target_m=poses[start - lo, :3, 3].tolist(),
            actual_neighbor_padding=True,
        )
        return frame, geometry, facing, poses[:, :3, 3], metadata
