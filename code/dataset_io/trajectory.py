"""Read-only RF trajectory loading and output-folder validation."""

from pathlib import Path

import numpy as np

from .formats import DATASET_ROOTS, dataset_root
from .pano_data import PanoData
from .self_data import SelfData


def prepare_output_folder(folder, input_folder=None):
    """Resolve symlinks before allowing any experiment output to be created."""
    folder = Path(folder).resolve()
    protected = [root.resolve() for root in DATASET_ROOTS.values()]
    if input_folder is not None:
        protected.append(Path(input_folder).resolve())
    if any(folder == root or root in folder.parents for root in protected):
        raise ValueError(
            "Outputs must be outside both dataset roots and the input trajectory"
        )
    folder.mkdir(parents=True, exist_ok=True)
    return folder


class RawTrajectory:
    """Read-only trajectory with a single streamed background pass per run."""

    def __init__(self, folder, subtract_background=True, dataset_type="PanoData", platform_motion="moving"):
        self.folder = Path(folder)
        dataset_root(dataset_type)
        source = SelfData(folder, platform_motion) if dataset_type == "SelfData" else PanoData(folder)
        self.dataset_type = dataset_type
        self.separators, self.radius, self.raw = source.separators, source.radius, source.raw
        self.timestamps, self.waveform = source.timestamps, source.waveform
        self.virtual_spacing_m = source.virtual_spacing_m
        self.frame_ends, self.framing = source.frame_ends, source.framing
        if (
            self.raw.ndim != 4
            or self.raw.shape[1:3] != (2, 4)
            or not np.iscomplexobj(self.raw)
        ):
            raise ValueError(
                "Expected complex raw data with shape (positions,2,4,samples)"
            )
        if self.separators.ndim != 1 or len(self.separators) < 2 or not np.issubdtype(self.separators.dtype, np.integer) or self.separators[0] < 0 or self.separators[-1] > len(self.raw) or np.any(np.diff(self.separators) < 3) or not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Invalid radar separation indices or rotation radius")
        self.frame_count = len(self.separators) - 1
        self.subtract_background = subtract_background
        self.background = None

    def frame_bounds(self, frame_num):
        if not 0 <= frame_num < self.frame_count:
            raise ValueError(f"frame_num must be in [0,{self.frame_count - 1}]")
        return int(self.separators[frame_num]), int(self.frame_ends[frame_num])

    def core_bounds(self, frame_num):
        start, end = self.frame_bounds(frame_num)
        return start, min(end, int(self.separators[frame_num + 1]))

    def read_chirps(self, start, end):
        """Apply the shared background and scale to one chronological interval."""
        frame = np.array(self.raw[start:end], dtype=np.complex64).reshape(
            end - start, 8, self.raw.shape[-1]
        )
        if self.subtract_background:
            if self.background is None:
                total = np.zeros(self.raw.shape[1:], dtype=np.complex128)
                for offset in range(0, len(self.raw), 2048):
                    total += self.raw[offset : offset + 2048].sum(
                        axis=0, dtype=np.complex128
                    )
                self.background = (
                    (total / len(self.raw)).reshape(8, -1).astype(np.complex64)
                )
            frame -= self.background[None]
        return frame / 32768

    def frame_metadata(self, frame_num):
        if not 0 <= frame_num < self.frame_count:
            raise ValueError(f"frame_num must be in [0,{self.frame_count - 1}]")
        start, end = self.frame_bounds(frame_num)
        if start < 0 or end > len(self.raw) or end - start < 3:
            raise ValueError(
                "Selected frame is incomplete or its separation indices are invalid"
            )
        timestamps = self.timestamps[start:end]
        if len(timestamps) != end - start or not np.isfinite(timestamps).all():
            raise ValueError("Radar timestamps do not cover this frame")
        return {
            "trajectory": str(self.folder.resolve()),
            "dataset_type": self.dataset_type,
            "framing": self.framing,
            "frame_num": frame_num,
            "raw_frame_indices": [start, end],
            "core_frame_indices": list(self.core_bounds(frame_num)),
            "native_frame_indices": self.separators[frame_num:frame_num + 2].tolist(),
            "rotation_radius_m": self.radius,
            "rotation_positions": end - start,
            "raw_channel_order": "reshape(2 Tx,4 Rx) in C order, as dataset.py",
            "background": "trajectory-wide complex channel mean"
            if self.subtract_background
            else "none",
            "normalization": "raw I/Q divided by 32768",
            "angle_model": "uniform clockwise rotation, beam 0 = +x, matching dataset.py",
            "timestamp_span_s": float(timestamps[-1] - timestamps[0]),
            "waveform": {
                "samples": self.waveform.samples,
                "start_hz": self.waveform.start_hz,
                "range_bandwidth_hz": self.waveform.bandwidth_hz,
                "spatial_step_hz": self.waveform.spatial_step_hz,
                "virtual_spacing_m": self.virtual_spacing_m,
                "convention": "upstream range FFT grid and endpoint-inclusive spatial frequency grid",
            },
            "geometry_calibration": "physical Tx/Rx offsets, Tx timing and channel gain calibration are not provided in data_params.npz",
        }
