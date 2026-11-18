"""RF-only SelfData input, following SURadar's IMU framing conventions."""

import struct
import zipfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

from ..imaging.core import C, Waveform


def mapped_npz_array(path, key):
    """Map a stored numeric NPY member without extracting or loading the archive."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(key + ".npy")
        if member.compress_type != zipfile.ZIP_STORED or member.flag_bits & 1:
            raise ValueError(
                f"{path}:{key} must be an uncompressed, unencrypted NPY member"
            )
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"Unsupported NPY version {version} in {path}:{key}")
            header_size = stream.tell()
        if dtype.hasobject or not shape or fortran:
            raise ValueError(f"{path}:{key} must be a numeric C-order array")
        size = int(np.prod(shape, dtype=object)) * dtype.itemsize
        if header_size + size != member.file_size:
            raise ValueError(f"Invalid array size in {path}:{key}")
        with path.open("rb") as stream:
            stream.seek(member.header_offset)
            header = stream.read(30)
            if len(header) != 30 or header[:4] != b"PK\x03\x04":
                raise ValueError(f"Invalid ZIP local header in {path}:{key}")
            name_size, extra_size = struct.unpack_from("<HH", header, 26)
        offset = member.header_offset + 30 + name_size + extra_size + header_size
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=offset)


class SelfData:
    def __init__(self, folder, platform_motion="moving"):
        folder = Path(folder)
        with np.load(folder / "data_params.npz", allow_pickle=False) as params:
            self.separators = params["radar_sep"].copy()
            self.radius = float(params["r_radar"])
            imu_sep = params["imu_sep"].copy() if platform_motion == "moving" else None
        config = yaml.safe_load((folder / "radar_config.yaml").read_text())
        if not isinstance(config, dict) or not isinstance(config.get("radar"), dict):
            raise TypeError("radar_config.yaml must contain a radar mapping")
        radar = config["radar"]
        if (radar["num_tx"], radar["num_rx"]) != (2, 4):
            raise ValueError("SelfData currently requires 2 Tx and 4 Rx")
        samples = radar["adc_samples"]
        if type(samples) is not int or samples < 2:
            raise ValueError("adc_samples must be an integer >= 2")
        start = float(radar["start_freq_ghz"]) * 1e9
        slope = float(radar["freq_slope_mhz_us"]) * 1e12
        rate = float(radar["sample_rate_ksps"]) * 1e3
        ramp = float(radar["ramp_end_time_us"]) * 1e-6
        if (
            not np.isfinite([start, slope, rate, ramp]).all()
            or min(start, slope, rate, ramp) <= 0
        ):
            raise ValueError("SelfData waveform values must be finite and positive")
        self.waveform = Waveform(samples, start, slope * samples / rate, slope * ramp)
        self.virtual_spacing_m = C / (2 * start)
        self.raw = mapped_npz_array(folder / "radar.npz", "frames")
        if self.raw.shape != (len(self.raw), 2, 4, samples):
            raise ValueError("SelfData RF shape differs from radar_config.yaml")
        with np.load(folder / "radar.npz", allow_pickle=False) as data:
            self.timestamps = data["timestamps"].copy()
        if (
            self.timestamps.shape != (len(self.raw),)
            or not np.isfinite(self.timestamps).all()
            or np.any(np.diff(self.timestamps) <= 0)
        ):
            raise ValueError("SelfData radar timestamps must be finite and increasing")
        sep = self.separators
        if (
            sep.ndim != 1
            or len(sep) < 2
            or not np.issubdtype(sep.dtype, np.integer)
            or sep[0] < 0
            or sep[-1] > len(self.raw)
            or np.any(np.diff(sep) < 3)
        ):
            raise ValueError("Invalid SelfData radar_sep")
        self.frame_ends = sep[1:].copy()
        self.framing = "native static radar_sep"
        if platform_motion == "moving":
            with np.load(folder / "imu.npz", allow_pickle=False) as data:
                timestamps = data["timestamps"]
                gyro = data["angular_velocity"]
            if (
                timestamps.ndim != 1
                or gyro.shape != (len(timestamps), 3)
                or not np.isfinite(timestamps).all()
                or np.any(np.diff(timestamps) <= 0)
                or not np.isfinite(gyro).all()
            ):
                raise ValueError("Invalid SelfData IMU timestamps or angular_velocity")
            if (
                imu_sep.shape != sep.shape
                or not np.issubdtype(imu_sep.dtype, np.integer)
                or imu_sep[0] < 0
                or imu_sep[-1] > len(gyro)
                or np.any(np.diff(imu_sep) < 1)
            ):
                raise ValueError("imu_sep must cover every SelfData rotation")
            filtered = gyro[:, 2].copy()
            jumps = np.flatnonzero(np.abs(np.r_[np.diff(filtered), 0]) > 2.0)
            jumps = np.clip(np.maximum(jumps, 20), 0, len(filtered) - 1)
            filtered[jumps] = filtered[np.maximum(jumps - 20, 0)]
            filtered = gaussian_filter(filtered, sigma=2.0, truncate=6)
            omega = np.array([filtered[a:b].mean() for a, b in pairwise(imu_sep)])
            counts = np.diff(sep)
            delta_t = (
                float(np.mean(np.diff(self.timestamps[sep[:-1]]) / counts[:-1]))
                if len(sep) > 2
                else float(np.mean(np.diff(self.timestamps)))
            )
            step = 2 * np.pi / counts - delta_t * omega
            if np.any(step <= 0) or not np.isfinite(step).all():
                raise ValueError("IMU framing implies nonpositive angular sampling")
            beams = np.clip(np.rint(2 * np.pi / step).astype(np.int64), 1, 2 * counts)
            self.frame_ends = np.minimum(sep[:-1] + beams, sep[-1])
            if np.any(self.frame_ends - sep[:-1] < 3):
                raise ValueError("IMU-framed SelfData rotation is incomplete")
            self.framing = "SURadar IMU-adjusted extent; nominal motor phase plus per-chirp platform yaw; disjoint voting cores"
