"""Complex backprojection and mmNorm's signed coherent normal estimator.

This module deliberately does not import the neural network pipeline. The default
geometry reproduces PanoRadar's array manifold; it is an effective phase-center
model, not a claim that physical Tx/Rx positions have been calibrated.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft

from .coordinates import panoramic_angles

C = 2.99792e8
VIRTUAL_SPACING_M = 1.9e-3


def radar_pixel_rays(azimuth_columns=512):
    """Fixed RF directions, including the duplicated panoramic endpoint."""
    azimuth, elevation = panoramic_angles(azimuth_columns)
    return np.stack(
        np.broadcast_arrays(
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ),
        axis=-1,
    )


def backend(name="auto"):
    if name not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be auto, cpu or cuda")
    if name != "cpu":
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount():
                return cp
        except (ImportError, RuntimeError):
            if name == "cuda":
                raise
        if name == "cuda":
            raise RuntimeError("No CUDA device is available")
    return np


def host(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)


@dataclass(frozen=True)
class Waveform:
    """Use both frequency grids actually used by PanoRadar.

    Range FFT bins use bandwidth / N, whereas spatial steering uses the
    endpoint-inclusive linspace in dataset.py. Keeping these separate avoids a
    range-scale error when migrating its FFT to near-field backprojection.
    A common, voxel-dependent carrier phase is omitted; mmNorm's weights are
    invariant to that phase.
    """

    samples: int = 256
    start_hz: float = 77e9
    bandwidth_hz: float = 4e9
    spatial_bandwidth_hz: float | None = None

    def __post_init__(self):
        values = [self.start_hz, self.bandwidth_hz]
        if self.spatial_bandwidth_hz is not None:
            values.append(self.spatial_bandwidth_hz)
        if self.samples < 2 or not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Invalid waveform")

    @property
    def range_step_hz(self):
        return self.bandwidth_hz / self.samples

    @property
    def spatial_step_hz(self):
        span = self.bandwidth_hz if self.spatial_bandwidth_hz is None else self.spatial_bandwidth_hz
        return span / (self.samples - 1)

    @property
    def frequencies_hz(self):
        span = self.bandwidth_hz if self.spatial_bandwidth_hz is None else self.spatial_bandwidth_hz
        return np.linspace(
            self.start_hz, self.start_hz + span, self.samples
        )

    def phase_cycles(self, path_m, origin_range_m):
        """Forward IF phase in cycles, matching PanoRadar's FFT/steering signs."""
        k = np.arange(self.samples)
        delta = (
            np.asarray(path_m)[..., None] - 2 * np.asarray(origin_range_m)[..., None]
        )
        return (
            self.frequencies_hz * delta
            + 2 * np.asarray(origin_range_m)[..., None] * k * self.range_step_hz
        ) / C


@dataclass
class Geometry:
    phase_centers_m: np.ndarray
    tx_m: np.ndarray | None = None
    rx_m: np.ndarray | None = None
    model: str = "panoradar_effective_phase_centers"

    @classmethod
    def rotating(cls, angles, radius_m, geometry_file=None, virtual_spacing_m=VIRTUAL_SPACING_M):
        angles = np.asarray(angles, dtype=np.float64)
        if angles.ndim != 1 or len(angles) < 3 or not np.isfinite(angles).all() or not np.isfinite([radius_m, virtual_spacing_m]).all() or min(radius_m, virtual_spacing_m) <= 0:
            raise ValueError("Invalid rotation geometry")
        origins = np.column_stack(
            (
                radius_m * np.cos(angles),
                radius_m * np.sin(angles),
                np.zeros_like(angles),
            )
        )
        if geometry_file is None:
            points = np.broadcast_to(origins[:, None, :], (len(angles), 8, 3)).copy()
            # Existing vertical compensation is exp(-i*2*pi*s_a*sin(elev)/lambda).
            # With positive-beat IF and a round-trip model this means p_z=-s_a/2.
            points[:, :, 2] = -virtual_spacing_m * np.arange(7, -1, -1) / 2
            return cls(points)
        data = json.loads(Path(geometry_file).read_text())
        tx = np.asarray(data["tx_offsets_m"], dtype=float)
        rx = np.asarray(data["rx_offsets_m"], dtype=float)
        if tx.shape != (8, 3) or rx.shape != (8, 3) or not np.isfinite([tx, rx]).all():
            raise ValueError(
                "Geometry JSON must contain finite (8,3) tx_offsets_m and rx_offsets_m"
            )

        def rotate(offsets):
            out = np.empty((len(angles), 8, 3))
            out[:, :, 0] = (
                np.cos(angles[:, None]) * offsets[None, :, 0]
                - np.sin(angles[:, None]) * offsets[None, :, 1]
            )
            out[:, :, 1] = (
                np.sin(angles[:, None]) * offsets[None, :, 0]
                + np.cos(angles[:, None]) * offsets[None, :, 1]
            )
            out[:, :, 2] = offsets[None, :, 2]
            return out + origins[:, None, :]

        tx, rx = rotate(tx), rotate(rx)
        return cls((tx + rx) / 2, tx, rx, "explicit_bistatic_tx_rx")


class MatchedResponse:
    """Oversampled range FFT with carrier-corrected complex interpolation.

    The FFT is a fast approximation to per-channel wideband matched filtering.
    No magnitude is taken before the signed coherent weighting stage.
    """

    def __init__(
        self,
        frame,
        angles,
        geometry,
        waveform=None,
        oversample=16,
        compute_backend="auto",
        reference_origin_m=None,
    ):
        self.xp = backend(compute_backend)
        xp = self.xp
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[1] != 8 or not np.iscomplexobj(frame):
            raise ValueError("Expected complex frame of shape (positions,8,samples)")
        self.waveform = waveform or Waveform(frame.shape[-1])
        if frame.shape[-1] != self.waveform.samples or not np.isfinite(frame).all():
            raise ValueError("Frame/waveform mismatch or nonfinite input")
        if geometry.phase_centers_m.shape != (len(frame), 8, 3) or len(angles) != len(
            frame
        ):
            raise ValueError("Frame and geometry dimensions differ")
        if int(oversample) != oversample or oversample < 2:
            raise ValueError("FFT oversampling must be an integer >= 2")
        self.angles = np.asarray(angles)
        self.geometry = geometry
        self.reference_origin = xp.asarray(
            np.zeros(3) if reference_origin_m is None else reference_origin_m
        )
        self.positions = xp.asarray(geometry.phase_centers_m)
        self.tx = None if geometry.tx_m is None else xp.asarray(geometry.tx_m)
        self.rx = None if geometry.rx_m is None else xp.asarray(geometry.rx_m)
        self.nfft = self.waveform.samples * int(oversample)
        window = xp.asarray(np.hanning(frame.shape[-1]), dtype=xp.float32)
        signal = xp.asarray(frame, dtype=xp.complex64) * window
        if xp is np:
            self.spectrum = (
                fft.fft(signal, n=self.nfft, axis=-1, workers=-1) / window.sum()
            )
        else:
            self.spectrum = xp.fft.fft(signal, n=self.nfft, axis=-1) / window.sum()

    def contributions(self, points, indices):
        xp = self.xp
        x = xp.asarray(points, dtype=xp.float64)[:, None, None, :]
        ids = xp.asarray(indices)
        if self.tx is None:
            ray = self.positions[ids] - x
            distance = xp.linalg.norm(ray, axis=-1)
            path = 2 * distance
            unit = ray / xp.maximum(distance[..., None], 1e-12)
        else:
            ray_tx, ray_rx = self.tx[ids] - x, self.rx[ids] - x
            d_tx, d_rx = (
                xp.linalg.norm(ray_tx, axis=-1),
                xp.linalg.norm(ray_rx, axis=-1),
            )
            path = d_tx + d_rx
            unit = ray_tx / xp.maximum(d_tx[..., None], 1e-12) + ray_rx / xp.maximum(
                d_rx[..., None], 1e-12
            )
            unit /= xp.maximum(xp.linalg.norm(unit, axis=-1)[..., None], 1e-12)
        radius = xp.linalg.norm(x[:, 0, 0] - self.reference_origin, axis=-1)[
            :, None, None
        ]
        delta = path - 2 * radius
        wave = self.waveform
        frequency = (2 * radius * wave.range_step_hz + delta * wave.spatial_step_hz) / C
        # Positive-frequency, unaliased range domain. Do not wrap negative or
        # beyond-maximum distances into apparently valid close-range returns.
        in_range = (frequency >= 0) & (frequency < 1 - 1 / self.nfft)
        coordinate = xp.clip(frequency * self.nfft, 0, self.nfft - 1.000001)
        lower = coordinate.astype(xp.int32)
        fraction = coordinate - lower
        channel = xp.arange(8)[None, None, :]
        v0 = self.spectrum[ids[:, :, None], channel, lower]
        v1 = self.spectrum[ids[:, :, None], channel, lower + 1]
        response = (v0 * (1 - fraction) + v1 * fraction) * xp.exp(
            -2j * np.pi * wave.start_hz * delta / C
        )
        return xp.where(in_range, response, 0), unit
