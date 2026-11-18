"""Motion-compensated RF beamforming and bounded near-field range refinement."""

import numpy as np

from ..motion.aperture import motion_aperture_indices
from .core import C, MatchedResponse, Waveform, backend, host, radar_pixel_rays


def panoradar_peak_surface(coarse, oversample, min_range, max_range):
    """Figure 9 baseline: native-FFT peaks of Eq. 11 beamforming, without ML."""
    amplitude = coarse["amplitude"][::oversample]
    ranges = coarse["range_m"][::oversample]
    allowed = (ranges >= min_range) & (ranges <= max_range)
    if not allowed.any():
        raise ValueError("No native PanoRadar range bins inside the configured range")
    selected = np.where(allowed[:, None, None], amplitude, -np.inf).argmax(axis=0)
    depth = ranges[selected].astype(np.float32)
    reflection = np.take_along_axis(amplitude, selected[None], axis=0)[0].copy()
    depth[:, -1], reflection[:, -1] = depth[:, 0], reflection[:, 0]
    return {
        "panoradar_range_m": depth,
        "panoradar_points_m": (radar_pixel_rays() * depth[..., None]).astype(
            np.float32
        ),
        "panoradar_reflection": reflection,
        "panoradar_defined": reflection > 1e-20,
    }


def motion_coarse_image(
    frame,
    geometry,
    origins,
    facing,
    aperture,
    compute_backend,
    range_oversample=1,
    waveform=None,
):
    """Far-field beamforming with actual moving antenna positions.

    This is PanoRadar's exp(+4j*pi*p.dot(ray)/lambda) spatial compensation,
    including translation and rotation. Output rays have a fixed target origin,
    so there is no rolling panorama to register by equal pixel indices.
    """
    xp = backend(compute_backend)
    wave = waveform or Waveform(frame.shape[-1])
    if type(range_oversample) is not int or range_oversample < 1:
        raise ValueError("range_oversample must be a positive integer")
    nfft = wave.samples * range_oversample
    signal = xp.asarray(frame)
    positions = xp.asarray(geometry.phase_centers_m)
    freq = xp.asarray(wave.frequencies_hz)
    rays = radar_pixel_rays()
    # Far-field selection depends only on azimuth; use a distant point to
    # suppress finite-range parallax in this coarse proposal stage.
    ids, mask = motion_aperture_indices(rays[32] * 1e6, origins, facing, aperture)
    # A panorama seam can split the selected timestamps into two intervals.
    # Center the taper on facing ANGLE, not the middle of sorted sample IDs.
    # For a static uniform aperture this is exactly the original Hann taper.
    angle = np.arctan2(facing[ids, 1], facing[ids, 0])
    target_angle = np.arctan2(rays[32, :, 1], rays[32, :, 0])[:, None]
    delta = np.angle(np.exp(1j * (angle - target_angle)))
    span = np.maximum(np.abs(delta).max(axis=1, keepdims=True), 1e-12)
    taper = (0.5 * (1 + np.cos(np.pi * delta / span))) * mask
    windows = xp.asarray(taper[..., None])
    fast = xp.asarray(np.hanning(wave.samples))
    result = np.empty((nfft, 64, 512), np.float32)
    # Planar yaw/translation permits the same separable vertical beamforming as
    # the original code. General SE(3) poses retain all per-channel positions.
    relative = geometry.phase_centers_m - geometry.phase_centers_m[:, :1]
    separable = np.allclose(relative, relative[:1], atol=1e-10) and np.allclose(
        relative[..., :2], 0, atol=1e-10
    )
    for elevation in range(64):
        if separable:
            phase = (
                4j
                * np.pi
                / C
                * xp.asarray(relative[0, :, 2, None])
                * rays[elevation, 0, 2]
                * freq
            )
            vertical = (signal * xp.exp(phase)[None]).mean(axis=1)
        for start in range(0, 512, 16):
            selected = xp.asarray(ids[start : start + 16])
            look = xp.asarray(rays[elevation, start : start + 16])
            if separable:
                projection = xp.einsum("baj,bj->ba", positions[selected, 0], look)
                steering = xp.exp(4j * np.pi / C * projection[..., None] * freq)
                focused = vertical[selected] * steering
            else:
                projection = xp.einsum("bacj,bj->bac", positions[selected], look)
                focused = (
                    signal[selected]
                    * xp.exp(4j * np.pi / C * projection[..., None] * freq)
                ).mean(axis=2)
            window = windows[start : start + 16]
            focused = (focused * window).sum(axis=1) * fast
            amplitude = xp.abs(xp.fft.fft(focused, n=nfft, axis=-1)) / (
                xp.maximum(window.sum(axis=1), 1e-12) * fast.sum()
            )
            result[:, elevation, start : start + len(selected)] = host(amplitude).T
    return {
        "amplitude": result,
        "range_m": C / (2 * wave.bandwidth_hz * range_oversample) * np.arange(nfft),
        "range_resolution_m": C / (2 * wave.bandwidth_hz),
    }


def _refine_range_candidates(
    args,
    candidates,
    frame,
    geometry,
    origins,
    facing,
    upper,
    sampler=None,
    aperture_selector=None,
    waveform=None,
):
    """Bounded near-field matched-power search before discrete surface selection.

    Reuses the mmNorm/PanoRadar matched response, not its device-specific kernel:
    https://github.com/signalkinetics/mmNorm/blob/main/src/data_processing/cuda/imaging_gpu.cu
    One fixed real aperture per hypothesis keeps the radial objective continuous.
    """
    if sampler is None:
        sampler = MatchedResponse(
            frame,
            np.arctan2(facing[:, 1], facing[:, 0]),
            geometry,
            waveform=waveform,
            oversample=args.oversample,
            compute_backend=args.backend,
        )
    if aperture_selector is None:
        aperture_selector = motion_aperture_indices
    xp = sampler.xp
    depths = candidates["range_m"][:, :-1].copy()
    valid = candidates["valid"][:, :-1]
    pixels, peaks = np.nonzero(valid.reshape(-1, valid.shape[-1]))
    rays = radar_pixel_rays()[:, :-1].reshape(-1, 3)[pixels]
    centers = depths.reshape(-1, depths.shape[-1])[pixels, peaks].astype(float)
    amplitude = np.zeros_like(depths, dtype=float).reshape(-1, depths.shape[-1])
    offsets = np.linspace(
        -args.surface_refine_radius_m,
        args.surface_refine_radius_m,
        args.surface_refine_steps,
    )
    boundary_count = 0
    displacement_sum = 0.0
    for start in range(0, len(centers), args.batch_size):
        sl = slice(start, start + args.batch_size)
        direction, center = rays[sl], centers[sl]
        ids, _ = aperture_selector(
            direction * center[:, None], origins, facing, args.aperture
        )
        positions = np.asarray(origins)[ids]
        look = np.asarray(facing)[ids]

        def power(
            distance, direction=direction, positions=positions, look=look, ids=ids
        ):
            points = direction * distance[:, None]
            ray = points[:, None] - positions
            cosine = np.einsum("baj,baj->ba", ray, look) / np.maximum(
                np.linalg.norm(ray, axis=-1), 1e-12
            )
            mask = cosine >= np.cos(np.pi / 4)
            response, _ = sampler.contributions(points, ids)
            response *= xp.asarray(mask)[..., None]
            count = xp.asarray(np.maximum(mask.sum(axis=1) * 8, 1))
            return host((xp.abs(response.sum(axis=(1, 2))) / count) ** 2)

        distances = np.clip(center[:, None] + offsets, args.min_range, upper)
        values = np.column_stack([power(distances[:, i]) for i in range(len(offsets))])
        best = values.argmax(axis=1)
        row = np.arange(len(center))
        chosen = distances[row, best]
        chosen_power = values[row, best]
        before, after = np.maximum(best - 1, 0), np.minimum(best + 1, len(offsets) - 1)
        log_power = np.log(np.maximum(values, np.finfo(float).tiny))
        left, middle, right = (
            log_power[row, before],
            log_power[row, best],
            log_power[row, after],
        )
        curvature = left - 2 * middle + right
        interior = (best > 0) & (best < len(offsets) - 1)
        step = offsets[1] - offsets[0]
        interior &= np.isclose(distances[row, best] - distances[row, before], step)
        interior &= np.isclose(distances[row, after] - distances[row, best], step)
        delta = np.divide(
            0.5 * (left - right),
            curvature,
            out=np.zeros_like(middle),
            where=interior & (curvature < -1e-8),
        )
        vertex = np.clip(
            chosen + np.clip(delta, -0.5, 0.5) * step, args.min_range, upper
        )
        vertex_power = power(vertex)
        improved = vertex_power > chosen_power
        chosen = np.where(improved, vertex, chosen)
        chosen_power = np.maximum(vertex_power, chosen_power)
        # Zero/missing RF leaves the original continuous hypothesis untouched.
        chosen = np.where(chosen_power > 1e-20, chosen, center)
        depths.reshape(-1, depths.shape[-1])[pixels[sl], peaks[sl]] = chosen
        amplitude[pixels[sl], peaks[sl]] = np.sqrt(chosen_power)
        boundary_count += int(((best == 0) | (best == len(offsets) - 1)).sum())
        displacement_sum += float(np.abs(chosen - center).sum())
    amplitude = amplitude.reshape(depths.shape)
    maximum = np.maximum(amplitude.max(axis=-1, keepdims=True), 1e-20)
    relative = 20 * np.log10(np.maximum(amplitude / maximum, 1e-8))
    candidates["range_m"] = np.concatenate((depths, depths[:, :1]), axis=1)
    candidates["relative_db"] = np.concatenate((relative, relative[:, :1]), axis=1)
    return {
        "method": "target-rotation near-field matched power; bounded radial grid and checked log-parabolic vertex",
        "source_frames": "target rotation with actual neighbor seam samples",
        "search_radius_m": args.surface_refine_radius_m,
        "search_steps": args.surface_refine_steps,
        "refined_candidates": len(centers),
        "boundary_peak_candidates": boundary_count,
        "mean_absolute_displacement_m": displacement_sum / max(len(centers), 1),
    }
