"""RF-only multi-peak surfaces with registered views and discontinuity costs.

No LiDAR input is accepted. View agreement is intensity support, not optical
photometric constancy or proof of a direct reflection. Inter-rotation phase is
never combined. Spatial min-sum messages use a truncated depth penalty, allowing
real discontinuities; no inferred depth outside the RF candidates is invented.
"""

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter

from .core import radar_pixel_rays

_MIN_PEAK_SNR_DB = 8.0
_FOREGROUND_COST_MARGIN_DB = 3.0
PREFERRED_RANGE_M = 4.5


def _range_window(coarse):
    """Keep the original +/- two native range cells after FFT zero padding."""
    ranges = np.asarray(coarse["range_m"], dtype=float)
    if ranges.ndim != 1 or len(ranges) < 2 or not np.isfinite(ranges).all():
        raise ValueError("RF range axis must contain at least two finite bins")
    step = ranges[1] - ranges[0]
    if step <= 0 or not np.allclose(np.diff(ranges), step):
        raise ValueError("RF range bins must be uniformly increasing")
    resolution = float(coarse.get("range_resolution_m", step))
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("RF range resolution must be finite and positive")
    half = max(1, int(np.ceil(2 * resolution / step - 1e-8)))
    return 2 * half + 1


def range_candidates(coarse, count, min_range, max_range, *, preferred_range=None):
    """Distinct local range maxima, bounded quadratic interpolation in log amplitude."""
    amplitude = np.asarray(coarse["amplitude"], np.float32)
    ranges = np.asarray(coarse["range_m"])
    if (
        amplitude.ndim != 3
        or amplitude.shape[1] != 64
        or amplitude.shape[2] < 3
        or len(ranges) != len(amplitude)
    ):
        raise ValueError("Multi-peak selection requires a (range,64,azimuth) RF volume")
    if count < 2 or count > len(ranges):
        raise ValueError("Invalid RF candidate count")
    allowed = (ranges >= min_range) & (ranges <= max_range)
    if not allowed.any():
        raise ValueError("No allowed RF range bins")
    a = np.maximum(amplitude, 0).copy()
    # The training grid includes both 0 and -2pi. Treat that as ONE graph node.
    a[:, :, 0] = (a[:, :, 0] + a[:, :, -1]) / 2
    a[:, :, -1] = a[:, :, 0]
    local = maximum_filter(a, size=(_range_window(coarse), 1, 1), mode="nearest")
    maxima = (a >= local) & (a > 0) & allowed[:, None, None]
    # Always retain the allowed global maximum, including a range-boundary peak.
    best = np.where(allowed[:, None, None], a, -np.inf).argmax(axis=0)
    row, col = np.indices(a.shape[1:])
    maxima[best, row, col] = a[best, row, col] > 0
    regions = (allowed,)
    if preferred_range is not None:
        near = ranges <= preferred_range
        regions = (allowed & near, allowed & ~near)
    picks = []
    for region in regions:
        bins = np.flatnonzero(region)
        if bins.size:
            section = slice(bins[0], bins[-1] + 1)
            score = np.where(maxima[section], a[section], -np.inf)
            picks.append(np.argsort(score, axis=0)[-count:][::-1] + section.start)
    ids = np.concatenate(picks, axis=0)
    valid = np.take_along_axis(maxima, ids, axis=0)
    peaks = np.take_along_axis(a, ids, axis=0)
    middle = np.log(np.maximum(peaks, 1e-20))
    before = np.take_along_axis(a, np.maximum(ids - 1, 0), axis=0)
    after = np.take_along_axis(a, np.minimum(ids + 1, len(ranges) - 1), axis=0)
    before = np.log(np.maximum(before, 1e-20))
    after = np.log(np.maximum(after, 1e-20))
    denominator = before - 2 * middle + after
    offset = np.divide(
        0.5 * (before - after),
        denominator,
        out=np.zeros_like(middle),
        where=np.abs(denominator) > 1e-8,
    )
    offset = np.where(
        (ids > 0) & (ids < len(ranges) - 1), np.clip(offset, -0.5, 0.5), 0
    )
    distance = np.clip(
        ranges[ids] + offset * (ranges[1] - ranges[0]),
        min_range,
        min(max_range, ranges[-1]),
    )
    strongest = np.maximum(np.where(valid, peaks, 0).max(axis=0), 1e-20)
    relative_db = 20 * np.log10(np.maximum(peaks / strongest, 1e-8))
    noise = np.median(a[allowed], axis=0)
    snr_db = 20 * np.log10(np.maximum(peaks / np.maximum(noise, 1e-20), 1e-8))
    return {
        "range_m": distance.transpose(1, 2, 0).astype(np.float32),
        "valid": valid.transpose(1, 2, 0),
        "relative_db": relative_db.transpose(1, 2, 0),
        "snr_db": snr_db.transpose(1, 2, 0),
        "profile": a,
    }


def _source_coordinates(source, ranges, min_range, max_range, azimuth_columns):
    radius = np.linalg.norm(source, axis=-1)
    phi = np.arcsin(np.clip(source[..., 2] / np.maximum(radius, 1e-12), -1, 1))
    theta = np.mod(-np.arctan2(source[..., 1], source[..., 0]), 2 * np.pi)
    coordinates = np.array(
        [
            (radius - ranges[0]) / (ranges[1] - ranges[0]),
            (np.pi / 4 - phi) * 63 / (np.pi / 2),
            theta * (azimuth_columns - 1) / (2 * np.pi),
        ]
    )
    valid = (
        (radius >= max(min_range, ranges[0]))
        & (radius <= min(max_range, ranges[-1]))
        & (np.abs(phi) <= np.pi / 4 + 1e-10)
    )
    return coordinates.reshape(3, -1), valid


def sample_registered_support(
    coarse, target_points, source_from_target, min_range, max_range
):
    """Query a source-frame RF volume at target-space points via a rigid pose.

    Allow two native range bins and about 0.7 degrees of azimuth tolerance for
    approximate RF poses and range sampling. Out-of-FoV and out-of-range queries
    are missing; elevation does not wrap across the poles.
    """
    transform = np.asarray(source_from_target)
    source = np.asarray(target_points) @ transform[:3, :3].T + transform[:3, 3]
    ranges = coarse["range_m"]
    amplitude = np.maximum(np.asarray(coarse["amplitude"], np.float32), 0)
    azimuth_half = max(1, round((amplitude.shape[2] - 1) / 511))
    # Wrap the unique azimuth cells before restoring the duplicated endpoint.
    smoothed = maximum_filter(
        amplitude[:, :, :-1],
        size=(_range_window(coarse), 1, 2 * azimuth_half + 1),
        mode=("nearest", "nearest", "wrap"),
    )
    smoothed = np.concatenate((smoothed, smoothed[:, :, :1]), axis=2)
    allowed = (ranges >= min_range) & (ranges <= max_range)
    normalizer = np.maximum(smoothed[allowed].max(axis=0), 1e-20)
    ratio = smoothed / normalizer[None]
    coordinates, valid = _source_coordinates(
        source, ranges, min_range, max_range, amplitude.shape[2]
    )
    sampled = map_coordinates(
        ratio, coordinates, order=1, mode="nearest", prefilter=False
    ).reshape(valid.shape)
    cost = np.clip(-20 * np.log10(np.maximum(sampled, 1e-8)), 0, 24)
    return cost.astype(np.float32), valid


def _add_registered_candidate(args, coarse, candidates, views, required_views, rays):
    """Pool corroborated RF power on the target lattice before multi-view peak picking."""
    resolution = coarse["range_resolution_m"]
    fft_grid = coarse["range_m"]
    stride = max(1, round(resolution / (fft_grid[1] - fft_grid[0])))
    ranges = fft_grid[::stride]
    ranges = ranges[(ranges >= args.min_range) & (ranges <= args.max_range)]
    if not len(ranges):
        return False
    rays = rays[:, :-1]
    score = np.zeros((len(ranges), *rays.shape[:2]), np.float32)
    count = np.zeros(score.shape, dtype=np.int32)
    relative_floor = 10 ** (-args.surface_support_db / 20)
    # A look contributes only if its response clears both the relative
    # amplitude threshold and the 8 dB RF noise-floor criterion.
    for image, transform in views():
        grid = image["range_m"]
        amplitude = image["amplitude"]
        roi = amplitude[(grid >= args.min_range) & (grid <= args.max_range)]
        scale = np.maximum(roi.max(axis=0), 1e-20)
        floor = np.maximum(
            relative_floor,
            np.median(roi, axis=0) / scale * 10 ** (_MIN_PEAK_SNR_DB / 20),
        )
        ratio = amplitude / scale
        rotated = rays @ transform[:3, :3].T
        for start in range(0, len(ranges), 16):
            sl = slice(start, start + 16)
            points = ranges[sl, None, None, None] * rotated + transform[:3, 3]
            coordinates, valid = _source_coordinates(
                points, grid, args.min_range, args.max_range, amplitude.shape[2]
            )
            observed = map_coordinates(
                ratio, coordinates, order=1, mode="nearest", prefilter=False
            ).reshape(valid.shape)
            local_floor = map_coordinates(
                floor, coordinates[1:], order=1, mode="nearest", prefilter=False
            ).reshape(valid.shape)
            supported = valid & (observed > 0) & (observed >= local_floor)
            count[sl] += supported
            score[sl] += np.where(supported, observed**2, 0)
        del ratio, roi, amplitude, image
    score[count < required_views] = 0
    del count
    maxima = score >= maximum_filter(score, size=(5, 1, 1), mode="nearest")
    keep_apart = max(2 * resolution, args.surface_refine_radius_m)
    for depth, valid in zip(
        candidates["range_m"][:, :-1].transpose(2, 0, 1),
        candidates["valid"][:, :-1].transpose(2, 0, 1),
    ):
        maxima &= ~((np.abs(ranges[:, None, None] - depth) <= keep_apart) & valid)
    maxima &= score > 0
    depths, supported = [], []
    near = ranges[:, None, None] <= PREFERRED_RANGE_M
    for region in (near, ~near):
        eligible = maxima & region
        indices = np.argmax(np.where(eligible, score, 0), axis=0)
        chosen = np.take_along_axis(eligible, indices[None], axis=0)[0]
        if chosen.any():
            depths.append(ranges[indices].astype(np.float32))
            supported.append(chosen)
    if not depths:
        return False
    for key, values in (("range_m", depths), ("valid", supported)):
        proposals = np.stack(values, axis=-1)
        proposals = np.concatenate((proposals, proposals[:, :1]), axis=1)
        candidates[key] = np.concatenate((candidates[key], proposals), axis=2)
    return True


def surface_graph(
    depths, profile, spatial_weight, *, candidate_support=None, reference_labels=None
):
    """Cylindrical graph with lower jump costs for RF-supported continuations."""
    h, w, k = depths.shape
    nodes = np.arange(h * w).reshape(h, w)
    u = np.r_[nodes.ravel(), nodes[:-1].ravel()]
    v = np.r_[np.roll(nodes, -1, axis=1).ravel(), nodes[1:].ravel()]
    profiles = profile.reshape(len(profile), -1).T.copy()
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-20)
    similarity = np.einsum("ij,ij->i", profiles[u], profiles[v])
    edge_weight = spatial_weight / 4 * np.exp(-np.maximum(1 - similarity, 0) / 0.3)
    rays = radar_pixel_rays(w + 1)[:, :-1].reshape(-1, 3)
    separation = np.sqrt(np.maximum(2 - 2 * np.einsum("ij,ij->i", rays[u], rays[v]), 0))
    d = depths.reshape(-1, k)
    tolerance = 0.10 + 2 * separation[:, None, None] * np.minimum(
        d[u, :, None], d[v, None, :]
    )
    jumps = np.abs(d[u, :, None] - d[v, None, :]) / tolerance
    penalty = np.minimum(jumps, 1)
    if candidate_support is not None:
        supported = candidate_support.reshape(-1, k)
        trusted = supported[np.arange(len(d)), reference_labels]
        links = np.arange(len(u))
        matches_u = (
            (jumps[links, :, reference_labels[v]] < 1) & supported[u] & trusted[v, None]
        )
        matches_v = (
            (jumps[links, reference_labels[u], :] < 1) & supported[v] & trusted[u, None]
        )
        continuations = np.zeros_like(supported, dtype=np.int8)
        np.add.at(continuations, u, matches_u)
        np.add.at(continuations, v, matches_v)
        # Each branch must continue through a neighbor other than the opposite endpoint.
        same_side_u = continuations[u] > matches_u
        same_side_v = continuations[v] > matches_v
        edge = (jumps > 1) & same_side_u[:, :, None] & same_side_v[:, None, :]
        penalty[edge] = 1 / jumps[edge]
    pair = edge_weight[:, None, None] * penalty
    return u, v, pair.astype(np.float32)


def optimize_candidates(
    unary,
    depths,
    profile,
    spatial_weight=4.0,
    iterations=12,
    *,
    candidate_support=None,
    admissible=None,
):
    """Damped min-sum BP, returning the lowest actual energy encountered."""
    if unary.shape != depths.shape or not np.isfinite(unary).all():
        raise ValueError("Finite unary costs must match candidate depths")
    h, w, k = unary.shape
    if h != 64 or w < 2:
        raise ValueError("Expected a cylindrical grid with 64 elevation rows")
    data = unary.reshape(-1, k)
    if admissible is not None:
        choices = admissible.reshape(-1, k)
        data = np.where(choices | ~choices.any(axis=1, keepdims=True), data, np.inf)
    best = data.argmin(axis=1)
    u, v, pair = surface_graph(
        depths,
        profile,
        spatial_weight,
        candidate_support=candidate_support,
        reference_labels=best,
    )
    forward = np.zeros((len(u), k), np.float32)
    backward = np.zeros_like(forward)

    def energy(labels):
        return float(
            data[np.arange(len(data)), labels].sum(dtype=np.float64)
            + pair[np.arange(len(u)), labels[u], labels[v]].sum(dtype=np.float64)
        )

    initial_energy = best_energy = energy(best)
    belief = data.copy()
    for _ in range(iterations):
        belief = data.copy()
        np.add.at(belief, v, forward)
        np.add.at(belief, u, backward)
        new_forward = ((belief[u] - backward)[:, :, None] + pair).min(axis=1)
        new_backward = ((belief[v] - forward)[:, None, :] + pair).min(axis=2)
        new_forward -= new_forward.min(axis=1, keepdims=True)
        new_backward -= new_backward.min(axis=1, keepdims=True)
        forward = 0.5 * (forward + new_forward)
        backward = 0.5 * (backward + new_backward)
        current = belief.argmin(axis=1)
        value = energy(current)
        if value < best_energy:
            best, best_energy = current.copy(), value
    return best.reshape(h, w), {
        "initial_energy": initial_energy,
        "final_energy": best_energy,
        "iterations": iterations,
        "optimizer": "damped min-sum BP; best encountered energy",
    }


def select_motion_surface(
    args, trajectory, target, frames, coarse, image_function, refine_candidates=None
):
    """Combine target peaks with independently corroborated multi-view candidates."""
    candidates = range_candidates(
        coarse,
        args.surface_candidates,
        args.min_range,
        args.max_range,
        preferred_range=PREFERRED_RANGE_M,
    )
    target_count = candidates["range_m"].shape[-1]
    target_peaks = slice(target_count)
    registered_peaks = slice(target_count, None)
    refinement = (
        refine_candidates(candidates) if refine_candidates is not None else None
    )
    rays = radar_pixel_rays(coarse["amplitude"].shape[2])
    target_pose = trajectory.poses.at(
        trajectory.timestamps[trajectory.separators[target]]
    )[0]
    transforms = {}
    for number in frames:
        pose = trajectory.poses.at(trajectory.timestamps[trajectory.separators[number]])[0]
        transforms[number] = (
            np.eye(4) if number == target else np.linalg.solve(pose, target_pose)
        )

    def views():
        for number, transform in transforms.items():
            if number == target:
                yield coarse, transform
                continue
            frame, geometry, facing, origins, _ = trajectory.registered_frame(
                number, number, args.aperture, args.geometry_json
            )
            source_coarse = image_function(
                frame, geometry, origins, facing, args.aperture, args.backend
            )
            del frame
            yield source_coarse, transform
            del source_coarse

    required_views = min(len(frames), max(2, args.surface_min_views))
    registered = len(frames) > 1 and _add_registered_candidate(
        args, coarse, candidates, views, required_views, rays
    )
    points = rays[..., None, :] * candidates["range_m"][..., None]
    costs, observations = [], []
    for source_coarse, transform in views():
        cost, visible = sample_registered_support(
            source_coarse,
            points,
            transform,
            args.min_range,
            args.max_range,
        )
        del source_coarse
        costs.append(cost)
        observations.append(visible)
    costs, observed = np.stack(costs), np.stack(observations)
    count = observed.sum(axis=0)
    temporal = np.divide(
        (costs * observed).sum(axis=0),
        count,
        out=np.full(points.shape[:-1], 24.0, np.float32),
        where=count > 0,
    )
    target_cost = np.clip(-candidates["relative_db"], 0, 24)
    support = ((costs <= args.surface_support_db) & observed).sum(axis=0)
    if registered:
        candidates["valid"][..., registered_peaks] &= (
            support[..., registered_peaks] >= required_views
        )
        target_cost = np.concatenate(
            (target_cost, costs[frames.index(target), ..., registered_peaks]), axis=-1
        )
    unary = (
        (1 - args.surface_temporal_weight) * target_cost
        + args.surface_temporal_weight * temporal
    ).astype(np.float32)
    depths = candidates["range_m"]
    rf_supported = candidates["valid"] & (support >= required_views)
    rf_supported[..., target_peaks] &= (
        (candidates["snr_db"] >= _MIN_PEAK_SNR_DB)
        & (target_cost[..., target_peaks] <= args.surface_support_db)
    )
    near = rf_supported & (depths <= PREFERRED_RANGE_M)
    prefer_near = near.any(axis=-1, keepdims=True)
    has_support = rf_supported.any(axis=-1, keepdims=True)
    admissible = np.where(has_support, rf_supported, candidates["valid"])
    admissible &= ~prefer_near | near
    if len(frames) > 1:
        best_cost = np.where(admissible, unary, np.inf).min(axis=-1, keepdims=True)
        foreground = (
            admissible[..., target_peaks]
            & rf_supported[..., target_peaks]
            & (unary[..., target_peaks] <= best_cost + _FOREGROUND_COST_MARGIN_DB)
        )
        front_range = np.min(
            np.where(foreground, depths[..., target_peaks], np.inf), axis=-1
        )
        separation = max(2 * coarse["range_resolution_m"], args.surface_refine_radius_m)
        admissible = admissible & (depths <= front_range[..., None] + separation)
    unary[~admissible] = 1e4
    candidate_support = rf_supported & admissible
    labels, _ = optimize_candidates(
        unary[:, :-1],
        depths[:, :-1],
        candidates["profile"][:, :, :-1],
        args.surface_spatial_weight,
        args.surface_bp_iterations,
        candidate_support=candidate_support[:, :-1],
        admissible=admissible[:, :-1],
    )
    labels = np.concatenate((labels, labels[:, :1]), axis=1)
    selected = np.take_along_axis(depths, labels[..., None], axis=-1)[..., 0]
    surface_valid = np.take_along_axis(candidate_support, labels[..., None], axis=-1)[
        ..., 0
    ]
    chosen_points = rays * selected[..., None]
    fields = {
        "surface_valid": surface_valid,
        "range_upper_m": np.where(
            prefer_near[..., 0], min(args.max_range, PREFERRED_RANGE_M), args.max_range
        ),
    }
    metadata = {
        "method": "registered_power_candidates_edge_aware_bp",
        "source_frames": frames,
        "candidates": args.surface_candidates,
        "preferred_range_max_m": PREFERRED_RANGE_M,
        "supported_near_pixels": int(prefer_near[:, :-1].sum()),
        "supported_far_fallback_pixels": int(
            ((selected > PREFERRED_RANGE_M) & surface_valid)[:, :-1].sum()
        ),
        "registered_proposals_min_views": required_views if len(frames) > 1 else None,
        "registered_proposal_pixels": (
            int(candidates["valid"][:, :-1, registered_peaks].any(axis=-1).sum())
            if registered
            else 0
        ),
        "registered_selected_pixels": (
            int((labels[:, :-1] >= target_count).sum()) if registered else 0
        ),
        "temporal_weight": args.surface_temporal_weight,
        "spatial_weight": args.surface_spatial_weight,
        "supported_surface_fraction": float(surface_valid.mean()),
        "prediction_uses_lidar": False,
        "missing_surface_policy": "retain best hypothesis for full-mask evaluation; surface_valid marks unsupported surface; never filter primary score",
        "limitations": "RF amplitude support, not verified visibility; no dynamic-object or occlusion inference; approximate poses",
    }
    if refinement is not None:
        metadata["near_field_refinement"] = refinement
    return chosen_points, fields, metadata
