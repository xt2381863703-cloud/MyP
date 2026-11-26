"""RF matching and piecewise surface estimation on a product manifold.

The state contains positive ranges and local unit plane normals/positive offsets.
Per-chirp geometry uses the supplied trajectory throughout reconstruction.
Complex range matching follows imaging.core.MatchedResponse. Rotations are
coherent internally; different rotations contribute separate power objectives.
"""

import time
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..motion.temporal import finish_statistics, look_statistics, temporal_frames
from ..normals.pixels import pack_normal_image
from .backprojection import (
    _refine_range_candidates,
    motion_coarse_image,
    panoradar_peak_surface,
)
from .core import C, backend, radar_pixel_rays
from .selection import select_motion_surface


def _to_tensor(value, device):
    # DLPack also supports the installed Torch 2 / NumPy 2 combination.
    array = np.ascontiguousarray(value)
    boolean = array.dtype == bool
    if boolean:
        array = array.astype(np.uint8)
    result = torch.from_dlpack(array).to(device)
    return result.bool() if boolean else result


def _to_numpy(value):
    if value.dtype == torch.bool:
        return np.from_dlpack(value.detach().to(torch.uint8).cpu()).astype(bool)
    return np.from_dlpack(value.detach().cpu()).copy()


def _initialize_planar_patches(ranges, rays, supported, shape):
    """Local connected patches; depth discontinuities split a chart."""
    h, w = ranges.shape
    rows, cols = np.indices((h, w))
    block = (rows // shape[0]) * int(np.ceil(w / shape[1])) + cols // shape[1]
    nodes = np.arange(h * w).reshape(h, w)
    u = np.r_[nodes[:, :-1].ravel(), nodes[:-1].ravel()]
    v = np.r_[nodes[:, 1:].ravel(), nodes[1:].ravel()]
    ray = rays.reshape(-1, 3)
    depth = ranges.ravel()
    separation = np.linalg.norm(ray[u] - ray[v], axis=-1)
    connect = (
        (block.ravel()[u] == block.ravel()[v])
        & (
            np.abs(depth[u] - depth[v])
            < 0.15 + 2 * separation * np.minimum(depth[u], depth[v])
        )
        & supported.ravel()[u]
        & supported.ravel()[v]
    )
    graph = coo_matrix(
        (np.ones(connect.sum()), (u[connect], v[connect])), shape=(h * w, h * w)
    ).tocsr()
    count, labels = connected_components(graph, directed=False)
    points = depth[:, None] * ray
    size = np.bincount(labels, minlength=count)
    centers = (
        np.stack(
            [
                np.bincount(labels, weights=points[:, j], minlength=count)
                for j in range(3)
            ],
            -1,
        )
        / size[:, None]
    )
    delta = points - centers[labels]
    covariance = (
        np.stack(
            [
                np.bincount(labels, weights=delta[:, j] * delta[:, k], minlength=count)
                for j in range(3)
                for k in range(3)
            ],
            -1,
        ).reshape(count, 3, 3)
        / size[:, None, None]
    )
    values, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    frontal = centers / np.maximum(
        np.linalg.norm(centers, axis=-1, keepdims=True), 1e-12
    )
    normals *= np.where((normals * frontal).sum(-1) < 0, -1, 1)[:, None]
    degenerate = (
        (size < 6) | (values[:, 1] < 1e-8) | ((normals * centers).sum(-1) < 0.02)
    )
    normals[degenerate] = frontal[degenerate]
    offsets = (normals * centers).sum(-1)
    axis = np.tile([0.0, 0.0, 1.0], (count, 1))
    axis[np.abs(normals[:, 2]) > 0.9] = [0.0, 1.0, 0.0]
    tangent = np.cross(normals, axis)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    basis = np.stack((tangent, np.cross(normals, tangent)), -1)
    return labels, normals, offsets, basis


class _RadialSurface(torch.nn.Module):
    def __init__(self, initial, supported, range_upper_m, args, device):
        super().__init__()
        rays = radar_pixel_rays(initial.shape[1])[:, :-1]
        ranges = np.linalg.norm(initial[:, :-1], axis=-1)
        labels, normals, offsets, tangent = _initialize_planar_patches(
            ranges, rays, supported[:, :-1], args.patch_shape
        )
        self.rays = _to_tensor(rays.reshape(-1, 3), device)
        self.initial_range = _to_tensor(ranges.ravel(), device)
        self.supported = _to_tensor(supported[:, :-1].ravel(), device)
        self.labels = _to_tensor(labels, device)
        self.normals = _to_tensor(normals, device)
        self.offsets = _to_tensor(offsets, device)
        self.tangent = _to_tensor(tangent, device)
        upper = _to_tensor(range_upper_m[:, :-1].ravel(), device)
        self.range_scale = torch.minimum(
            torch.log1p(args.range_bound_m / self.initial_range),
            torch.minimum(
                torch.log(self.initial_range / args.min_range).clamp_min(0),
                torch.log(upper / self.initial_range).clamp_min(0),
            ),
        )
        self.offset_scale = torch.log1p(args.range_bound_m / self.offsets)
        self.range_chart = torch.nn.Parameter(torch.zeros_like(self.initial_range))
        self.normal_chart = torch.nn.Parameter(
            torch.zeros((len(normals), 2), device=device, dtype=torch.float64)
        )
        self.offset_chart = torch.nn.Parameter(torch.zeros_like(self.offsets))
        self.args = args

    def ranges(self, selected=slice(None)):
        return self.initial_range[selected] * torch.exp(
            self.range_scale[selected] * torch.tanh(self.range_chart[selected])
        )

    def points(self, selected=slice(None)):
        return self.ranges(selected)[:, None] * self.rays[selected]

    def planes(self):
        delta = 0.35 * torch.tanh(self.normal_chart)
        normal = self.normals + torch.einsum("nij,nj->ni", self.tangent, delta)
        normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        offset = self.offsets * torch.exp(
            self.offset_scale * torch.tanh(self.offset_chart)
        )
        return normal, offset

    def regularizer(self):
        normal, offset = self.planes()
        residual = (
            (normal[self.labels] * self.points()).sum(-1) - offset[self.labels]
        ) / 0.05
        weight = self.supported.to(residual.dtype)
        plane = (
            (torch.sqrt(1 + residual.square()) - 1) * weight
        ).sum() / weight.sum().clamp_min(1)
        range_prior = torch.tanh(self.range_chart).square().mean()
        chart_prior = (
            torch.tanh(self.normal_chart).square().mean()
            + torch.tanh(self.offset_chart).square().mean()
        )
        total = (
            self.args.geometry_weight * plane
            + self.args.range_prior_weight * range_prior
            + 0.001 * chart_prior
        )
        return total, plane


class _RegisteredRadarLook:
    def __init__(
        self,
        trajectory,
        number,
        registered_frame,
        args,
        device,
        xp,
        *,
        include_padding=False,
    ):
        frame, geometry, facing, origins, record = registered_frame
        raw_lo = record["registered_raw_indices"][0]
        if include_padding:
            start, end = record["registered_raw_indices"]
        else:
            start, end = trajectory.core_bounds(number)
        take = slice(start - raw_lo, end - raw_lo)
        frame = frame[take]
        self.wave = trajectory.waveform
        self.positions = _to_tensor(geometry.phase_centers_m[take], device)
        self.tx = (
            None if geometry.tx_m is None else _to_tensor(geometry.tx_m[take], device)
        )
        self.rx = (
            None if geometry.rx_m is None else _to_tensor(geometry.rx_m[take], device)
        )
        self.facing = _to_tensor(facing[take], device)
        self.origins = _to_tensor(origins[take], device)
        self.timestamps = trajectory.timestamps[start:end]
        self.record = dict(record, registered_raw_indices=[start, end])
        self.nfft = frame.shape[-1] * args.oversample
        window = torch.hann_window(frame.shape[-1], periodic=False, device=device)
        signal = _to_tensor(frame, device)
        self.spectrum = (
            torch.fft.fft(signal * window, n=self.nfft, dim=-1) / window.sum()
        )
        self.device = device
        self.xp = xp

    def _aperture(self, points, count):
        ray = points[:, None] - self.origins[None]
        cosine = (ray * self.facing).sum(-1) / torch.linalg.vector_norm(
            ray, dim=-1
        ).clamp_min(1e-12)
        indices = torch.topk(cosine, count, dim=-1).indices.sort(-1).values
        return indices, cosine.gather(1, indices) >= np.cos(np.pi / 4)

    @torch.no_grad()
    def select_aperture(self, points, _origins, _facing, count):
        indices, mask = self._aperture(_to_tensor(points, self.device), count)
        return _to_numpy(indices), _to_numpy(mask)

    @torch.no_grad()
    def prepare_apertures(self, points, count, batch_size):
        index_batches, mask_batches = [], []
        for start in range(0, len(points), batch_size):
            indices, mask = self._aperture(points[start : start + batch_size], count)
            index_batches.append(indices)
            mask_batches.append(mask)
        self.indices = torch.cat(index_batches)
        self.mask = torch.cat(mask_batches)

    def response(self, points, selected, split=None):
        ids, mask = self.indices[selected], self.mask[selected]
        if split is not None:
            ids, mask = ids[:, split::2], mask[:, split::2]
        return self.sample(points, ids, mask)

    def sample(self, points, ids, mask):
        point = points[:, None, None]
        if self.tx is None:
            path = 2 * torch.linalg.vector_norm(self.positions[ids] - point, dim=-1)
        else:
            path = torch.linalg.vector_norm(
                self.tx[ids] - point, dim=-1
            ) + torch.linalg.vector_norm(self.rx[ids] - point, dim=-1)
        radius = torch.linalg.vector_norm(point[:, 0, 0], dim=-1)[:, None, None]
        delta = path - 2 * radius
        frequency = (
            2 * radius * self.wave.range_step_hz + delta * self.wave.spatial_step_hz
        ) / C
        valid = (frequency >= 0) & (frequency < 1 - 1 / self.nfft) & mask[..., None]
        coordinate = (frequency * self.nfft).clamp(0, self.nfft - 1.000001)
        lower = coordinate.to(torch.int64)
        fraction = coordinate - lower
        channels = torch.arange(8, device=self.device)[None, None]
        a = self.spectrum[ids[..., None], channels, lower]
        b = self.spectrum[ids[..., None], channels, lower + 1]
        phase = (-2 * np.pi * self.wave.start_hz / C * delta).to(torch.float32)
        response = (a * (1 - fraction) + b * fraction) * torch.polar(
            torch.ones_like(phase), phase
        )
        return torch.where(valid, response, 0), mask.sum(-1) * 8

    @torch.no_grad()
    def contributions(self, points, indices):
        ids = _to_tensor(indices, self.device)
        response, _ = self.sample(
            _to_tensor(points, self.device),
            ids,
            torch.ones_like(ids, dtype=torch.bool),
        )
        value = _to_numpy(response) if self.xp is np else self.xp.from_dlpack(response)
        return value, None

    def power(self, scene, selected, split=None):
        response, count = self.response(scene.points(selected), selected, split)
        return (response.sum((1, 2)).abs() / count.clamp_min(1)).square()


@torch.no_grad()
def _measure_surface(scene, looks, batch_size):
    points = scene.points()
    power, coherence, beta, training, acceptance = [], [], [], [], []
    for look in looks:
        p, c, b, train, accept = [], [], [], [], []
        for start in range(0, len(scene.rays), batch_size):
            ids = torch.arange(
                start,
                min(start + batch_size, len(scene.rays)),
                device=scene.rays.device,
            )
            response, count = look.response(points[ids], ids)
            summed = response.sum((1, 2))
            p.append((summed.abs() / count.clamp_min(1)).square())
            c.append(summed.abs() / response.abs().sum((1, 2)).clamp_min(1e-20))
            b.append(summed / count.clamp_min(1))
            for split, collection in ((0, train), (1, accept)):
                split_count = look.mask[ids, split::2].sum(-1) * 8
                collection.append(
                    (
                        response[:, split::2].sum((1, 2)).abs()
                        / split_count.clamp_min(1)
                    ).square()
                )
        power.append(torch.cat(p))
        coherence.append(torch.cat(c))
        beta.append(torch.cat(b))
        training.append(torch.cat(train))
        acceptance.append(torch.cat(accept))
    return tuple(
        torch.stack(values) for values in (power, coherence, beta, training, acceptance)
    )


@torch.no_grad()
def _final_normals(scene, looks, points, args):
    """Vote at finalized points with the same per-chirp matching geometry."""
    count = len(scene.rays)
    targets = scene.points()
    stats = {}
    for look in looks:
        for start in range(0, count, args.batch_size):
            selected = torch.arange(
                start, min(start + args.batch_size, count), device=scene.rays.device
            )
            ids, mask = look.indices[selected], look.mask[selected]
            target = targets[selected]
            response, _ = look.sample(target, ids, mask)
            if look.tx is None:
                ray = look.positions[ids] - target[:, None, None]
                directions = ray / torch.linalg.vector_norm(
                    ray, dim=-1, keepdim=True
                ).clamp_min(1e-12)
            else:
                a = look.tx[ids] - target[:, None, None]
                b = look.rx[ids] - target[:, None, None]
                directions = a / torch.linalg.vector_norm(
                    a, dim=-1, keepdim=True
                ).clamp_min(1e-12) + b / torch.linalg.vector_norm(
                    b, dim=-1, keepdim=True
                ).clamp_min(1e-12)
                directions /= torch.linalg.vector_norm(
                    directions, dim=-1, keepdim=True
                ).clamp_min(1e-12)
            if args.temporal_fusion == "coherent":
                values = {
                    "complex_sum": response.sum(dim=(1, 2)),
                    "complex_direction": (response[..., None] * directions).sum(
                        dim=(1, 2)
                    ),
                    "absolute_sum": response.abs().sum(dim=(1, 2)),
                }
            else:
                values = look_statistics(response, directions, torch)
            for key, value in values.items():
                value = _to_numpy(value)
                if key not in stats:
                    stats[key] = np.zeros(
                        (count, *value.shape[1:]),
                        dtype=value.dtype if np.iscomplexobj(value) else np.float64,
                    )
                stats[key][start : start + len(selected)] += value
    if args.temporal_fusion == "coherent":
        samples = len(looks) * args.aperture * 8
        summed = stats["complex_sum"]
        stats = {
            "vector": np.real(stats["complex_direction"] * summed.conj()[:, None])
            / samples**2,
            "power": np.abs(summed / samples) ** 2,
            "absolute_bound": (stats["absolute_sum"] / samples) ** 2,
        }
    estimated = finish_statistics(
        stats,
        1 if args.temporal_fusion == "coherent" else len(looks),
        args.min_coherence,
        args.dynamic_range_db,
    )
    index = np.arange(count).reshape(points.shape[0], points.shape[1] - 1)
    inverse = np.concatenate((index, index[:, :1]), axis=1).ravel()
    field = pack_normal_image(estimated, inverse, points)
    return {
        "pred_rhs": field["pred_rhs"],
        "normal_comparison": field["normal_comparison"].astype(np.float32),
        "normal_defined": field["defined"],
        "normal_valid": field["valid"],
        "normal_reflection": field["reflection"].astype(np.float32),
        "normal_coherence": field["coherence"].astype(np.float32),
    }


@dataclass(frozen=True)
class _SurfaceOptimization:
    initial_objective: float
    best_objective: float
    best_iteration: int
    initial_state: dict[str, torch.Tensor]
    history: list[dict[str, int | float]]


def _optimize_surface(
    scene, looks, supported, noise, training_power, acceptance_power, args
):
    device = scene.rays.device
    initial_state = {
        key: value.detach().clone() for key, value in scene.state_dict().items()
    }
    random = np.random.default_rng(args.seed)
    validation = supported[
        _to_tensor(
            random.choice(len(supported), min(2048, len(supported)), replace=False),
            device,
        )
    ]
    reference = acceptance_power[:, validation]

    def rf_objective(ids, split, normalizer):
        power = torch.stack([look.power(scene, ids, split) for look in looks])
        ratio = (power + noise[:, None]) / (normalizer + noise[:, None])
        return -torch.log(ratio.clamp_min(1e-12)).mean()

    def objective(ids, split, normalizer):
        rf = rf_objective(ids, split, normalizer)
        regularizer, plane = scene.regularizer()
        return rf + regularizer, rf, plane

    with torch.no_grad():
        value, rf, plane = objective(validation, 1, reference)
    initial_objective = value.item()
    best_value = initial_objective
    best_iteration = 0
    best_state = initial_state
    history = [
        {
            "iteration": 0,
            "objective": best_value,
            "rf_loss": rf.item(),
            "plane_loss": plane.item(),
        }
    ]
    optimizer = torch.optim.Adam(scene.parameters(), lr=args.learning_rate)
    for step in range(args.iterations):
        optimizer.zero_grad()
        training_rf = torch.zeros((), dtype=torch.float64, device=device)
        # Accumulate the full surface mean, including a short final batch,
        # before updating surface parameters.
        for start in range(0, len(supported), args.batch_size):
            ids = supported[start : start + args.batch_size]
            rf = rf_objective(ids, 0, training_power[:, ids])
            weighted = rf * (len(ids) / len(supported))
            weighted.backward()
            training_rf += weighted.detach()
        regularizer, _ = scene.regularizer()
        if not torch.isfinite(training_rf + regularizer):
            raise RuntimeError("Nonfinite RF/manifold objective")
        regularizer.backward()
        torch.nn.utils.clip_grad_norm_(scene.parameters(), 10)
        optimizer.step()
        if (step + 1) % 10 == 0 or step + 1 == args.iterations:
            with torch.no_grad():
                value, rf, plane = objective(validation, 1, reference)
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite RF acceptance objective")
            history.append(
                {
                    "iteration": step + 1,
                    "objective": value.item(),
                    "rf_loss": rf.item(),
                    "plane_loss": plane.item(),
                    "training_rf_loss_before_update": training_rf.item(),
                }
            )
            if (
                np.isfinite(value.item())
                and value.item() < best_value
                and rf.item() <= 0
            ):
                best_value = value.item()
                best_iteration = step + 1
                best_state = {
                    key: value.detach().clone()
                    for key, value in scene.state_dict().items()
                }
            print(
                f"  optimization {step + 1}/{args.iterations}: RF acceptance loss={rf.item():.5f}",
                flush=True,
            )
    scene.load_state_dict(best_state)
    return _SurfaceOptimization(
        initial_objective=initial_objective,
        best_objective=best_value,
        best_iteration=best_iteration,
        initial_state=initial_state,
        history=history,
    )


def _initialize_surface(args, trajectory, frames, image, device, xp, stages):
    stage_started = time.monotonic()
    print("  Reading RF and preparing target geometry", flush=True)
    registered = trajectory.registered_frame(
        args.frame_num, args.frame_num, args.aperture, args.geometry_json
    )
    frame, geometry, facing, origins, _ = registered
    stages["target_read_preprocess_seconds"] = time.monotonic() - stage_started
    stage_started = time.monotonic()
    coarse = image(frame, geometry, origins, facing, args.aperture, args.backend)
    baseline = panoradar_peak_surface(
        coarse, args.coarse_oversample, args.min_range, args.max_range
    )
    stages["target_coarse_image_seconds"] = time.monotonic() - stage_started
    stage_started = time.monotonic()
    print("  Selecting RF surface initialization", flush=True)
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
    refinement_look = _RegisteredRadarLook(
        trajectory, args.frame_num, registered, args, device, xp, include_padding=True
    )
    refine = partial(
        _refine_range_candidates,
        args,
        frame=frame,
        geometry=geometry,
        origins=origins,
        facing=facing,
        upper=min(
            args.max_range,
            C
            / (2 * refinement_look.wave.bandwidth_hz)
            * (refinement_look.wave.samples - 1 / args.oversample),
        ),
        sampler=refinement_look,
        aperture_selector=refinement_look.select_aperture,
    )
    initial, fields, initialization = select_motion_surface(
        args, trajectory, args.frame_num, frames, coarse, image, refine
    )
    stages["multiview_surface_initialization_seconds"] = (
        time.monotonic() - stage_started
    )
    return baseline, initial, fields, initialization


def reconstruct(args, trajectory):
    """Finalize one RF-only scene; never load labels, images or LiDAR poses."""
    started = time.monotonic()
    frames = temporal_frames(
        args.frame_num, trajectory.frame_count, args.temporal_window
    )
    if args.temporal_fusion == "coherent" and trajectory.platform_motion == "moving":
        raise ValueError(
            "Coherent normal fusion requires a static platform and stable inter-rotation phase"
        )
    record = trajectory.frame_metadata(args.frame_num)
    if args.aperture > min(
        trajectory.core_bounds(i)[1] - trajectory.core_bounds(i)[0] for i in frames
    ):
        raise ValueError("Aperture exceeds a contributing radar rotation")
    image = partial(
        motion_coarse_image,
        range_oversample=args.coarse_oversample,
        waveform=trajectory.waveform,
        azimuth_columns=1024 if trajectory.dataset_type == "SelfData" else 512,
    )
    xp = backend(args.backend)
    device = torch.device("cpu" if xp is np else "cuda")
    stages = {}
    baseline, initial, fields, initialization = _initialize_surface(
        args, trajectory, frames, image, device, xp, stages
    )
    stage_started = time.monotonic()
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()
    looks = []
    for number in frames:
        registered = trajectory.registered_frame(
            number, args.frame_num, args.aperture, args.geometry_json
        )
        looks.append(
            _RegisteredRadarLook(trajectory, number, registered, args, device, xp)
        )
    del registered
    anchor_time = trajectory.timestamps[int(trajectory.separators[args.frame_num])]
    scene = _RadialSurface(
        initial, fields["surface_valid"], fields["range_upper_m"], args, device
    )
    with torch.no_grad():
        initial_points = scene.points()
    for look in looks:
        look.prepare_apertures(initial_points, args.aperture, args.batch_size)
    stages["manifold_and_spectra_setup_seconds"] = time.monotonic() - stage_started
    stage_started = time.monotonic()
    initial_power, initial_coherence, _, training_power, acceptance_power = (
        _measure_surface(scene, looks, args.batch_size)
    )
    stages["initial_dense_matching_seconds"] = time.monotonic() - stage_started
    noise = initial_power.median(dim=1).values.clamp_min(1e-20) * 0.01
    supported = torch.nonzero(scene.supported, as_tuple=False).flatten()
    if len(supported) < 16 or initial_power.max().item() <= 1e-20:
        raise ValueError("Insufficient RF-supported surface for optimization")
    initialization_seconds = time.monotonic() - started
    stage_started = time.monotonic()
    optimization = _optimize_surface(
        scene, looks, supported, noise, training_power, acceptance_power, args
    )
    stages["optimization_seconds"] = time.monotonic() - stage_started
    stage_started = time.monotonic()
    power, coherence, reflectivity, _, _ = _measure_surface(
        scene, looks, args.batch_size
    )
    dense_gain = (
        torch.log(
            (power[:, supported] + noise[:, None])
            / (initial_power[:, supported] + noise[:, None])
        )
        .mean()
        .item()
    )
    accepted = (
        optimization.best_objective < optimization.initial_objective and dense_gain >= 0
    )
    if not accepted:
        scene.load_state_dict(optimization.initial_state)
        power, coherence, reflectivity, _, _ = _measure_surface(
            scene, looks, args.batch_size
        )
    stages["final_dense_matching_seconds"] = time.monotonic() - stage_started
    selected_passes = optimization.best_iteration if accepted else 0
    surface_shape = initial[:, :-1].shape[:2]
    points = _to_numpy(scene.points()).reshape(*surface_shape, 3)

    def panorama(value):
        return np.concatenate((value, value[:, :1]), axis=1)

    anchor = trajectory.poses.at([anchor_time])[0]
    points = panorama(points)
    normal_started = time.monotonic()
    normal_fields = _final_normals(scene, looks, points, args)
    normal_fields["normal_valid"] &= fields["surface_valid"]
    stages["final_echo_normals_seconds"] = time.monotonic() - normal_started
    times = np.concatenate([look.timestamps for look in looks])
    with torch.no_grad():
        plane_normals, plane_offsets = scene.planes()
    poses = trajectory.poses.at(times)
    result = {
        **baseline,
        **normal_fields,
        "points_m": points.astype(np.float32),
        "world_points_m": (points @ anchor[:3, :3].T + anchor[:3, 3]).astype(
            np.float32
        ),
        "predicted_range_m": np.linalg.norm(points, axis=-1).astype(np.float32),
        "initial_points_m": initial.astype(np.float32),
        "initial_range_m": np.linalg.norm(initial, axis=-1).astype(np.float32),
        "surface_valid": fields["surface_valid"],
        "rf_update_count": panorama(
            _to_numpy(scene.supported).reshape(surface_shape).astype(np.int32)
            * selected_passes
        ),
        "patch_ids": panorama(_to_numpy(scene.labels).reshape(surface_shape)),
        "geometric_patch_normals": _to_numpy(plane_normals),
        "geometric_patch_offsets_m": _to_numpy(plane_offsets),
        "initial_reflection": panorama(
            _to_numpy(initial_power.mean(0).sqrt()).reshape(surface_shape)
        ).astype(np.float32),
        "reflection": panorama(
            _to_numpy(power.mean(0).sqrt()).reshape(surface_shape)
        ).astype(np.float32),
        "initial_coherence": panorama(
            _to_numpy(initial_coherence.mean(0)).reshape(surface_shape)
        ).astype(np.float32),
        "coherence": panorama(
            _to_numpy(coherence.mean(0)).reshape(surface_shape)
        ).astype(np.float32),
        "complex_reflectivity": np.stack(
            [panorama(x.reshape(surface_shape)) for x in _to_numpy(reflectivity)]
        ),
        "pose_timestamps_s": times,
        "world_from_platform": poses,
        "initial_world_from_platform": poses.copy(),
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite finalized scene field: {name}")
    report = dict(
        record,
        prediction_uses_lidar=False,
        single_reconstruction=True,
        normal_estimation="signed RF echo votes at the finalized surface with the same per-chirp geometry; no range gradients",
        normal_fusion=args.temporal_fusion,
        panoradar_baseline={
            "method": "PanoRadar Eq. 11 beamforming; native range FFT peak per direction as Figure 9; no neural network",
            "source_frames": [args.frame_num],
            "aperture": args.aperture,
            "native_range_bins": trajectory.waveform.samples,
            "range_refinement": False,
            "surface_graph": False,
            "scope": "independent local equation implementation using the target RF circle and input trajectory geometry",
        },
        initialization=initialization,
        source_frames=frames,
        sample_intervals=[look.record["registered_raw_indices"] for look in looks],
        pose_source=trajectory.poses.source,
        pose_refinement_enabled=False,
        backend=str(device),
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        state_space="positive radial surface field and local S2 x R+ planes",
        phase_model="coherent within each disjoint rotation; separate powers and complex reflectivities across rotations",
        objective="log matched-power ratio with RF-only acceptance and robust local plane/range priors",
        optimization_accepted=accepted,
        optimization_coverage={
            "sampling": "full supported surface per Adam step; batch-weighted RF gradient accumulation",
            "supported_surface_points": len(supported),
            "attempted_full_passes": args.iterations,
            "attempted_point_updates": args.iterations * len(supported),
            "selected_full_passes": selected_passes,
            "finalized_covered_fraction": 1.0 if selected_passes else 0.0,
            "acceptance_uses_lidar": False,
        },
        rf_acceptance_objective_initial=optimization.initial_objective,
        rf_acceptance_objective_best=optimization.best_objective,
        dense_log_power_gain_before_acceptance=dense_gain,
        initialization_seconds=initialization_seconds,
        prediction_seconds=time.monotonic() - started,
        stages=stages,
        surface_support_reference="RF initialization; never used to exclude primary evaluation pixels",
        initial_mean_coherence=float(initial_coherence[:, supported].mean().item()),
        final_mean_coherence=float(coherence[:, supported].mean().item()),
        range_correction={
            "mean_absolute_m": float(
                np.abs(
                    result["predicted_range_m"][:, :-1]
                    - result["initial_range_m"][:, :-1]
                ).mean()
            ),
            "maximum_absolute_m": float(
                np.abs(
                    result["predicted_range_m"][:, :-1]
                    - result["initial_range_m"][:, :-1]
                ).max()
            ),
            "supported_fraction_near_chart_bound": float(
                (torch.tanh(scene.range_chart[supported]).abs() >= 0.95)
                .to(torch.float64)
                .mean()
                .item()
            ),
        },
        optimization_history=optimization.history,
        limitations=[
            "Local single-surface radial charts initialized from RF peaks; no guarantee of global optimum or recovery outside those branches",
            "Effective PanoRadar array unless calibrated Tx/Rx geometry is supplied; nominal motor angles remain fixed",
            "Matched-power surface optimization is not a full multiple-scatterer or multipath forward inversion",
            "RF acceptance chirps are excluded from gradient updates, but initialization uses all chirps",
        ],
    )
    return result, report
