"""Paths, provenance and serialization for finalized RF scene artifacts."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from ..config.paths import CODE_ROOT

PREDICTION_FORMAT = "unified_rf_v1"


@dataclass(frozen=True)
class SceneArtifacts:
    prediction: Path
    point_cloud: Path
    report: Path
    reference: Path
    evaluation: Path
    figures: tuple[Path, Path, Path]

    @classmethod
    def for_frame(cls, folder: Path, frame_num: int) -> "SceneArtifacts":
        tag = f"{frame_num:05d}"
        return cls(
            prediction=folder / f"scene_{tag}.npz",
            point_cloud=folder / f"scene_{tag}.ply",
            report=folder / f"report_{tag}.json",
            reference=folder / f"lidar_reference_{tag}.npz",
            evaluation=folder / f"evaluation_{tag}.json",
            figures=(
                folder / f"rf_range_comparison_{tag}.png",
                folder / f"normal_comparison_{tag}.png",
                folder / f"geometry_comparison_{tag}.png",
            ),
        )


def _prediction_sha256(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    stream.seek(0)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_fingerprint(args, input_paths):
    digest = hashlib.sha256()
    for name in (
        "imaging/scene.py",
        "imaging/artifacts.py",
        "imaging/manifold.py",
        "imaging/backprojection.py",
        "motion/trajectory.py",
        "motion/aperture.py",
        "motion/temporal.py",
        "imaging/selection.py",
        "imaging/core.py",
        "imaging/coordinates.py",
        "dataset_io/trajectory.py",
        "dataset_io/pano_data.py",
        "dataset_io/self_data.py",
        "dataset_io/formats.py",
        "config/settings.py",
        "config/data.py",
        "config/paths.py",
        "config/evaluation.py",
        "normals/pixels.py",
        "evaluation/scene.py",
        "evaluation/comparison.py",
        "evaluation/normals.py",
        "evaluation/reference.py",
        "evaluation/lidar.py",
        "evaluation/self_lidar.py",
        "visualization/comparison.py",
    ):
        digest.update((CODE_ROOT / name).read_bytes())
    for path in input_paths:
        stat = path.stat()
        digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(json.dumps(vars(args), sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _write_point_cloud(path, prediction):
    points = prediction["points_m"][:, :-1].reshape(-1, 3)
    intensity = prediction["reflection"][:, :-1].ravel()
    support = prediction["surface_valid"][:, :-1].ravel()
    with path.open("x", encoding="ascii") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float intensity\nproperty uchar rf_supported\nend_header\n"
        )
        np.savetxt(
            stream,
            np.column_stack((points, intensity, support)),
            fmt="%.7g %.7g %.7g %.7g %d",
        )


def save_finalized_prediction(
    paths: SceneArtifacts, prediction, report, *, settings, fingerprint, started
):
    """Publish the finalization report after writing both RF prediction files."""
    with paths.prediction.open("x+b") as stream:
        np.savez_compressed(stream, **prediction)
        prediction_sha256 = _prediction_sha256(stream)
    _write_point_cloud(paths.point_cloud, prediction)
    report.update(
        format=PREDICTION_FORMAT,
        status="prediction_finalized",
        settings=settings,
        run_fingerprint=fingerprint,
        prediction_finalized=True,
        prediction_sha256=prediction_sha256,
        prediction_and_save_seconds=time.monotonic() - started,
        evaluation_complete=False,
        timing_excludes="interpreter startup, LiDAR evaluation and plotting",
    )
    write_json(paths.report, report)


def load_finalized_prediction(prediction_path: Path):
    """Read and verify a saved RF prediction before any reference is loaded."""
    if prediction_path.suffix != ".npz" or not prediction_path.stem.startswith(
        "scene_"
    ):
        raise ValueError("--evaluate-only 需要 scene_NNNNN.npz")
    tag = prediction_path.stem.removeprefix("scene_")
    report = json.loads(
        prediction_path.with_name(f"report_{tag}.json").read_text(encoding="utf-8")
    )
    if report.get("format") != PREDICTION_FORMAT or not report.get(
        "prediction_finalized"
    ):
        raise ValueError(
            "该预测缺少统一的原方法基线或回波法线，请用 python -m code.image_scene 重新生成"
        )
    with prediction_path.open("rb") as stream:
        if _prediction_sha256(stream) != report.get("prediction_sha256"):
            raise ValueError("预测文件与保存的哈希不符，已保留原产物")
        with np.load(stream, allow_pickle=False) as data:
            prediction = {key: data[key] for key in data.files}
    required = {
        "points_m",
        "predicted_range_m",
        "panoradar_range_m",
        "panoradar_points_m",
        "pred_rhs",
        "normal_defined",
        "normal_valid",
    }
    if not required <= prediction.keys():
        raise ValueError(f"统一预测缺少字段：{sorted(required - prediction.keys())}")
    return prediction, report
