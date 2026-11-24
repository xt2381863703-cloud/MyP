"""Reconstruct RF scenes and write finalized Cartesian predictions."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from ..config.data import (
    DEFAULT_DATA_CONFIG,
    load_data_settings,
    validate_algorithm_data_keys,
)
from ..config.evaluation import validate_evaluation_settings
from ..config.paths import CODE_ROOT, CONFIG_ROOT, PROJECT_ROOT
from ..config.settings import DEFAULTS
from ..dataset_io.formats import dataset_root, rf_input_paths
from ..dataset_io.trajectory import prepare_output_folder


def _settings(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "imaging.yaml")
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    for key in (
        "in_traj_folder",
        "out_folder",
        "geometry_json",
        "processed_traj_folder",
    ):
        parser.add_argument("--" + key.replace("_", "-"), type=Path)
    for key in ("frame_num", "iterations", "temporal_window", "batch_size"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--dataset-type", choices=("PanoData", "SelfData"))
    parser.add_argument("--platform-motion", choices=("static", "moving"))
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"))
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument(
        "--evaluate",
        action="store_true",
        help="兼容旧参数；默认已在预测完成后执行激光评价",
    )
    evaluation.add_argument(
        "--evaluate-only",
        type=Path,
        help="Evaluate a finalized scene NPZ and add missing plots without inference",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="仅保存预测与评价，不生成图片"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="显示统一配置，不读取传感器数据"
    )
    parser.add_argument("--temporal-fusion", choices=("multilook", "coherent"))
    cli = parser.parse_args(argv)
    if cli.evaluate_only is not None:
        return None, cli
    try:
        config = yaml.safe_load(cli.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise TypeError("Config must be a YAML mapping")
        if config.keys() - DEFAULTS.keys():
            raise ValueError(
                f"Unknown config keys: {sorted(config.keys() - DEFAULTS.keys())}"
            )
        validate_algorithm_data_keys(config)
        settings = DEFAULTS | load_data_settings(cli.data_config) | config
        settings.update(
            {
                key: value
                for key, value in vars(cli).items()
                if key in DEFAULTS and value is not None
            }
        )
        settings["data_config"] = cli.data_config.resolve()
        for key in ("frame_num", "seed"):
            if type(settings[key]) is not int or settings[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in (
            "iterations",
            "temporal_window",
            "batch_size",
            "aperture",
            "oversample",
            "coarse_oversample",
            "surface_candidates",
            "surface_bp_iterations",
            "surface_min_views",
            "surface_refine_steps",
        ):
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if (
            settings["aperture"] < 3
            or settings["aperture"] % 2 == 0
            or settings["temporal_window"] % 2 == 0
        ):
            raise ValueError(
                "aperture must be odd and >=3; temporal_window must be odd"
            )
        if settings["oversample"] < 2 or not 2 <= settings["surface_candidates"] <= 16:
            raise ValueError(
                "oversample must be >=2; surface_candidates must be between 2 and 16"
            )
        if (
            settings["surface_refine_steps"] < 3
            or settings["surface_refine_steps"] % 2 == 0
        ):
            raise ValueError("surface_refine_steps must be odd and >=3")
        for key in (
            "learning_rate",
            "range_bound_m",
            "surface_refine_radius_m",
            "min_range",
            "max_range",
            "surface_support_db",
            "geometry_weight",
            "range_prior_weight",
        ):
            value = settings[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
        for key in ("surface_temporal_weight", "surface_spatial_weight"):
            value = settings[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if (
            settings["surface_temporal_weight"] > 1
            or not settings["min_range"] < settings["max_range"]
        ):
            raise ValueError("Invalid temporal weight or range interval")
        shape = settings["patch_shape"]
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or any(type(v) is not int or v < 1 for v in shape)
        ):
            raise ValueError("patch_shape must contain two positive integers")
        if shape[0] > 64 or shape[1] > 127:
            raise ValueError(
                "Local charts must fit within 64 elevation and 127 azimuth cells"
            )
        if type(settings["keep_background"]) is not bool:
            raise ValueError("keep_background must be boolean")
        if settings["backend"] not in ("auto", "cpu", "cuda") or settings[
            "platform_motion"
        ] not in ("static", "moving"):
            raise ValueError("Invalid backend or platform_motion")
        if settings["temporal_fusion"] not in ("multilook", "coherent"):
            raise ValueError("temporal_fusion must be multilook or coherent")
        if (
            not 0 <= settings["min_coherence"] <= 1
            or not np.isfinite(settings["dynamic_range_db"])
            or settings["dynamic_range_db"] <= 0
        ):
            raise ValueError("Invalid normal confidence thresholds")
        validate_evaluation_settings(settings)
        root = dataset_root(settings["dataset_type"])
        for key in (
            "in_traj_folder",
            "out_folder",
            "geometry_json",
            "processed_traj_folder",
        ):
            if settings[key] is not None:
                settings[key] = Path(settings[key]).expanduser().resolve()
        if root not in settings["in_traj_folder"].parents:
            raise ValueError(f"Input trajectory must be under {root}")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return SimpleNamespace(**settings), cli


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _fingerprint(args, input_paths):
    digest = hashlib.sha256()
    for name in (
        "imaging/scene.py",
        "imaging/manifold.py",
        "imaging/backprojection.py",
        "motion/trajectory.py",
        "motion/aperture.py",
        "motion/temporal.py",
        "imaging/selection.py",
        "imaging/core.py",
        "dataset_io/trajectory.py",
        "dataset_io/pano_data.py",
        "dataset_io/self_data.py",
        "dataset_io/formats.py",
        "config/settings.py",
        "config/data.py",
        "config/paths.py",
        "config/evaluation.py",
        "normals/pixels.py",
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


def _paths(folder, frame_num):
    tag = f"{frame_num:05d}"
    return {
        "prediction": folder / f"scene_{tag}.npz",
        "points": folder / f"scene_{tag}.ply",
        "report": folder / f"report_{tag}.json",
        "reference": folder / f"lidar_reference_{tag}.npz",
        "evaluation": folder / f"evaluation_{tag}.json",
        "figures": [
            folder / f"{name}_{tag}.png"
            for name in (
                "rf_range_comparison",
                "normal_comparison",
                "geometry_comparison",
            )
        ],
    }


def _evaluate_and_plot(args, cli, prediction, paths, report):
    # No sensor-dependent loader is imported or initialized before RF finalization.
    from ..evaluation.comparison import compare
    from ..evaluation.reference import create_reference

    reference, unmasked, metadata = create_reference(args).load_pair(args.frame_num)
    evaluation, errors = compare(
        prediction,
        reference,
        args,
        pose_refinement_enabled=report.get("pose_refinement_enabled", True),
    )
    evaluation["reference"] = metadata
    if paths["evaluation"].exists():
        saved = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
        if saved != evaluation:
            raise ValueError("重新计算的激光评价与已有记录不同，已保留原产物")
    else:
        _json(paths["evaluation"], evaluation)
    if not paths["reference"].exists():
        values = {f"filtered_{key}": value for key, value in reference.items()}
        values.update({f"unmasked_{key}": value for key, value in unmasked.items()})
        values.update({f"comparison_{key}": value for key, value in errors.items()})
        np.savez_compressed(paths["reference"], **values)
    if not cli.no_plot:
        from ..visualization.comparison import plot_comparisons

        plot_comparisons(
            prediction,
            reference,
            unmasked,
            evaluation,
            errors,
            metadata,
            args,
            paths["figures"],
        )
    report.update(
        status="success",
        evaluation_complete=True,
        artifacts={
            "prediction": paths["prediction"].name,
            "point_cloud": paths["points"].name,
            "lidar_reference": paths["reference"].name,
            "evaluation": paths["evaluation"].name,
            "figures": [path.name for path in paths["figures"] if path.exists()],
        },
    )
    _json(paths["report"], report)
    print(
        json.dumps(
            {
                "距离误差": evaluation["range"],
                "三维位置误差": evaluation["position"],
                "当前法线平均夹角_度": {
                    key: value["sn_angle_mean"]
                    for key, value in evaluation["normal"].items()
                },
                "玻璃掩码可用": metadata["glass_labels_available"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    for path in paths["figures"]:
        if path.exists():
            print(f"对比图：{path}", flush=True)
    return report


def _evaluate_saved(cli):
    prediction_path = cli.evaluate_only.expanduser().resolve()
    if prediction_path.suffix != ".npz" or not prediction_path.stem.startswith(
        "scene_"
    ):
        raise ValueError("--evaluate-only 需要 scene_NNNNN.npz")
    tag = prediction_path.stem.removeprefix("scene_")
    report = json.loads(
        prediction_path.with_name(f"report_{tag}.json").read_text(encoding="utf-8")
    )
    if report.get("format") != "unified_rf_v1" or not report.get(
        "prediction_finalized"
    ):
        raise ValueError(
            "该预测缺少统一的原方法基线或回波法线，请用 python -m code.image_scene 重新生成"
        )
    if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != report.get(
        "prediction_sha256"
    ):
        raise ValueError("预测文件与保存的哈希不符，已保留原产物")
    args = SimpleNamespace(**(DEFAULTS | report["settings"]))
    args.in_traj_folder = Path(args.in_traj_folder)
    with np.load(prediction_path, allow_pickle=False) as data:
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
    paths = _paths(prediction_path.parent, args.frame_num)
    return _evaluate_and_plot(args, cli, prediction, paths, report)


def main(argv=None):
    started = time.monotonic()
    args, cli = _settings(argv)
    if cli.evaluate_only is not None:
        return _evaluate_saved(cli)
    if cli.dry_run:
        result = {
            "dataset_type": args.dataset_type,
            "in_traj_folder": str(args.in_traj_folder),
            "frame_num": args.frame_num,
            "temporal_window": args.temporal_window,
            "iterations": args.iterations,
            "shared_surface_and_trajectory": True,
            "normal_method": "RF signed echo votes on the finalized surface",
            "figures": [
                "全景距离与误差",
                "回波法线与两种激光参考",
                "三维点云与零高度带俯视图",
            ],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    input_paths = rf_input_paths(args)
    from ..motion.trajectory import MotionTrajectory
    from .manifold import reconstruct

    if args.out_folder is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.out_folder = (
            PROJECT_ROOT
            / "outputs"
            / "imaging"
            / args.in_traj_folder.parent.name
            / args.in_traj_folder.name
            / f"{args.platform_motion}_unified_{stamp}"
        )
    args.out_folder = prepare_output_folder(args.out_folder, args.in_traj_folder)
    paths = _paths(args.out_folder, args.frame_num)
    if any(args.out_folder.glob(f"*_{args.frame_num:05d}.*")):
        raise FileExistsError("该圈的产物已存在，请使用新的输出目录")
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    fingerprint = _fingerprint(args, input_paths)
    trajectory = MotionTrajectory(
        args.in_traj_folder,
        subtract_background=not args.keep_background,
        platform_motion=args.platform_motion,
        dataset_type=args.dataset_type,
    )
    print(f"统一重建第 {args.frame_num} 圈：表面、轨迹与回波法线", flush=True)
    prediction, report = reconstruct(args, trajectory)
    np.savez_compressed(paths["prediction"], **prediction)
    points = prediction["points_m"][:, :-1].reshape(-1, 3)
    intensity = prediction["reflection"][:, :-1].ravel()
    support = prediction["surface_valid"][:, :-1].ravel()
    with paths["points"].open("w", encoding="ascii") as stream:
        stream.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nproperty uchar rf_supported\nend_header\n"
        )
        np.savetxt(
            stream,
            np.column_stack((points, intensity, support)),
            fmt="%.7g %.7g %.7g %.7g %d",
        )
    report.update(
        format="unified_rf_v1",
        status="prediction_finalized",
        settings=settings,
        run_fingerprint=fingerprint,
        prediction_finalized=True,
        prediction_sha256=hashlib.sha256(paths["prediction"].read_bytes()).hexdigest(),
        prediction_and_save_seconds=time.monotonic() - started,
        evaluation_complete=False,
        timing_excludes="interpreter startup, LiDAR evaluation and plotting",
    )
    _json(paths["report"], report)
    print(f"射频预测已固定并保存：{paths['prediction']}；现在开始激光评价", flush=True)
    return _evaluate_and_plot(args, cli, prediction, paths, report)


if __name__ == "__main__":
    main()
