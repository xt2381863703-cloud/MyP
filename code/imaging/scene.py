"""Reconstruct RF scenes and write finalized Cartesian predictions."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..config.data import DEFAULT_DATA_CONFIG
from ..config.paths import CONFIG_ROOT, PROJECT_ROOT
from ..config.settings import load_imaging_settings
from ..dataset_io.formats import rf_input_paths
from ..dataset_io.trajectory import prepare_output_folder
from .artifacts import SceneArtifacts, run_fingerprint, save_finalized_prediction


def _parse_command_line(argv):
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
        args = load_imaging_settings(cli.config, cli.data_config, vars(cli))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return args, cli


def main(argv=None):
    started = time.monotonic()
    args, cli = _parse_command_line(argv)
    if cli.evaluate_only is not None:
        from ..evaluation.scene import evaluate_saved_scene

        return evaluate_saved_scene(cli.evaluate_only, no_plot=cli.no_plot)
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
    paths = SceneArtifacts.for_frame(args.out_folder, args.frame_num)
    if any(args.out_folder.glob(f"*_{args.frame_num:05d}.*")):
        raise FileExistsError("该圈的产物已存在，请使用新的输出目录")
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    fingerprint = run_fingerprint(args, input_paths)
    trajectory = MotionTrajectory(
        args.in_traj_folder,
        subtract_background=not args.keep_background,
        platform_motion=args.platform_motion,
        dataset_type=args.dataset_type,
    )
    print(f"统一重建第 {args.frame_num} 圈：表面、轨迹与回波法线", flush=True)
    prediction, report = reconstruct(args, trajectory)
    save_finalized_prediction(
        paths,
        prediction,
        report,
        settings=settings,
        fingerprint=fingerprint,
        started=started,
    )
    print(f"射频预测已固定并保存：{paths.prediction}；现在开始激光评价", flush=True)

    from ..evaluation.scene import evaluate_scene

    return evaluate_scene(args, prediction, paths, report, no_plot=cli.no_plot)


if __name__ == "__main__":
    main()
