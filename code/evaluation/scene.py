"""Post-prediction evaluation and plotting for RF scenes."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..config.settings import DEFAULTS
from ..imaging.artifacts import SceneArtifacts, load_finalized_prediction, write_json


def evaluate_scene(args, prediction, paths, report, *, no_plot=False):
    """Evaluate an already finalized RF scene and fill missing figures."""
    from .comparison import compare
    from .reference import create_reference

    azimuth_columns = prediction["predicted_range_m"].shape[1]
    reference, unmasked, metadata = create_reference(
        args, azimuth_columns=azimuth_columns
    ).load_pair(args.frame_num)
    evaluation, errors = compare(
        prediction,
        reference,
        args,
        pose_refinement_enabled=report.get("pose_refinement_enabled", True),
    )
    evaluation["reference"] = metadata
    if paths.evaluation.exists():
        saved = json.loads(paths.evaluation.read_text(encoding="utf-8"))
        if saved != evaluation:
            raise ValueError("重新计算的激光评价与已有记录不同，已保留原产物")
    else:
        write_json(paths.evaluation, evaluation)
    if not paths.reference.exists():
        values = {f"filtered_{key}": value for key, value in reference.items()}
        values.update({f"unmasked_{key}": value for key, value in unmasked.items()})
        values.update({f"comparison_{key}": value for key, value in errors.items()})
        np.savez_compressed(paths.reference, **values)
    if not no_plot:
        from ..visualization.comparison import plot_comparisons

        plot_comparisons(
            prediction,
            reference,
            unmasked,
            evaluation,
            errors,
            metadata,
            args,
            paths.figures,
        )
    report.update(
        status="success",
        evaluation_complete=True,
        artifacts={
            "prediction": paths.prediction.name,
            "point_cloud": paths.point_cloud.name,
            "lidar_reference": paths.reference.name,
            "evaluation": paths.evaluation.name,
            "figures": [path.name for path in paths.figures if path.exists()],
        },
    )
    write_json(paths.report, report)
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
    for path in paths.figures:
        if path.exists():
            print(f"对比图：{path}", flush=True)
    return report


def evaluate_saved_scene(prediction_path, *, no_plot=False):
    prediction_path = prediction_path.expanduser().resolve()
    prediction, report = load_finalized_prediction(prediction_path)
    args = SimpleNamespace(**(DEFAULTS | report["settings"]))
    args.in_traj_folder = Path(args.in_traj_folder)
    paths = SceneArtifacts.for_frame(prediction_path.parent, args.frame_num)
    return evaluate_scene(args, prediction, paths, report, no_plot=no_plot)
