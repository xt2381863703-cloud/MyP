"""Visualize a synchronized raw LiDAR frame in radar coordinates."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from ..config.data import (
    DEFAULT_DATA_CONFIG,
    load_data_settings,
    validate_data_settings,
)
from ..config.evaluation import EVALUATION_DEFAULTS
from ..dataset_io.trajectory import prepare_output_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--dataset-type", choices=("PanoData", "SelfData"))
    parser.add_argument("--lidar_translation_m", type=float, nargs=3, default=EVALUATION_DEFAULTS["lidar_translation_m"])
    parser.add_argument("--in_traj_folder", type=Path)
    parser.add_argument("--frame_num", type=int, default=0)
    parser.add_argument("--out_folder", type=Path, default=Path("outputs/lidar"))
    parser.add_argument(
        "--max_range",
        type=float,
        default=10.0,
        help="Maximum displayed range in meters",
    )
    args = parser.parse_args()
    try:
        data = load_data_settings(args.data_config)
        for key in ("dataset_type", "in_traj_folder"):
            if getattr(args, key) is not None:
                data[key] = getattr(args, key)
        validate_data_settings(data)
        args.dataset_type, args.in_traj_folder = data["dataset_type"], data["in_traj_folder"]
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.max_range <= 0 or not np.isfinite(args.max_range):
        parser.error("--max_range must be finite and positive")
    reference_metadata = None
    if args.dataset_type == "SelfData":
        from ..evaluation.self_lidar import SelfLidarReference

        settings = EVALUATION_DEFAULTS | {"lidar_translation_m": args.lidar_translation_m}
        points, reference_metadata = SelfLidarReference(args.in_traj_folder, 0, args.max_range, **settings).points(args.frame_num)
        points = points.astype(np.float32)
        raw_index = reference_metadata["raw_lidar_index"]
        raw_shape = [len(points), 3]
    else:
        with np.load(args.in_traj_folder / "data_params.npz") as params:
            frame_ids = params["lidar_sep"]
            if not 0 <= args.frame_num < len(frame_ids) - 1:
                parser.error(f"--frame_num must be between 0 and {len(frame_ids) - 2}")
            raw_index = int(frame_ids[args.frame_num])
            transform = np.eye(3)
            transform[:2, :2] = params["transform"]
        raw = np.load(args.in_traj_folder / "lidar_data.npy", mmap_mode="r")
        points = np.asarray(raw[raw_index]).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 0)
        points = (points[valid] @ transform).astype(np.float32)
        raw_shape = list(raw.shape[1:])
    ranges = np.linalg.norm(points, axis=1)
    shown = points[ranges <= args.max_range]
    if not len(shown):
        parser.error("No valid points within --max_range")
    args.out_folder = prepare_output_folder(args.out_folder, args.in_traj_folder)
    tag = f"{args.frame_num:05d}"
    np.save(args.out_folder / f"lidar_points_{tag}.npy", points)
    metadata = {
        "trajectory": str(args.in_traj_folder.resolve()),
        "dataset_type": args.dataset_type,
        "frame_num": args.frame_num,
        "raw_lidar_index": raw_index,
        "organized_frame_shape": raw_shape,
        "valid_points": len(points),
        "displayed_points": len(shown),
        "display_max_range_m": args.max_range,
        "coordinates": "radar-aligned x, y, z in meters",
        "color": "height z in meters; not RGB or return intensity",
        "saved_points": "all valid points, including those beyond the displayed range",
    }
    if reference_metadata is not None:
        metadata["reference"] = reference_metadata
    (args.out_folder / f"lidar_points_{tag}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    # Keep a fixed sample for the static preview; HTML retains every point in range.
    rng = np.random.default_rng(0)
    sample = shown[rng.choice(len(shown), min(45000, len(shown)), replace=False)]
    z_min, z_max = np.percentile(shown[:, 2], [1, 99])
    fig = plt.figure(figsize=(15, 7), layout="constrained")
    fig.suptitle(
        f"{args.dataset_type} LiDAR | {args.in_traj_folder.name} | frame {args.frame_num}\n"
        f"{len(shown):,} points within {args.max_range:g} m; colored by height",
        fontsize=14,
    )
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(
        *sample.T,
        c=sample[:, 2],
        cmap="turbo",
        vmin=z_min,
        vmax=z_max,
        s=0.5,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter([0], [0], [0], marker="x", color="black", s=50)
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="3D point cloud")
    ax.set_box_aspect(np.maximum(np.ptp(shown, axis=0), 0.1))
    ax.view_init(elev=28, azim=-60)

    ax = fig.add_subplot(1, 2, 2)
    image = ax.scatter(
        shown[:, 0],
        shown[:, 1],
        c=shown[:, 2],
        cmap="turbo",
        vmin=z_min,
        vmax=z_max,
        s=0.4,
        alpha=0.8,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter([0], [0], marker="x", color="black", s=60, label="Radar origin")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Top view")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=fig.axes, shrink=0.75, label="Height z (m)")
    png = args.out_folder / f"lidar_pointcloud_{tag}.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    figure = go.Figure(
        go.Scatter3d(
            x=shown[:, 0],
            y=shown[:, 1],
            z=shown[:, 2],
            mode="markers",
            marker={
                "size": 1.5,
                "opacity": 0.8,
                "color": shown[:, 2],
                "colorscale": "Turbo",
                "cmin": z_min,
                "cmax": z_max,
                "colorbar": {"title": "Height z (m)"},
            },
            name="LiDAR",
            hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker={"size": 5, "color": "black", "symbol": "cross"},
            name="Radar origin",
        )
    )
    figure.update_layout(
        title=f"LiDAR | {args.in_traj_folder.name} | frame {args.frame_num} | range <= {args.max_range:g} m",
        scene={
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 60},
    )
    html = args.out_folder / f"lidar_pointcloud_{tag}.html"
    figure.write_html(html, include_plotlyjs=True, full_html=True)
    print(json.dumps(metadata, indent=2))
    print(f"PNG: {png.resolve()}")
    print(f"HTML: {html.resolve()}")


if __name__ == "__main__":
    main()
