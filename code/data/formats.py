"""Dataset locations and RF-only fingerprint inputs."""

from pathlib import Path

DATASET_ROOTS = {
    "PanoData": Path("/mnt/dataset/panoradar"),
    "SelfData": Path("/mnt/dataset/suradar"),
}


def dataset_root(dataset_type):
    if dataset_type not in DATASET_ROOTS:
        raise ValueError(f"dataset_type must be one of {tuple(DATASET_ROOTS)}")
    return DATASET_ROOTS[dataset_type].resolve()


def rf_input_paths(args):
    names = ["data_params.npz"]
    if getattr(args, "dataset_type", "PanoData") == "SelfData":
        names.extend(("radar.npz", "radar_config.yaml"))
        if args.platform_motion == "moving":
            names.append("imu.npz")
    else:
        names.extend(("radar_data.npy", "radar_ts.txt"))
    if args.platform_motion == "moving" and args.motion_poses is None:
        names.append("motion_output.npz")
    paths = [args.in_traj_folder / name for name in names]
    return paths + [
        value for value in (args.motion_poses, args.geometry_json) if value is not None
    ]
