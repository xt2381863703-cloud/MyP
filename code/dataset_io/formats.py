"""Dataset locations, required motion files and RF-only fingerprint inputs."""

import shlex
import sys
from pathlib import Path

from ..config.paths import PROJECT_ROOT

DATASET_ROOTS = {
    "PanoData": Path("/mnt/dataset/panoradar"),
    "SelfData": Path("/mnt/dataset/suradar"),
}

_PANORADAR_GENERATION = """import cupy as cp
from panoradar_SP import velocity_estimation
from panoradar_SP.motion_estimation import estimate_params_for_trajectory

original_range_aoa = velocity_estimation.faster_range_AoA_images
cp.fft.config.get_plan_cache().set_memsize(256 * 1024**2)

def batched_range_aoa(signal_bf, azimuth_compen, params):
    window = int(params["N_syn_ante"])
    beam_count = signal_bf.shape[0] - window + 1
    for start in range(0, beam_count, 64):
        stop = min(start + 64, beam_count)
        batch = original_range_aoa(
            signal_bf[start:stop + window - 1], azimuth_compen, params
        )
        if start == 0:
            result = cp.empty((beam_count, *batch.shape[1:]), dtype=batch.dtype)
        result[start:stop] = batch
    return result

velocity_estimation.faster_range_AoA_images = batched_range_aoa
estimate_params_for_trajectory(sys.argv[1])"""


def dataset_root(dataset_type):
    if dataset_type not in DATASET_ROOTS:
        raise ValueError(f"dataset_type must be one of {tuple(DATASET_ROOTS)}")
    return DATASET_ROOTS[dataset_type].resolve()


def require_motion_output(folder, dataset_type):
    path = Path(folder) / "motion_output.npz"
    if path.is_file():
        return path

    if dataset_type == "SelfData":
        source = Path("/data/glz/project/SURadar")
        generator = (
            "from suradar_SP.motion_estimation import estimate_params_for_trajectory\n"
            'estimate_params_for_trajectory(sys.argv[1], framing="imu")'
        )
    else:
        source = PROJECT_ROOT / "paper" / "PanoRadar"
        generator = _PANORADAR_GENERATION
    command = shlex.join((sys.executable, "-B", "-u", "-", str(path.parent.resolve())))
    guard = shlex.join(("test", "!", "-e", str(path.absolute())))
    raise FileNotFoundError(
        f"{dataset_type} moving data requires {path}\n"
        "Run the following command offline to generate this file.\n"
        f"cd {shlex.quote(str(source))} && {guard} && {command} <<'PY'\n"
        f"import sys\n{generator}\nPY"
    )


def rf_input_paths(args):
    names = ["data_params.npz"]
    dataset_type = getattr(args, "dataset_type", "PanoData")
    if dataset_type == "SelfData":
        names.extend(("radar.npz", "radar_config.yaml"))
        if args.platform_motion == "moving":
            names.append("imu.npz")
    else:
        names.extend(("radar_data.npy", "radar_ts.txt"))
    paths = [args.in_traj_folder / name for name in names]
    if args.platform_motion == "moving":
        paths.append(require_motion_output(args.in_traj_folder, dataset_type))
    if args.geometry_json is not None:
        paths.append(args.geometry_json)
    return paths
