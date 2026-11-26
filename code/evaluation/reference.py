"""Select a reference loader only after an RF prediction has been finalized."""

from pathlib import Path

from ..config.evaluation import EVALUATION_DEFAULTS
from ..dataset_io.formats import dataset_root


def reference_method(args):
    return (
        "selfdata_raw_lidar_local_pca"
        if getattr(args, "dataset_type", "PanoData") == "SelfData"
        else "panoradar_original_training_range_gradient"
    )


def create_reference(args, *, azimuth_columns=512):
    raw_folder = Path(args.in_traj_folder)
    if getattr(args, "dataset_type", "PanoData") == "SelfData":
        from .self_lidar import SelfLidarReference

        settings = {
            key: getattr(args, key, value) for key, value in EVALUATION_DEFAULTS.items()
        }
        return SelfLidarReference(
            raw_folder,
            args.min_range,
            args.max_range,
            azimuth_columns=azimuth_columns,
            **settings,
        )
    from .lidar import TrainingLidarReference

    processed = getattr(args, "processed_traj_folder", None)
    if processed is None:
        processed = (
            dataset_root("PanoData")
            / "processed"
            / raw_folder.parent.name
            / raw_folder.name
        )
    return TrainingLidarReference(raw_folder, processed)
