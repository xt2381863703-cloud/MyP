"""Read only the RF fields of the original PanoRadar format."""

from pathlib import Path

import numpy as np

from ..imaging.core import VIRTUAL_SPACING_M, Waveform


class PanoData:
    def __init__(self, folder):
        folder = Path(folder)
        with np.load(folder / "data_params.npz", allow_pickle=False) as params:
            self.separators = params["radar_sep"].copy()
            self.radius = float(params["r_radar"])
        self.raw = np.load(folder / "radar_data.npy", mmap_mode="r")
        self.timestamps = np.loadtxt(folder / "radar_ts.txt", ndmin=1)
        self.waveform = Waveform(self.raw.shape[-1])
        self.virtual_spacing_m = VIRTUAL_SPACING_M
        self.frame_ends = self.separators[1:].copy()
        self.framing = "native radar_sep; per-chirp platform pose"
