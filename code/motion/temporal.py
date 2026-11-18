"""Frame windows and covariance statistics for registered moving RF looks."""

import numpy as np


def temporal_frames(frame_num, frame_count, window):
    """Centered window clipped at sequence ends; never duplicate boundary looks."""
    if type(window) is not int or window < 1 or window % 2 != 1:
        raise ValueError("temporal_window must be a positive odd integer")
    if not 0 <= frame_num < frame_count:
        raise ValueError("Target frame is outside trajectory")
    half = window // 2
    frames = list(
        range(max(0, frame_num - half), min(frame_count, frame_num + half + 1))
    )
    if window > 1 and len(frames) < 2:
        raise ValueError("Multi-frame estimation requires at least two distinct frames")
    return frames


def look_statistics(response, directions, xp=np):
    """Compute C1 and its directional moment without materializing C=yy^H.

    y=I/J; C=y y^H; w=Re(C1)=Re(I conj(sum I))/J^2.
    Average these sufficient statistics across frames BEFORE normalizing vectors.
    Negative votes are retained, as required by mmNorm Eq. 6.
    """
    count = response.shape[1] * response.shape[2]
    summed = response.sum(axis=(1, 2))
    weights = xp.real(response * summed.conj()[:, None, None]) / count**2
    return {
        "vector": (weights[..., None] * directions).sum(axis=(1, 2)),
        "power": (xp.abs(summed) / count) ** 2,
        "absolute_bound": (xp.abs(response).sum(axis=(1, 2)) / count) ** 2,
    }


def finish_statistics(stats, looks, min_coherence, dynamic_range_db):
    """Normalize the joint vote and compute RF confidence."""
    stats = {key: value / looks for key, value in stats.items()}
    eps = np.finfo(np.float64).eps
    length = np.linalg.norm(stats["vector"], axis=-1)
    reflection = np.sqrt(np.maximum(stats["power"], 0))
    defined = (reflection > 1e-10) & (length > eps * stats["absolute_bound"])
    normal = stats["vector"] / np.maximum(length[:, None], np.finfo(float).tiny)
    normal = np.where(defined[:, None], normal, np.nan)
    coherence = np.sqrt(
        np.maximum(stats["power"], 0)
        / np.maximum(stats["absolute_bound"], np.finfo(float).tiny)
    )
    relative_db = 20 * np.log10(
        np.maximum(reflection, 1e-30) / max(float(reflection.max()), 1e-30)
    )
    return {
        "normal": normal,
        "reflection": reflection,
        "coherence": np.clip(coherence, 0, 1),
        "defined": defined,
        "relative_db": relative_db,
        "valid": defined
        & (reflection > 1e-10)
        & (relative_db >= -dynamic_range_db)
        & (coherence >= min_coherence),
    }
