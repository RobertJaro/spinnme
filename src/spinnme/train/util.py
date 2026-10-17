import os
import tempfile

import torch
from pytorch_lightning.utilities import rank_zero_only

from spinnme.data.differential_rotation import carrington_rotation_rate, differential_rotation_rate
from spinnme.data.util import cartesian_to_spherical, spherical_to_cartesian


def atan2_safe(numerator, denominator):
    epsilon = 1e-7
    nudge = (denominator == 0) * epsilon
    denominator = denominator + nudge
    out = torch.atan2(numerator, denominator)
    return out


def acos_safe(x):
    epsilon = 1e-7
    nudge_pos = (x == 1) * epsilon
    nudge_neg = (x == -1) * epsilon
    x = x - nudge_pos + nudge_neg
    out = torch.acos(x)
    return out


@rank_zero_only
def log_wandb_image(fig, name: str, dpi: int = 100, *, logger=None, step=None):
    """Send a rendered figure to the trainer's W&B run at its training step."""
    if logger is None:
        return
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as stream:
        path = stream.name
    try:
        fig.savefig(path, format='jpg', dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.log_image(key=name, images=[path], step=step)
    finally:
        os.remove(path)


def random_time_shift_coords(coords, seconds_per_dt, scale=0.1, ):
    time_shift = torch.randn_like(coords[..., 3:4]) * scale  # small random shift
    spherical_coords = cartesian_to_spherical(coords[..., :3], torch)
    lat = torch.pi / 2 - spherical_coords[..., 1:2]
    lon = spherical_coords[..., 2:3]
    omega_carrington = carrington_rotation_rate()
    omega_differential = differential_rotation_rate(lat, torch)
    lon_shift = (omega_differential - omega_carrington) * time_shift * seconds_per_dt
    spherical_coords_shift = torch.cat([spherical_coords[..., 0:1], spherical_coords[..., 1:2],
                                        lon + lon_shift], dim=-1)
    cartesian_coords_shift = spherical_to_cartesian(spherical_coords_shift, torch)
    coords_shift = torch.cat([cartesian_coords_shift, coords[..., 3:4] + time_shift], dim=-1)
    return coords_shift


def to_cartesian(b, inc, azi):
    import numpy as np
    return np.stack((b * np.sin(inc) * np.cos(azi), b * np.sin(inc) * np.sin(azi), b * np.cos(inc)), axis=-1)


def to_spherical(b):
    import numpy as np
    magnitude = np.linalg.norm(b, axis=-1)
    return magnitude, np.arccos(b[..., 2] / (magnitude + 1e-10)), np.arctan2(b[..., 1], b[..., 0])
