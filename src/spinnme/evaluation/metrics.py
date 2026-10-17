"""Array comparison metrics independent of instruments, files and plot layouts."""
from __future__ import annotations

import numpy as np


def broadcast_mask(mask: np.ndarray, shape: tuple) -> np.ndarray:
    """Extend a public spatial mask over trailing Stokes/wavelength axes."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == shape:
        return mask
    if mask.ndim < len(shape) and mask.shape == shape[:mask.ndim]:
        return np.broadcast_to(mask.reshape(*mask.shape, *((1,) * (len(shape)-mask.ndim))), shape)
    raise ValueError(f"Mask shape {mask.shape} does not match parameter shape {shape}.")


def residual(predicted: np.ndarray, reference: np.ndarray,
             period: float | None = None) -> np.ndarray:
    difference = np.asarray(predicted, dtype=float) - np.asarray(reference, dtype=float)
    return difference if period is None else (difference + period / 2) % period - period / 2


def comparison(predicted: np.ndarray, reference: np.ndarray,
               mask: np.ndarray | None = None, *, period: float | None = None) -> tuple[dict, np.ndarray]:
    predicted, reference = np.asarray(predicted), np.asarray(reference)
    if predicted.shape != reference.shape:
        raise ValueError(f"Prediction shape {predicted.shape} differs from reference {reference.shape}.")
    if np.iscomplexobj(predicted) or np.iscomplexobj(reference):
        raise ValueError("Physical parameter comparisons require real arrays.")
    valid = np.isfinite(predicted) & np.isfinite(reference)
    if mask is not None:
        valid &= broadcast_mask(mask, valid.shape)
    if period is not None and (not np.isfinite(period) or period <= 0):
        raise ValueError("Angular period must be finite and positive.")
    differences = np.full(predicted.shape, np.nan, dtype=float)
    differences[valid] = residual(predicted[valid], reference[valid], period)
    count = int(valid.sum())
    if not count:
        raise ValueError("No finite paired samples remain after masking.")
    values = differences[valid]
    a, b = predicted[valid], reference[valid]
    correlation = (float(np.corrcoef(a, b)[0, 1])
                   if period is None and count > 1 and np.std(a) > 0 and np.std(b) > 0 else None)
    return {"n": count, "bias": float(np.mean(values)), "mae": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "median_absolute_error": float(np.median(np.abs(values))),
            "p95_absolute_error": float(np.percentile(np.abs(values), 95)),
            "pearson_r": correlation, "angular_period": period}, differences
