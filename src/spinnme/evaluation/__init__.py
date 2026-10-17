"""Standard scalar-map summaries and residual metrics, independent of paper studies."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np

from .metrics import broadcast_mask, comparison, residual

__all__ = ["comparison", "residual", "evaluate_result", "read_reference", "write_metrics"]


def read_reference(path, result):
    from ..exports import read_npz, EvaluationResult
    path = Path(path)
    if path.suffix.lower() == ".npz":
        reference = read_npz(path)
    else:
        from astropy.io import fits
        from astropy import units as u
        if len(result.arrays) != 1:
            raise ValueError("A single reference FITS requires one selected parameter; use NPZ for several maps.")
        name = next(iter(result.arrays))
        with fits.open(path, memmap=False) as hdus:
            header = hdus[0].header.copy()
            values = np.asarray(hdus[0].data)
            reference_mask = (np.asarray(hdus['VALID_MASK'].data, dtype=bool)
                              if 'VALID_MASK' in hdus else np.ones(values.shape, dtype=bool))
        if 'BUNIT' not in header:
            raise ValueError('Reference FITS requires an explicit BUNIT; units must not be guessed.')
        values = (values * u.Unit(header['BUNIT'])).to_value(u.Unit(result.units[name]))
        reference = EvaluationResult({name: values}, {name: result.units[name]}, header, reference_mask,
                                     {'model_time': header.get('SPINNME MODEL_TIME', header.get('DATE-OBS'))})
    from astropy.wcs import WCS
    if reference.mask.shape != result.mask.shape:
        raise ValueError("Reference and prediction grids differ; align explicitly before evaluation.")
    y, x = np.indices(result.mask.shape)
    first = WCS(result.header).celestial.pixel_to_world_values(x, y)
    second = WCS(reference.header).celestial.pixel_to_world_values(x, y)
    if not all(np.allclose(a, b, rtol=0, atol=1e-8, equal_nan=True) for a, b in zip(first, second)):
        raise ValueError("Reference WCS differs; align explicitly before evaluation.")
    from astropy.time import Time
    from astropy import units as u
    from sunpy.map import Map
    first_map = Map(np.zeros((1, 1)), result.header)
    second_map = Map(np.zeros((1, 1)), reference.header)
    first_observer = first_map.observer_coordinate.cartesian.xyz.to_value(u.m)
    second_observer = second_map.observer_coordinate.cartesian.xyz.to_value(u.m)
    if not np.allclose(first_observer, second_observer, rtol=1e-8, atol=1000):
        raise ValueError('Reference observer geometry differs; align to a common physical observing grid first.')
    first_time = result.metadata.get('model_time', result.header.get('DATE-OBS'))
    second_time = reference.metadata.get('model_time', reference.header.get('DATE-OBS'))
    if first_time and second_time and abs((Time(first_time) - Time(second_time)).to_value(u.s)) > 1e-3:
        raise ValueError('Reference model epoch differs; select a common physical evaluation time first.')
    return reference


def evaluate_result(result, reference=None):
    """Return finite summaries and optional bias/MAE/RMSE/angular residual metrics."""
    rows = []
    for name, value in result.arrays.items():
        value = np.asarray(value)
        valid = broadcast_mask(result.mask, value.shape).copy()
        valid &= np.isfinite(value)
        row = {"parameter": name, "unit": result.units.get(name, "1"), "comparison": reference is not None}
        if reference is not None:
            if name not in reference.arrays:
                raise ValueError(f"Reference has no {name} array.")
            if reference.units.get(name) != result.units.get(name):
                raise ValueError(f"Reference units differ for {name}; convert explicitly before comparison.")
            target = np.asarray(reference.arrays[name])
            if target.shape != value.shape:
                raise ValueError(f"Reference shape differs for {name}.")
            valid &= broadcast_mask(reference.mask, value.shape)
            valid &= np.isfinite(target)
        row["count"] = int(valid.sum())
        if not valid.any():
            raise ValueError(f"No common finite pixels for {name}.")
        selected = value[valid].astype(np.float64)
        row.update(mean=float(selected.mean()), std=float(selected.std()), minimum=float(selected.min()), maximum=float(selected.max()))
        if reference is not None:
            truth = target[valid].astype(np.float64)
            metric, _ = comparison(selected, truth, period=180 if name == 'azimuth' else None)
            row.update({key: value for key, value in metric.items() if key not in ('n', 'pearson_r')})
            row['correlation'] = metric['pearson_r']
        rows.append(row)
    return {"metadata": result.metadata, "metrics": rows, "reference": None if reference is None else reference.metadata}


def write_metrics(metrics, output):
    path = Path(output)
    if path.suffix.lower() != ".json":
        path = path / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False))
    csv_path = path.with_suffix(".csv")
    rows = metrics["metrics"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    return [path, csv_path]
