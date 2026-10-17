"""High-rate spherical circular truth and controlled synthetic acquisitions.

The public ME generator supplies all physics and geometry.  Exposure integration
uses midpoint quadrature of detector-frame Stokes, never averages physical
parameters, and keeps exact instantaneous midpoint truth as a separate target.
"""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import platform
import re
from pathlib import Path
from glob import escape as glob_escape

import numpy as np

def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def _environment():
    versions = {}
    for package in ('spinnme', 'numpy', 'torch', 'astropy', 'sunpy'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {'python': platform.python_version(), 'versions': versions}


def expand_acquisition_grid(specification: dict) -> dict:
    """Expand a readable cadence × noise grid into explicit acquisition variants."""
    if not isinstance(specification, dict):
        raise ValueError('Synthetic configuration must contain generation and acquisition mappings.')
    unknown = set(specification) - {'schema_version', 'generation', 'acquisition'}
    if unknown:
        raise ValueError(f'Unknown synthetic configuration keys: {sorted(unknown)}. '
                         'Use a separate inversion YAML for training settings.')
    if specification.get('schema_version', 1) != 1 or isinstance(specification.get('schema_version'), bool):
        raise ValueError('Synthetic schema_version must be 1.')
    spec = copy.deepcopy(specification)
    acquisition = spec.setdefault("acquisition", {})
    if not isinstance(acquisition, dict):
        raise ValueError('acquisition must be a mapping.')
    unknown = set(acquisition) - {'variants', 'noise_model', 'continuum_reference',
        'noise_reference_exposure_seconds', 'cadences_seconds', 'phases_seconds', 'noise_levels',
        'realization_seeds', 'exposure_seconds'}
    if unknown:
        raise ValueError(f'Unknown acquisition keys: {sorted(unknown)}.')
    if "variants" in acquisition:
        if not isinstance(acquisition['variants'], list) or not acquisition["variants"]:
            raise ValueError("acquisition.variants must be a nonempty list.")
        return spec
    cadences = acquisition.get("cadences_seconds", [])
    levels = acquisition.get("noise_levels", [])
    phases = acquisition.get("phases_seconds", [0])
    seeds = acquisition.get("realization_seeds", [20240323])
    if any(not isinstance(value, list) or not value for value in (cadences, levels, phases, seeds)):
        raise ValueError("Supply nonempty acquisition cadences, noise levels, phases and realization seeds.")
    cadences = [_finite(value, 'cadences_seconds', positive=True) for value in cadences]
    phases = [_finite(value, 'phases_seconds', nonnegative=True) for value in phases]
    if len(set(cadences)) != len(cadences) or len(set(phases)) != len(phases):
        raise ValueError("Cadences and phases must not contain duplicates.")
    _seeds(seeds, "acquisition.realization_seeds")
    variants = []
    for cadence in cadences:
        for phase in phases:
            for level in levels:
                if not isinstance(level, dict) or set(level) != {'name', 'std'}:
                    raise ValueError('Each noise level must contain name and std.')
                label = level["name"]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", str(label)):
                    raise ValueError("Noise names may contain only letters, numbers, hyphens and underscores.")
                sigma = np.asarray(level["std"], dtype=float)
                if sigma.shape == ():
                    sigma = np.repeat(sigma, 4)
                if sigma.shape != (4,) or not np.isfinite(sigma).all() or (sigma < 0).any():
                    raise ValueError("Each noise std must be a nonnegative scalar or four I/Q/U/V values.")
                variant = {"name": f"cadence{cadence:g}s-phase{phase:g}s-{label}",
                           "noise_label": label, "cadence_seconds": cadence,
                           "exposure_seconds": acquisition.get("exposure_seconds", 5),
                           "phase_seconds": phase, "noise_std": sigma.tolist(),
                           "realization_seeds": seeds[:1] if not sigma.any() else list(seeds)}
                for key in ("noise_reference_exposure_seconds", "noise_model", "continuum_reference"):
                    if key in acquisition:
                        variant[key] = acquisition[key]
                variants.append(variant)
    if len({row["name"] for row in variants}) != len(variants):
        raise ValueError("Acquisition variant names must be unique.")
    acquisition["variants"] = variants
    return spec


def _seeds(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list of distinct integer seeds.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in values):
        raise ValueError(f"{label} must contain integer seeds in [0, 2**32).")
    if len(set(values)) != len(values):
        raise ValueError(f'{label} must contain unique seeds.')

def _finite(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"Invalid {label}: {value}.")
    return value


def _bins(value, step, label, *, positive=False):
    value = _finite(value, label, positive=positive, nonnegative=not positive)
    bins = int(round(value / step))
    if not np.isclose(bins * step, value, rtol=1e-10, atol=1e-9):
        raise ValueError(f"{label} must be an integer multiple of base_step_seconds.")
    return bins


def _shape(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain two integer dimensions >= 4.")
    try:
        valid = all(not isinstance(v, bool) and int(v) == v and v >= 4 for v in value)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{label} must contain two integer dimensions >= 4.")
    return tuple(int(v) for v in value)


def _validate(spec):
    from astropy.time import Time

    if not isinstance(spec, dict) or not isinstance(spec.get("generation"), dict):
        raise ValueError("Specify generation and acquisition mappings.")
    generation = copy.deepcopy(spec["generation"])
    unknown = set(generation) - {'start_time', 'base_step_seconds', 'duration_seconds',
        'shape', 'pattern_resolution', 'evolution_seconds_per_step', 'oscillation_period_seconds',
        'oscillation_amplitude_steps', 'observer', 'wavelength_center_angstrom',
        'wavelength_offsets_angstrom', 'tile_rows', 'tile_cols', 'seed'}
    if unknown:
        raise ValueError(f'Unknown generation keys: {sorted(unknown)}.')
    if 'seed' in generation:
        _seeds([generation['seed']], 'generation.seed')
    acquisition = copy.deepcopy(spec.get("acquisition", {}))
    if not isinstance(acquisition, dict):
        raise ValueError("acquisition must be a mapping.")
    step = _finite(generation.get("base_step_seconds", 5), "base_step_seconds", positive=True)
    count = _bins(generation.get("duration_seconds", 900), step, "duration_seconds", positive=True)
    shape = _shape(generation.get("shape", [32, 32]), "shape")
    pattern = _shape(generation.get("pattern_resolution", [180, 180]), "pattern_resolution")
    evolution = _finite(generation.get("evolution_seconds_per_step", 90),
                        "evolution_seconds_per_step", positive=True)
    period = _finite(generation.get("oscillation_period_seconds", 180),
                     "oscillation_period_seconds", positive=True)
    amplitude = _finite(generation.get("oscillation_amplitude_steps", 30),
                        "oscillation_amplitude_steps", nonnegative=True)
    # The inherited analytic motif uses r0 = 50 + time_step/6.
    if amplitude >= 300:
        raise ValueError("oscillation_amplitude_steps must be < 300 to keep the circular radius positive.")
    if amplitude > 0 and step > period / 4:
        raise ValueError("base_step_seconds must resolve oscillation_period_seconds with at least four samples per period.")
    observer = generation.get("observer", {})
    if not isinstance(observer, dict):
        raise ValueError("observer must be a mapping.")
    if set(observer) - {'longitude_deg', 'latitude_deg', 'distance_au'}:
        raise ValueError('observer accepts longitude_deg, latitude_deg and distance_au.')
    longitude = _finite(observer.get("longitude_deg", 0), "observer longitude")
    latitude = _finite(observer.get("latitude_deg", 0), "observer latitude")
    distance = _finite(observer.get("distance_au", 1), "observer distance", positive=True)
    if abs(latitude) > 90 or distance <= 0.004651:
        raise ValueError("Observer must be outside the Sun with latitude in [-90, 90].")
    start = Time(generation.get("start_time", "2025-01-01T00:00:00"), scale="utc")
    continuum = _finite(acquisition.get("continuum_reference", 1), "continuum_reference", positive=True)
    if acquisition.get("noise_model", "gaussian") != "gaussian":
        raise ValueError("Only the explicitly calibrated Gaussian continuum-relative noise model is supported.")
    variants = acquisition.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("acquisition.variants must be a nonempty list.")
    normalized, names, ticks = [], set(), set(2 * i + 1 for i in range(count))
    for original in variants:
        if not isinstance(original, dict):
            raise ValueError("Each acquisition variant must be a mapping.")
        variant = copy.deepcopy(original)
        unknown = set(variant) - {'name', 'noise_label', 'noise_model', 'continuum_reference',
            'cadence_seconds', 'exposure_seconds', 'phase_seconds', 'noise_std',
            'noise_reference_exposure_seconds', 'realization_seeds'}
        if unknown:
            raise ValueError(f'Unknown acquisition variant keys: {sorted(unknown)}.')
        if variant.get("noise_model", acquisition.get("noise_model", "gaussian")) != "gaussian":
            raise ValueError("Each variant must use the global Gaussian noise_model.")
        if "continuum_reference" in variant and _finite(
                variant["continuum_reference"], "variant continuum_reference", positive=True) != continuum:
            raise ValueError("Variant continuum_reference must equal the global acquisition.continuum_reference.")
        name = str(variant.get("name", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError("Variant names must be unique, nonempty, path-safe identifiers.")
        names.add(name)
        label = variant.get('noise_label', name)
        if not isinstance(label, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label):
            raise ValueError('noise_label must be a nonempty, path-safe identifier.')
        cadence = _bins(variant.get("cadence_seconds"), step, "cadence_seconds", positive=True)
        exposure = _bins(variant.get("exposure_seconds"), step, "exposure_seconds", positive=True)
        phase = _bins(variant.get("phase_seconds", 0), step, "phase_seconds")
        if exposure > cadence:
            raise ValueError("exposure_seconds cannot exceed cadence_seconds for a single acquisition stream.")
        windows = [(begin, begin + exposure) for begin in range(phase, count - exposure + 1, cadence)]
        if not windows:
            raise ValueError(f"Variant {name!r} has no complete exposure window.")
        sigma = np.asarray(variant.get("noise_std", [0, 0, 0, 0]), dtype=float)
        if sigma.shape != (4,) or not np.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError("noise_std must contain four finite nonnegative I,Q,U,V values.")
        reference_exposure = variant.get("noise_reference_exposure_seconds",
                                         acquisition.get("noise_reference_exposure_seconds"))
        factor = 1.0
        if reference_exposure is not None:
            reference_exposure = _finite(reference_exposure, "noise_reference_exposure_seconds", positive=True)
            factor = math.sqrt(reference_exposure / (exposure * step))
        seeds = variant.get("realization_seeds", [0])
        _seeds(seeds, 'realization_seeds')
        ticks.update(begin + end for begin, end in windows)
        normalized.append({**variant, "name": name, 'noise_label': label, "cadence_seconds": cadence * step,
                           "exposure_seconds": exposure * step, "phase_seconds": phase * step,
                           "noise_std": sigma.tolist(), "effective_noise_std": (sigma * factor).tolist(),
                           "noise_reference_exposure_seconds": reference_exposure,
                           "realization_seeds": seeds, "_windows": windows})
    return {"generation": generation, "step": step, "count": count, "shape": shape,
            "pattern": pattern, "evolution": evolution, "period": period,
            "amplitude": amplitude, "observer": (longitude, latitude, distance), "start": start,
            "continuum": continuum, "variants": normalized, "ticks": sorted(ticks)}


def _observer(config, seconds):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    longitude, latitude, distance = config["observer"]
    return SkyCoord(lon=longitude*u.deg, lat=latitude*u.deg, radius=distance*u.AU,
                    obstime=config["start"] + seconds*u.s, frame=frames.HeliographicStonyhurst)


def _header(config, seconds, center, offsets, *, exposure=0, begin=None, end=None):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from sunpy.coordinates import frames
    from sunpy.map import make_fitswcs_header
    from sunpy.sun import constants

    observer = _observer(config, seconds)
    radius = np.arcsin(constants.radius / observer.radius).to(u.arcsec)
    shape = config["shape"]
    reference = SkyCoord(0*u.arcsec, 0*u.arcsec, observer=observer,
                         frame=frames.Helioprojective)
    scale = u.Quantity([radius / (shape[1] / 2 * u.pix), radius / (shape[0] / 2 * u.pix)])
    header = fits.Header(dict(make_fitswcs_header(np.zeros(shape), reference, scale=scale)))
    carrington = observer.transform_to(frames.HeliographicCarrington(observer=observer))
    header["CRLN_OBS"] = float(carrington.lon.to_value(u.deg))
    header["CRLT_OBS"] = float(carrington.lat.to_value(u.deg))
    midpoint = observer.obstime.utc.isot
    header["DATE-OBS"] = midpoint
    header["DATE-AVG"] = midpoint
    if begin is not None:
        header["DATE-BEG"] = (config["start"] + begin*u.s).utc.isot
    if end is not None:
        header["DATE-END"] = (config["start"] + end*u.s).utc.isot
    header["TIMESYS"] = "UTC"
    header["EXPTIME"] = exposure
    header["WAVELNTH"] = center
    header["WAVEUNIT"] = "Angstrom"
    header["BUNIT"] = "1"
    header["INSTRUME"] = "SPINNME-CIRCULAR"
    header["TELESCOP"] = "SYNTHETIC"
    header["ORIGIN"] = "SPINNME"
    for key in ("OBS_VR", "OBS_VW", "OBS_VN"):
        header[key] = 0.0
    for i, offset in enumerate(offsets):
        header[f"WAVELN{i+1:02d}"] = float(center + offset)
    return header


def _write_cube(path, cube, mask, header):
    from astropy.io import fits

    path.parent.mkdir(parents=True, exist_ok=True)
    # Spatial WCS lives in axes 1,2; arbitrary spectral sampling is explicit in
    # WAVELNnn and WAVELENGTH, avoiding a fictitious linear wavelength WCS.
    cube_header = header.copy()
    cube_header["WCSAXES"] = 2
    cube_header["SPNAXES"] = "stokes,wavelength,y,x"
    cube_header["SPNSTOKS"] = "IQUV"
    wave = np.asarray([header[f"WAVELN{i+1:02d}"] for i in range(cube.shape[1])], dtype=np.float64)
    wave_hdu = fits.ImageHDU(wave, name="WAVELENGTH")
    wave_hdu.header["BUNIT"] = "Angstrom"
    fits.HDUList([fits.PrimaryHDU(np.asarray(cube, dtype=np.float32), cube_header),
                  fits.ImageHDU(np.asarray(mask, dtype=np.uint8), header, name="VALID_MASK"),
                  wave_hdu]).writeto(path, overwrite=False, checksum=True)


def _canonical_truth(parameters, profiles, mask, header, center, offsets, time):
    from astropy import units as u
    from sunpy.coordinates import frames
    from sunpy.map import Map, all_coordinates_from_map
    from spinnme.data.differential_rotation import carrington_rotation_velocity

    arrays = {name: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
              for name, value in parameters.items()}
    b = arrays["b_field"]
    result = {"B": b, "B_los": b * arrays["cos_inc"],
              "B_transverse": b * np.sqrt(np.maximum(arrays["sin_inc2"], 0)),
              "v_los": arrays["vdop"], "inclination": np.rad2deg(arrays["inc"]),
              "azimuth": np.mod(np.rad2deg(arrays["azi"]), 180), "mask": mask}
    result.update({name: arrays["b_rtp"][..., i] for i, name in enumerate(("Br", "Btheta", "Bphi"))})
    # The analytic generator stores inertial velocity, while the public scalar
    # vr/vtheta/vphi exports are residuals in the rotating Carrington frame.
    latitude = all_coordinates_from_map(Map(np.zeros(mask.shape), header)).transform_to(
        frames.HeliographicCarrington).lat.to_value(u.rad)
    velocity = arrays['v_rtp'].copy()
    velocity[..., 2] -= carrington_rotation_velocity(latitude, header.get('RSUN_REF', u.Rsun.to(u.m)), f=np)
    result.update({name: velocity[..., i] for i, name in enumerate(("vr", "vtheta", "vphi"))})
    result.update({name: arrays[name] for name in ("vmac", "damping", "b0", "b1", "kl", "mu")})
    result.update({component: profiles[..., i, :] for i, component in enumerate("IQUV")})
    units = {name: ("G" if name.startswith("B") else "m / s" if name in
                        ("v_los", "vr", "vtheta", "vphi", "vmac") else
                        "deg" if name in ("inclination", "azimuth") else "1") for name in result if name != "mask"}
    metadata = {"schema_version": 1, "kind": "instantaneous-spherical-circular-truth",
                "units": units, "header": dict(header), "model_time": time,
                "wavelength_center_m": center * 1e-10,
                "wavelength_grid_m": (np.asarray(offsets) * 1e-10).tolist(),
                "conventions": {"B_los": "positive toward observer", "v_los": "positive redshift",
                                "Btheta": "increasing colatitude", "azimuth": "modulo 180 degrees",
                                "velocity_spherical": "Carrington residual; v_los remains inertial positive redshift"}}
    result["metadata"] = np.asarray(json.dumps(metadata))
    return result


def _noise(seed, center_tick, exposure_bins, shape):
    # A stable acquisition key gives paired standard-normal deviates across
    # noise levels and does not depend on ordering/naming of variants.
    key = f"spinnme-circular-noise-v1:{seed}:{center_tick}:{exposure_bins}".encode()
    entropy = np.frombuffer(hashlib.sha256(key).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(entropy)).standard_normal(shape)


def _write_acquisitions(path, acquisitions):
    """Write one shell-readable row per generated input stream."""
    columns = ('RUN_ID', 'CADENCE_SECONDS', 'NOISE_LABEL', 'NOISE_SEED', 'INPUT_GLOB')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', lineterminator='\n',
                                quoting=csv.QUOTE_NONE, quotechar=None)
        writer.writeheader()
        for acquisition in acquisitions:
            for realization in acquisition['realizations']:
                writer.writerow({
                    'RUN_ID': f"{acquisition['name']}-noise{realization['seed']}",
                    'CADENCE_SECONDS': f"{acquisition['cadence_seconds']:g}",
                    'NOISE_LABEL': acquisition['noise_label'], 'NOISE_SEED': realization['seed'],
                    'INPUT_GLOB': glob_escape(str(path.parent / realization['path'])) + '/frame_*.fits',
                })


def generate_dataset(spec: dict, output: str | Path) -> Path:
    """Generate truth once, then exact-window acquisitions with seeded noise.

    ``spec`` contains ``generation`` and ``acquisition`` settings. Acquisition
    variants can be explicit or expanded from cadence, phase and noise lists.
    All manifest paths are relative to the returned ``dataset.json`` parent.
    ``acquisitions.tsv`` provides absolute input globs for shell run loops.
    An existing output directory is rejected, including incomplete previous
    generation attempts; choose a new directory to retain their generation records.
    """
    config = _validate(expand_acquisition_grid(spec))
    output = Path(output).expanduser().resolve()
    if '\t' in str(output) or '\n' in str(output) or '\r' in str(output):
        raise ValueError('Synthetic output paths cannot contain tabs or newlines.')
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite synthetic dataset directory: {output}")

    from astropy import units as u
    from astropy.io import fits
    from spinnme.data.create_spherical_test_set import SphericalTestSetGenerator
    from spinnme.instrument import hmi_wavelength_config

    wavelength = hmi_wavelength_config()
    center = _finite(config["generation"].get("wavelength_center_angstrom",
                     wavelength["wavelength_center"].to_value(u.AA)), "wavelength_center_angstrom", positive=True)
    offsets = np.asarray(config["generation"].get("wavelength_offsets_angstrom",
                         wavelength["wavelength_grid"].to_value(u.AA)), dtype=float)
    if (offsets.ndim != 1 or len(offsets) < 2 or not np.isfinite(offsets).all()
            or len(np.unique(offsets)) != len(offsets) or (center + offsets <= 0).any()):
        raise ValueError("wavelength_offsets_angstrom must contain unique finite spectral samples.")
    tile_rows = config["generation"].get("tile_rows", 2)
    tile_cols = config["generation"].get("tile_cols", 4)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (tile_rows, tile_cols)):
        raise ValueError("tile_rows and tile_cols must be positive integers.")
    generator = SphericalTestSetGenerator(nx=config["shape"][0], ny=config["shape"][1],
                  wavelength_center=center*u.AA, wavelength_grid=offsets*u.AA,
                  pattern_resolution=config["pattern"], tile_rows=tile_rows, tile_cols=tile_cols)
    output.mkdir(parents=True)
    manifest = {"schema_version": 1, "kind": "spherical-circular-noise-cadence-study",
                "created_utc": _utc_now(), "environment": _environment(), "spec": copy.deepcopy(spec),
                "specification": copy.deepcopy(spec),
                "specification_sha256": _canonical_hash(spec), "path_base": "dataset.json parent",
                'acquisition_table': 'acquisitions.tsv',
                "axes": ["stokes", "wavelength", "y", "x"], "stokes_order": "IQUV",
                "support_start_seconds": 0., "support_end_seconds": config["count"] * config["step"],
                "wavelength_center_m": center*1e-10, "wavelength_grid_m": (offsets*1e-10).tolist(),
                "time_semantics": "UTC DATE-OBS and DATE-AVG are exposure midpoint; phase is exposure-window start offset.",
                "exposure_integration": "Arithmetic mean of detector-frame Stokes at base-bin midpoints (midpoint quadrature); complete windows only.",
                "physical_truth": "Exact instantaneous parameters at each listed epoch; no physical parameter or vector averaging.",
                "noise": {"model": "independent Gaussian I/Q/U/V, pixels and wavelengths",
                          "continuum_reference": config["continuum"],
                          "std_convention": "noise_std is relative to the explicit scalar continuum_reference; effective_noise_std includes optional sqrt(reference_exposure/exposure).",
                          "seed_policy": "SHA256(spinnme-circular-noise-v1:seed:midpoint_halfstep_tick:exposure_base_bins) seeds NumPy PCG64. Equal seed/time/exposure share standard-normal deviates across noise levels.",
                          "scope": "Controlled Gaussian approximation; no mission-specific detector or modulation calibration implied."},
                "dense_truth": {"start_time": config["start"].utc.isot,
                                "base_step_seconds": config["step"], "base_frame_count": config["count"],
                                "frames": []}, "acquisitions": []}
    tick_records = {}
    for index, tick in enumerate(config["ticks"]):
        seconds = tick * config["step"] / 2
        physical_step = seconds / config["evolution"] + config["amplitude"] * np.sin(2*np.pi*seconds/config["period"])
        profiles, parameters, _ = generator.load_spherical_time_step(physical_step, _observer(config, seconds))
        mask = np.isfinite(profiles).all(axis=(-2, -1))
        header = _header(config, seconds, center, offsets)
        time = str(header["DATE-OBS"])
        reference = Path("truth") / f"frame_{index:06d}.fits"
        truth = Path("truth") / f"frame_{index:06d}.npz"
        _write_cube(output/reference, profiles.transpose(2, 3, 0, 1), mask, header)
        np.savez_compressed(output/truth, **_canonical_truth(parameters, profiles, mask, header,
                                                          center, offsets, time))
        base_grid = tick % 2 == 1
        record = {"index": index, "time_seconds": seconds, "time": time,
                  "base_grid": base_grid, "kind": "base" if base_grid else "acquisition_midpoint",
                  "evolution_time_step": float(physical_step), "reference_fits": str(reference),
                  "truth_npz": str(truth), "reference_fits_sha256": _checksum(output/reference),
                  "truth_npz_sha256": _checksum(output/truth)}
        record["sha256"], record["truth_sha256"] = record["reference_fits_sha256"], record["truth_npz_sha256"]
        tick_records[tick] = record
        manifest["dense_truth"]["frames"].append(record)
    for variant in config["variants"]:
        windows = variant.pop("_windows")
        acquisition = {**variant, "realizations": []}
        frame_records = []
        for index, (begin, end) in enumerate(windows):
            tick = begin + end
            seconds = tick * config["step"] / 2
            latent = [tick_records[2*i+1] for i in range(begin, end)]
            total, mask = None, None
            for row in latent:
                cube = np.asarray(fits.getdata(output/row["reference_fits"], 0), dtype=np.float64)
                current_mask = np.isfinite(cube).all(axis=(0, 1))
                total = cube if total is None else total + cube
                mask = current_mask if mask is None else mask & current_mask
            average = np.asarray(total / (end - begin), dtype=np.float32)
            header = _header(config, seconds, center, offsets, exposure=variant["exposure_seconds"],
                             begin=begin*config["step"], end=end*config["step"])
            clean_path = Path("acquisitions") / variant["name"] / "noiseless" / f"frame_{index:06d}.fits"
            _write_cube(output/clean_path, average, mask, header)
            midpoint = tick_records[tick]
            frame_records.append({"index": index, "time_seconds": seconds, "time": str(header["DATE-OBS"]),
                                  "window_start_seconds": begin*config["step"],
                                  "window_end_seconds": end*config["step"],
                                  "latent_indices": [row["index"] for row in latent],
                                  "base_bin_indices": list(range(begin, end)),
                                  "reference_fits": midpoint["reference_fits"],
                                  "truth_npz": midpoint["truth_npz"],
                                  "truth_stokes_path": str(clean_path),
                                  "truth_stokes_sha256": _checksum(output/clean_path)})
        for seed in variant["realization_seeds"]:
            folder = "observed" if len(variant["realization_seeds"]) == 1 else f"seed_{seed}"
            realization = {"seed": seed, "path": str(Path("acquisitions")/variant["name"]/folder), "frames": []}
            for record in frame_records:
                clean = output/record["truth_stokes_path"]
                with fits.open(clean, memmap=False) as hdus:
                    average, header = hdus[0].data.copy(), hdus[0].header.copy()
                    mask = hdus["VALID_MASK"].data.astype(bool)
                exposure_bins = len(record["base_bin_indices"])
                tick = int(round(record["time_seconds"] / config["step"] * 2))
                normal = _noise(seed, tick, exposure_bins, average.shape)
                sigma = np.asarray(variant["effective_noise_std"])[:, None, None, None] * config["continuum"]
                noisy = np.asarray(average + normal * sigma, dtype=np.float32)
                noisy[..., ~mask] = np.nan
                header["SPNSEED"] = seed
                for component, value in zip("IQUV", variant["effective_noise_std"]):
                    header[f"NOISE_{component}"] = float(value*config["continuum"])
                path = Path(realization["path"]) / f"frame_{record['index']:06d}.fits"
                _write_cube(output/path, noisy, mask, header)
                realization["frames"].append({**record, "path": str(path), "sha256": _checksum(output/path)})
            acquisition["realizations"].append(realization)
        manifest["acquisitions"].append(acquisition)
    manifest_path = output/"dataset.json"
    _write_acquisitions(output/'acquisitions.tsv', manifest['acquisitions'])
    _write_json(manifest_path, manifest)
    return manifest_path
