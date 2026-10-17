"""Versioned, tensor-only inference snapshots with portable observation metadata."""
from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from dataclasses import fields, is_dataclass
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.io import fits
from astropy.time import Time

from . import __version__
from .configuration import config_hash
from .synthesis import SphericalSynthesis

FORMAT_VERSION = 1


def encode(value):
    """Convert scientific metadata into a JSON-safe representation."""
    if isinstance(value, u.Quantity):
        return {"__quantity__": np.asarray(value.value).tolist(), "unit": str(value.unit)}
    if isinstance(value, (datetime, Time)):
        return {"__datetime__": Time(value).utc.isot}
    if isinstance(value, fits.Header):
        return {"__fits_header__": value.tostring(sep="\n", endcard=False, padding=False)}
    if is_dataclass(value):
        return {"__domain_type__": type(value).__name__, "fields": {field.name: encode(getattr(value, field.name)) for field in fields(value)}}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported snapshot metadata type: {type(value).__name__}")


def decode(value):
    if isinstance(value, dict):
        if "__quantity__" in value:
            return np.asarray(value["__quantity__"]) * u.Unit(value["unit"])
        if "__datetime__" in value:
            return Time(value["__datetime__"], scale="utc").to_datetime()
        if "__fits_header__" in value:
            return fits.Header.fromstring(value["__fits_header__"], sep="\n")
        if "__domain_type__" in value:
            from .train.collocation import LongitudeInterval, SphericalCollocationDomain
            domain_types = {"LongitudeInterval": LongitudeInterval, "SphericalCollocationDomain": SphericalCollocationDomain}
            return domain_types[value["__domain_type__"]](**decode(value["fields"]))
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def _paths(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _paths(v)
    elif isinstance(value, (tuple, list)):
        for v in value:
            yield from _paths(v)
    elif isinstance(value, (str, Path)):
        yield str(value)


def _replace_assets(value, mapping):
    if isinstance(value, dict):
        return {k: _replace_assets(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_assets(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def save_snapshot(me_module, data_module, config, path):
    """Atomically save weights, reconstructible configuration, grids, and calibration assets."""
    init = getattr(me_module, "snapshot_init", None)
    if init is None:
        init = {
            "image_shape": data_module.image_shape,
            "wavelength_config": data_module.wavelength_config,
            "gauss_per_dB": data_module.gauss_per_dB,
            "Rs_per_ds": data_module.Rs_per_ds,
            "seconds_per_dt": data_module.seconds_per_dt,
            "instrument_config": config["instrument"],
            "model_config": config["model"],
            "normalization_config": config.get("normalization", {}),
            "stokes_loss_config": config.get("stokes_loss", {}),
            "physics_config": config.get("physics"),
            "weights_config": config.get("weights", {}),
        }
    observations = (data_module.get_observations() if hasattr(data_module, "get_observations")
                    else getattr(data_module, "observations", []))
    response_mapping = getattr(data_module, "spectral_response_by_acquisition", {})
    response_files = getattr(data_module, "spectral_response_files", {})
    assets, replacements = {}, {}
    for source in sorted(set(_paths(response_mapping)) | set(_paths(response_files))):
        file = Path(source)
        if not file.is_file():
            raise FileNotFoundError(f"Cannot make portable snapshot: missing spectral response {source}")
        raw = file.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        name = digest + file.suffix
        assets[name] = {"sha256": digest, "data": torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())}
        replacements[source] = "asset:" + name
    metadata = {
        "init": init,
        "config": config,
        "config_hash": config_hash(config),
        "coordinate_order": "xyzt",
        "ref_time": data_module.ref_time,
        "times": data_module.times,
        "observations": observations,
        "spectral_response_files": response_files,
        "spectral_response_by_acquisition": response_mapping,
        "stokes_normalization_by_instrument": getattr(data_module, "stokes_normalization_by_instrument", {}),
        "data_cache_signature": getattr(data_module, "cache_signature", None),
        "conventions": {"Btheta": "increasing colatitude (southward)", "Bphi": "increasing Carrington longitude",
                        "B_los": "positive toward observer", "v_los": "positive redshift; includes rotation; observer correction excluded",
                        "ambiguity": "unresolved model azimuth branch", "velocity_frame": "Carrington for vr/vtheta/vphi"},
    }
    metadata = _replace_assets(encode(metadata), replacements)
    payload = {"format": "spinnme", "format_version": FORMAT_VERSION, "spinnme_version": __version__,
               "metadata_json": json.dumps(metadata, allow_nan=False),
               "state_dict": {k: v.detach().cpu() for k, v in me_module.state_dict().items()}, "assets": assets}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


class Snapshot(SphericalSynthesis):
    """Load a portable SPINNME snapshot and evaluate saved or supplied solar grids."""

    def __init__(self, path, *, device=None):
        self.path = Path(path)
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                   if device is None or str(device) == "auto" else device)
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("Not a tensor-only SPINNME snapshot.") from error
        if not isinstance(payload, dict):
            raise ValueError("Unsupported .pme payload; expected a versioned SPINNME mapping.")
        if payload.get("format") != "spinnme" or payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported .pme format/version: {payload.get('format_version')!r}")
        self.metadata = decode(json.loads(payload["metadata_json"]))
        if self.metadata.get('coordinate_order') != 'xyzt':
            raise ValueError('Snapshot coordinates must use xyzt order.')
        self._asset_directory = tempfile.TemporaryDirectory(prefix="spinnme-response-")
        try:
            replacements = {}
            for name, asset in payload.get("assets", {}).items():
                if Path(name).name != name:
                    raise ValueError("Invalid embedded response asset name.")
                raw = asset["data"].numpy().tobytes()
                if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                    raise ValueError(f"Corrupt embedded response asset {name}")
                destination = Path(self._asset_directory.name) / name
                destination.write_bytes(raw)
                replacements["asset:" + name] = str(destination)
            self.metadata = _replace_assets(self.metadata, replacements)
            from .train.me_spherical_module import MESphericalModule
            init = dict(self.metadata["init"])
            # Inference remains eager, independent of the training compiler backend.
            init["compile_config"] = {"me_solver": False}
            self.module = MESphericalModule(**init)
            self.module.load_state_dict(payload["state_dict"], strict=True)
            self.module.to(self.device).eval()
            for name in ("parameter_model", "forward_models", "velocity_correction_models", "limb_correction_models", "artifact_correction_models", "normalization"):
                setattr(self, name, getattr(self.module, name))
            self.instrument_ids = list(self.forward_models)
            self.instrument_id = self.instrument_ids[0] if len(self.instrument_ids) == 1 else None
            self.wavelength_config = init["wavelength_config"]
            self.ref_time = self.metadata["ref_time"]
            self.seconds_per_dt = init["seconds_per_dt"]
            self.Rs_per_ds = init["Rs_per_ds"]
            self.gauss_per_dB = init["gauss_per_dB"]
            self.meters_per_ds = self.Rs_per_ds * u.Rsun.to(u.m)
            self.stokes_loss_config = init.get("stokes_loss_config") or {}
            self.continuum_normalization_mode = "per_profile" if self.stokes_loss_config.get("continuum_normalization") else None
            self.stokes_normalization_by_instrument = self.metadata.get("stokes_normalization_by_instrument", {})
            self.spectral_response_files = self.metadata.get("spectral_response_files", {})
            self.spectral_response_by_acquisition = self.metadata.get("spectral_response_by_acquisition", {})
            self.observations = self.metadata.get("observations", [])
            self.config = self.metadata['config']
            self.times = self.metadata['times']
        except Exception:
            self.close()
            raise

    def _normalize_time(self, time):
        return (Time(time) - Time(self.ref_time)).to_value(u.s) / self.seconds_per_dt

    def _observation(self, observation, instrument_id):
        if observation is None:
            if len(self.observations) != 1:
                raise ValueError("Select --observation ID or --all-observations; use inspect to list saved grids.")
            return deepcopy(self.observations[0])
        if isinstance(observation, dict):
            return deepcopy(observation)
        from sunpy.map import GenericMap
        if isinstance(observation, GenericMap):
            row = {'id': 'supplied-map', 'instrument_id': instrument_id or self.instrument_id,
                   'shape': list(observation.data.shape), 'header': dict(observation.meta),
                   'time': {'model_time': observation.meta.get('SPINNME MODEL_TIME')}}
            if observation.mask is not None:
                import base64
                row['valid_mask_bits'] = base64.b64encode(np.packbits(~observation.mask).tobytes()).decode()
            return row
        for row in self.observations:
            if row["id"] == str(observation):
                return deepcopy(row)
        path = Path(observation)
        if not path.is_file():
            raise ValueError(f"Unknown observation {observation!r}. Available IDs: {[o['id'] for o in self.observations]}")
        with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdus:
            hdu = next(h for h in hdus if h.header.get('NAXIS', 0) >= 2)
            shape = (hdu.header['NAXIS2'], hdu.header['NAXIS1'])
            header = hdus[0].header.copy()
            header.update(hdu.header)
            extra = {}
            if 'VALID_MASK' in hdus:
                import base64
                mask = np.asarray(hdus['VALID_MASK'].data, dtype=bool)
                if mask.shape != shape:
                    raise ValueError('Reference FITS validity mask does not match its spatial grid.')
                extra['valid_mask_bits'] = base64.b64encode(np.packbits(mask).tobytes()).decode()
            if 'TIME_OFFSETS' in hdus:
                extra.update(time_offsets_seconds=np.broadcast_to(hdus['TIME_OFFSETS'].data, shape).tolist(),
                             time_offset_axis='yx')
        from astropy.wcs import WCS
        header.update(WCS(header).celestial.to_header(relax=True))
        selected_id = instrument_id or self.instrument_id
        instrument_types = {item['instrument_id']: item['type'] for item in self.metadata['init']['instrument_config']}
        originals = {key: header[key] for key in ('DATE-OBS', 'DATE-AVG', 'T_OBS', 'TIMESYS') if key in header}
        row = {"id": path.stem, "instrument_id": selected_id,
                "shape": list(shape), "header": dict(header),
                "time": {"model_time": header.get('SPINNME MODEL_TIME'), "original": originals},
                **extra}
        if instrument_types.get(selected_id) == 'hmi':
            from .data.hmi_transmission import read_hmi_fits_acquisition
            try:
                acquisition = read_hmi_fits_acquisition(path)
            except (ValueError, KeyError):
                acquisition = {}
            row['acquisition_key'] = acquisition.get('acquisition_key')
        return row

    def grid(self, observation=None, *, shape=None, bounds=None, instrument_id=None):
        """Return a reusable observing grid from a saved ID, FITS, Map or dict.

        ``shape`` is (ny,nx); ``bounds`` is xmin,ymin,xmax,ymax in arcsec.
        Support masks, raster timing and detector calibration follow the grid.
        """
        from .grids import prepare_grid
        row = self._observation(observation, instrument_id)
        native_shape = row.get('shape', ())
        if len(native_shape) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1 for v in native_shape):
            raise ValueError('Observation shape must contain two positive integer dimensions in (ny,nx) order.')
        if not isinstance(row.get('time', {}), dict):
            row['time'] = {'model_time': row['time']}
        selected_id = instrument_id or row.get('instrument_id') or self.instrument_id
        self._instrument(selected_id)
        row['instrument_id'] = selected_id
        from sunpy.map import Map

        from .grids import spatial_header
        header = spatial_header(row['header'])
        types = {item['instrument_id']: item['type'] for item in self.metadata['init']['instrument_config']}
        is_hmi = types.get(selected_id) == 'hmi' or 'HMI' in str(header.get('INSTRUME', '')).upper()
        if is_hmi:
            from .data.fits import parse_time
            if not header.get('DATE-OBS'):
                raise ValueError('HMI grid requires DATE-OBS, matching training coordinates.')
            original = {key: header[key] for key in ('DATE-OBS', 'DATE-AVG', 'T_OBS', 'TIMESYS') if key in header}
            header.remove('T_OBS', ignore_missing=True)
            header.remove('DATE-AVG', ignore_missing=True)
            row['time'] = {**(row.get('time') or {}),
                           'model_time': row.get('time', {}).get('model_time') or
                                         parse_time(header['DATE-OBS'], header.get('TIMESYS', 'utc').lower()).isoformat(),
                           'original': row.get('time', {}).get('original', original)}
        row['header'] = dict(header)
        if types.get(selected_id) == 'hmi' and row.get('detector_mapping') is None:
            from .loader.spherical import hmi_detector_mapping
            map_ = Map(np.broadcast_to(np.zeros((1, 1), dtype=np.uint8), row['shape']), header)
            try:
                origin, step = hmi_detector_mapping(map_)
            except ValueError:
                pass
            else:
                row.update(detector_mapping=[origin.tolist(), step.tolist()], pix_convention='detector')
        return prepare_grid(row, shape=shape, bounds=bounds)

    def geometry(self, observation=None, *, indices=None, time=None, instrument_id=None):
        """Return model xyzt coordinates and projection matrices for grid pixels.

        Arrays have a leading flat pixel axis; ``indices`` selects flat y,x IDs
        to keep full-disk geometry bounded in memory. Units/scales match training.
        """
        from .grids import GridGeometry
        row = self.grid(observation, instrument_id=instrument_id)
        context = GridGeometry(row)
        info = row.get('time', {})
        epoch = time if time is not None else info.get('model_time') if isinstance(info, dict) else info
        epoch = context.map.date if epoch is None else epoch
        if indices is None:
            indices = np.arange(np.prod(row['shape']))
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= np.prod(row['shape'])):
            raise ValueError('indices must be a one-dimensional array of valid flat pixel IDs.')
        return context.pixels(indices, Rs_per_ds=self.Rs_per_ds, normalized_time=self._normalize_time(epoch),
                              seconds_per_dt=self.seconds_per_dt, scan_times=time is None)

    def cadence_times(self, cadence_seconds, *, instrument_id=None):
        """Return regular UTC epochs within the selected instrument's training span.

        Endpoints include raster slit offsets. Staying within the overall span does
        not establish coverage across data gaps or prove temporal resolving power.
        """
        if isinstance(cadence_seconds, (bool, np.bool_)):
            raise ValueError('cadence_seconds must be finite and positive.')
        cadence_seconds = float(cadence_seconds)
        if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
            raise ValueError('cadence_seconds must be finite and positive.')
        instrument_id, _ = self._instrument(instrument_id)
        bounds = []
        for row in self.observations:
            if row.get('role') != 'train' or row['instrument_id'] != instrument_id:
                continue
            epoch = Time(row['time']['model_time'], scale='utc')
            offsets = np.asarray(row.get('time_offsets_seconds', [0]), dtype=float)
            if not offsets.size or not np.isfinite(offsets).all():
                raise ValueError('Training observation contains invalid raster time offsets.')
            bounds.extend([epoch+offsets.min()*u.s, epoch+offsets.max()*u.s])
        if not bounds:
            raise ValueError('No saved training times are available for this instrument.')
        start, stop = min(bounds), max(bounds)
        duration = (stop-start).to_value(u.s)
        if duration <= 1e-6:
            raise ValueError('Custom cadence requires a snapshot trained on a time series, not one instantaneous acquisition.')
        count = int(np.floor(duration/cadence_seconds+1e-9))+1
        if count < 2:
            raise ValueError('cadence_seconds exceeds the training time span.')
        return start+np.arange(count)*cadence_seconds*u.s

    def sample(self, coordinates, *, observation=None, time=None, **kwargs):
        """Evaluate physical SkyCoord positions through the same map/product engine.

        ``observation`` selects the observer and instrument calibration. A result
        without image WCS supports NPZ exports; use evaluate() for image maps.
        """
        return self.evaluate(observation, coordinates=coordinates, time=time, **kwargs)

    def evaluate(self, observation=None, *, parameters=None, instrument_id=None, time=None, batch_size=8192,
                 mu_limit=0.0, progress=False, shape=None, bounds=None, coordinates=None):
        """Evaluate named physical products on a saved or supplied observing grid.

        Evaluation batches geometry and synthesis. Output maps retain invalid pixels as NaN.
        ``time`` changes the model epoch while retaining the selected observing geometry.
        ``shape`` and ``bounds`` use grid(); ``coordinates`` accepts physical SkyCoord samples.
        """
        from astropy.coordinates import SkyCoord

        from .exports import EvaluationResult, expand_parameters, parameter_units
        from .grids import GridGeometry, support_mask
        from .progress import progress as track_progress
        from .train.spherical_synthesis import scale_spherical_forward_parameters, transform_spherical_parameters
        if isinstance(batch_size, (bool, np.bool_)) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if isinstance(mu_limit, (bool, np.bool_)) or not isinstance(mu_limit, Real) or not np.isfinite(mu_limit) or not 0 <= mu_limit < 1:
            raise ValueError("mu_limit must be finite and in [0,1).")
        selected = expand_parameters(parameters)
        if coordinates is not None and (shape is not None or bounds is not None):
            raise ValueError('Coordinate samples cannot also specify image shape or bounds.')
        row = self.grid(observation, shape=shape, bounds=bounds, instrument_id=instrument_id)
        instrument_id = row['instrument_id']
        context = GridGeometry(row)
        header, map_, velocity = context.header, context.map, context.velocity
        time_info = row.get('time', {})
        if not isinstance(time_info, dict):
            time_info = {'model_time': time_info}
        model_time = time if time is not None else time_info.get('model_time') or map_.date
        normalized_time = self._normalize_time(model_time)
        point_coordinates = None
        if coordinates is not None:
            if not isinstance(coordinates, SkyCoord):
                raise ValueError('coordinates must be physical astropy.coordinates.SkyCoord positions.')
            shape = coordinates.shape
            point_coordinates = coordinates.reshape(-1)
        else:
            shape = tuple(row['shape'])
        n = int(np.prod(shape))
        arrays, mask = {}, np.zeros(n, dtype=bool)
        sampled_coords = np.full((n, 4), np.nan) if point_coordinates is not None else None
        native_valid = support_mask(row).reshape(-1) if point_coordinates is None else None
        synthesis_needed = any(name in {"I", "Q", "U", "V", "stokes", "c_b0", "c_b1", "c_vdop", "limb_filling_factor"}
                               or name.startswith(("limb_", "artifact_", "forward_")) for name in selected)
        grid = row.get("wavelength_grid_m")
        if grid is None:
            grid = self.wavelength_config[instrument_id]["wavelength_grid"].to_value(u.m)
        response_file = row.get("spectral_response_file")
        if response_file is None:
            acquisition = row.get("acquisition_key")
            response_file = self.spectral_response_by_acquisition.get(instrument_id, {}).get(acquisition)
        batches = range(0, n, batch_size)
        for start in track_progress(batches, desc='Evaluating snapshot') if progress else batches:
            stop = min(start + batch_size, n)
            indices = np.arange(start, stop)
            if point_coordinates is None:
                g = context.pixels(indices, Rs_per_ds=self.Rs_per_ds, normalized_time=normalized_time,
                                   seconds_per_dt=self.seconds_per_dt, scan_times=time is None)
            else:
                g = context.points(point_coordinates[start:stop], Rs_per_ds=self.Rs_per_ds,
                                   normalized_time=normalized_time)
                sampled_coords[start:stop] = g['coords']
            valid = np.isfinite(g["coords"]).all(-1) & (g["mu"].reshape(-1) > mu_limit)
            if native_valid is not None:
                valid &= native_valid[start:stop]
            mask[start:stop] = valid
            if not valid.any():
                continue
            def tensor(key, geometry=g, valid_mask=valid):
                return torch.as_tensor(geometry[key][valid_mask], dtype=torch.float32, device=self.device)
            coords = tensor("coords")
            vector_potential = bool(getattr(self.parameter_model, "vector_potential", False))
            with torch.set_grad_enabled(vector_potential):
                coords.requires_grad_(vector_potential)
                raw = self.parameter_model(coords, instrument_id=instrument_id)
                transformed = transform_spherical_parameters(raw, coords, tensor("cartesian_to_spherical_transform"), tensor("rtp_to_img_transform"), self.meters_per_ds, self.seconds_per_dt)
                forward = scale_spherical_forward_parameters(raw, transformed, tensor("v_obs_los"), self.gauss_per_dB, self.meters_per_ds, self.seconds_per_dt)
                b = transformed["b_rtp"] * self.gauss_per_dB
                v = transformed["v_rtp"] * self.meters_per_ds / self.seconds_per_dt
                b_img = transformed["b_img"] * self.gauss_per_dB
                products = {**forward,
                            **{key: b[..., i] for i, key in enumerate(("Br", "Btheta", "Bphi"))},
                            **{key: v[..., i] for i, key in enumerate(("vr", "vtheta", "vphi"))},
                            **{key: b_img[..., i] for i, key in enumerate(("B_image_x", "B_image_y", "B_los"))},
                            "B": forward["b_field"], "B_transverse": torch.linalg.vector_norm(b_img[..., :2], dim=-1),
                            "inclination": forward["inc"] * 180 / np.pi,
                            "azi": transformed["azi_physical"],
                            "azimuth": torch.remainder(transformed["azi_physical"] * 180 / np.pi, 180),
                            "v_los": -transformed["v_img"][..., 2] * self.meters_per_ds / self.seconds_per_dt,
                            "mu": tensor("mu")}
            if 'disambiguation_angle' in raw:
                products['disambiguation_angle'] = raw['disambiguation_angle']
            products['Blos'] = products['B_los']
            products['Vdop'] = products['v_los']
            products = {key: value.detach().cpu().numpy() for key, value in products.items()}
            if synthesis_needed:
                detector = g.get('detector_coords')
                if detector is not None:
                    detector = detector[valid]
                synthesized = self.synthesize_observation(
                    g["coords"][valid], g["cartesian_to_spherical_transform"][valid], g["rtp_to_img_transform"][valid],
                    g["v_obs_los"][valid], instrument_id=instrument_id, wavelength_grid=grid,
                    spectral_response_file=response_file, detector_coords=detector, pix=g["pix"][valid],
                    batch_size=batch_size, progress=progress)
                # Intrinsic parameters keep one meaning regardless of whether spectra
                # were also requested. Instrument-corrected parameters are explicit.
                for name, value in synthesized.items():
                    if name in forward:
                        products['forward_' + name] = value
                    elif name not in products:
                        products[name] = value
                if 'artifact_correction_params' in synthesized:
                    correction = synthesized['artifact_correction_params']
                    for i, component in enumerate('QUV'):
                        products['artifact_alpha_' + component] = correction[..., i]
            for name in selected:
                if name not in products:
                    raise ValueError(f"Product {name!r} is not available for this instrument/correction configuration.")
                values = np.asarray(products[name])
                if values.ndim == 2 and values.shape[-1] == 1:
                    values = values[:, 0]
                if name not in arrays:
                    arrays[name] = np.full((n, *values.shape[1:]), np.nan, dtype=np.float32)
                arrays[name][start + np.flatnonzero(valid)] = values
        if not arrays:
            raise ValueError("Selected grid contains no visible finite solar surface pixels.")
        arrays = {name: values.reshape((*shape, *values.shape[1:])) for name, values in arrays.items()}
        metadata = {"snapshot": self.path.name, "snapshot_version": FORMAT_VERSION, "spinnme_version": __version__,
                    "observation_id": row["id"], "instrument_id": instrument_id, "model_time": Time(model_time).utc.isot,
                    "geometry_time": map_.date.utc.isot, "wavelength_grid_m": np.asarray(grid).tolist(),
                    "observer_velocity": np.asarray(velocity, dtype=float).tolist(),
                    "observer_velocity_correction_applied": bool(np.any(velocity)),
                    "wavelength_center_m": self.wavelength_config[instrument_id]["wavelength_center"].to_value(u.m),
                    "conventions": self.metadata["conventions"], "config_hash": self.metadata["config_hash"],
                    "line_id": row.get('line_id'), "original_times": time_info.get('original', {}),
                    "raster_time_offsets_applied": point_coordinates is None and time is None and row.get('time_offsets_seconds') is not None}
        if metadata['raster_time_offsets_applied']:
            metadata['time_offsets_seconds'] = row['time_offsets_seconds']
            metadata['time_offset_axis'] = row['time_offset_axis']
        if point_coordinates is not None:
            metadata['coordinates'] = {
                'frame': 'Carrington Cartesian normalized xyzt',
                'values': np.where(np.isfinite(sampled_coords), sampled_coords, None).tolist(),
                'meters_per_ds': self.meters_per_ds, 'seconds_per_dt': self.seconds_per_dt,
                'ref_time_utc': Time(self.ref_time).utc.isot,
                'shape': list(shape),
            }
        return EvaluationResult(arrays, {name: parameter_units(name) for name in arrays},
                                header if point_coordinates is None else None, mask.reshape(shape), metadata)

    def close(self):
        self._asset_directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
