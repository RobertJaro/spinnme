"""Named scientific products and format writers shared by CLI and Python callers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

DEFAULT_PARAMETERS = ["Br", "Btheta", "Bphi"]
GROUPS = {"Brtp": DEFAULT_PARAMETERS, "brtp": DEFAULT_PARAMETERS,
          "Vrtp": ["vr", "vtheta", "vphi"], "vrtp": ["vr", "vtheta", "vphi"],
          "IQUV": ["I", "Q", "U", "V"],
          "Artifact": ["artifact_alpha_Q", "artifact_alpha_U", "artifact_alpha_V", "artifact_gate"]}
ALIASES = {"Bt": "Btheta", "Bp": "Bphi", "blos": "B_los", "b_los": "B_los", "vlos": "v_los",
           "Vlos": "v_los", "b_field": "B"}


def expand_parameters(parameters=None):
    if parameters is None:
        return DEFAULT_PARAMETERS.copy()
    if isinstance(parameters, str):
        parameters = [parameters]
    result = []
    for name in parameters:
        if name in {'LOS', 'ME', 'ME_forward'}:
            raise ValueError(f"{name!r} is too broad; select explicit parameter keys, e.g. "
                             "Brtp, Blos, Vdop, vmac, damping, b0, b1, kl or forward_vdop.")
        result.extend(GROUPS.get(name, [ALIASES.get(name, name)]))
    if not result:
        raise ValueError("Select at least one parameter.")
    return list(dict.fromkeys(result))


def parameter_units(name):
    name = name.removeprefix('forward_')
    if name.startswith("B") or name == "b_field":
        return "G"
    if name in {"vr", "vtheta", "vphi", "v_los", "Vdop", "vdop", "vmac", "c_vdop"}:
        return "m / s"
    if name in {"inclination", "azimuth"}:
        return "deg"
    if name in {"inc", "azi", "disambiguation_angle"}:
        return "rad"
    return "1"


def _safe(name):
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


@dataclass
class EvaluationResult:
    """Named arrays, units, support and provenance; image results also carry WCS."""
    arrays: dict
    units: dict
    header: fits.Header | None
    mask: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.header is not None and self.mask.ndim != 2:
            raise ValueError("Result mask must have spatial shape (y,x).")
        for name, array in self.arrays.items():
            if np.asarray(array).shape[:self.mask.ndim] != self.mask.shape:
                raise ValueError(f"Product {name} does not match the result's spatial grid.")

    def _metadata(self):
        return {**self.metadata, "units": self.units,
                "array_axes": "y,x[,stokes,wavelength]" if self.header is not None else "coordinates[,components]",
                "header": None if self.header is None else self.header.tostring(sep="\n", endcard=False, padding=False)}

    @property
    def maps(self):
        """Scalar products as SunPy Maps, with units, WCS and the validity mask."""
        from sunpy.map import Map
        self._require_image()
        result = {}
        for name, values in self.arrays.items():
            if np.asarray(values).ndim == 2:
                header = self.header.copy()
                header['BUNIT'] = self.units[name]
                header['CONTENT'] = name
                result[name] = Map(np.where(self.mask, values, np.nan), header, mask=~self.mask)
        return result

    def _require_image(self):
        if self.header is None:
            raise ValueError('Coordinate samples have no image WCS; use NPZ or evaluate on an image grid.')

    def write_npz(self, path, *, overwrite=False):
        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path / "parameters.npz"
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays, mask=self.mask,
                            metadata=np.array(json.dumps(self._metadata(), allow_nan=False)))
        return [path]

    def _fits_hdus(self, name, array):
        # Build from native spatial WCS while dropping unrelated input spectral axes.
        spatial = WCS(self.header).celestial.to_header(relax=True)
        for key in ("DATE-OBS", "DATE-AVG", "DATE-END", "TIMESYS", "DSUN_OBS", "RSUN_REF", "RSUN_OBS", "HGLN_OBS", "HGLT_OBS", "CRLN_OBS", "CRLT_OBS", "OBS_VR", "OBS_VW", "OBS_VN", "TELESCOP", "INSTRUME", "OBSRVTRY"):
            if key in self.header:
                spatial[key] = self.header[key]
        spatial["BUNIT"] = self.units.get(name, "1")
        spatial["CONTENT"] = name
        spatial["ORIGIN"] = "SPINNME"
        spatial["SPINNVER"] = self.metadata.get("spinnme_version", "unknown")
        spatial["HIERARCH SPINNME AMBIGUITY"] = "unresolved model azimuth branch"
        if "model_time" in self.metadata:
            spatial["HIERARCH SPINNME MODEL_TIME"] = self.metadata["model_time"]
        for key in ("Btheta", "Bphi", "B_los", "Blos", "v_los", "Vdop"):
            if name == key:
                spatial["HISTORY"] = self.metadata.get("conventions", {}).get({"Blos": "B_los", "Vdop": "v_los"}.get(key, key), "")
        data = np.asarray(array, dtype=np.float32)
        extra = []
        if data.ndim > 2:
            if name not in {'I', 'Q', 'U', 'V', 'stokes'}:
                raise ValueError(f"{name} is a parameter vector; select scalar components (e.g. Artifact) for FITS, or export NPZ.")
            # FITS axis 1/2 remain x/y. Canonical trailing axes precede y/x in NumPy FITS data.
            data = np.moveaxis(data, (0, 1), (-2, -1))
            offsets = np.asarray(self.metadata.get("wavelength_grid_m", []), dtype=np.float64)
            if not len(offsets) or array.shape[-1] != len(offsets):
                raise ValueError(f"Spectral product {name} requires matching wavelength metadata.")
            wavelengths = offsets + self.metadata["wavelength_center_m"]
            column = fits.Column(name="WAVELENGTH", format=f"{len(wavelengths)}D", dim=f"({len(wavelengths)})", unit="m", array=[wavelengths])
            extra.append(fits.BinTableHDU.from_columns([column], name="WCS-TAB"))
            spatial["CTYPE3"] = "WAVE-TAB"
            spatial["CUNIT3"] = "m"
            spatial["CRPIX3"] = 1.
            spatial["CRVAL3"] = 1.
            spatial["CDELT3"] = 1.
            spatial["PS3_0"] = "WCS-TAB"
            spatial["PS3_1"] = "WAVELENGTH"
            spatial["PV3_1"] = 1
            if array.ndim == 4:
                spatial["CTYPE4"] = "STOKES"
                spatial["CRPIX4"] = 1.
                spatial["CRVAL4"] = 1.
                spatial["CDELT4"] = 1.
        if self.metadata.get('raster_time_offsets_applied'):
            offsets = np.asarray(self.metadata['time_offsets_seconds'], dtype=np.float64)
            axis = self.metadata['time_offset_axis']
            if axis == 'x':
                offsets = offsets[None, :]
            elif axis == 'y':
                offsets = offsets[:, None]
            elif axis != 'yx':
                raise ValueError('Unknown raster time-offset axis.')
            offsets = np.broadcast_to(offsets, self.mask.shape)
            timing = fits.ImageHDU(offsets, name='TIME_OFFSETS')
            timing.header.update(WCS(self.header).celestial.to_header(relax=True))
            timing.header['BUNIT'] = 's'
            timing.header['DATEREF'] = self.metadata['model_time']
            timing.header['TIMESYS'] = 'UTC'
            extra.append(timing)
        provenance = json.dumps(self._metadata(), allow_nan=False)
        text = fits.Column(name="JSON", format=f"{max(1,len(provenance.encode('utf-8')))}A", array=[provenance])
        return fits.HDUList([fits.PrimaryHDU(data, spatial), fits.ImageHDU(self.mask.astype(np.uint8), name="VALID_MASK"),
                             *extra, fits.BinTableHDU.from_columns([text], name="PROVENANCE")])

    def write_fits(self, directory, *, overwrite=False):
        self._require_image()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        # Validate filenames and collisions before writing any file.
        names = [_safe(name) for name in self.arrays]
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("Parameter filenames collide after normalization.")
        for name in names:
            path = directory / f"{name}.fits"
            if path.exists() and not overwrite:
                raise FileExistsError(path)
        for name, array in self.arrays.items():
            path = directory / f"{_safe(name)}.fits"
            self._fits_hdus(name, array).writeto(path, overwrite=overwrite, checksum=True)
            paths.append(path)
        return paths

    def write_jpg(self, directory, *, overwrite=False, dpi=180, cmap=None):
        self._require_image()
        import matplotlib.pyplot as plt

        from ..plotting import magnetic_norm
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        names = [_safe(name).casefold() for name in self.arrays]
        if len(set(names)) != len(names):
            raise ValueError("Parameter filenames collide after normalization; use NPZ or separate output directories.")
        for name, value in self.arrays.items():
            path = directory / f"{_safe(name)}.jpg"
            if path.exists() and not overwrite:
                raise FileExistsError(path)
            array = np.asarray(value)
            if array.ndim != 2:
                raise ValueError(f"JPG needs a scalar map: {name} has shape {array.shape}; export spectral cubes as FITS/NPZ.")
            signed = name in {"Br", "Btheta", "Bphi", "B_los", "Blos", "vr", "vtheta", "vphi", "v_los", "Vdop", "B_image_x", "B_image_y"}
            valid = array[self.mask & np.isfinite(array)]
            if not len(valid):
                raise ValueError(f"No finite pixels in {name}")
            limits = np.nanpercentile(valid, [1, 99])
            if signed:
                maximum = max(abs(limits))
                limits = (-maximum, maximum)
            figure = plt.figure(figsize=(7, 6))
            axes = figure.add_subplot(projection=WCS(self.header).celestial)
            norm = magnetic_norm(name, valid)
            if norm is None:
                from matplotlib.colors import Normalize
                norm = Normalize(*limits)
            image = axes.imshow(np.where(self.mask, array, np.nan), origin="lower", cmap=cmap or ("RdBu_r" if signed else "viridis"), norm=norm)
            axes.set_title(f"{name} · {self.metadata.get('model_time', '')}")
            figure.colorbar(image, ax=axes, label=self.units.get(name, "1"))
            figure.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
        sidecar = directory / "metadata.json"
        sidecar.write_text(json.dumps(self._metadata(), indent=2))
        return [*paths, sidecar]

    def write(self, output, *, format="fits", overwrite=False, **kwargs):
        methods = {"fits": self.write_fits, "npz": self.write_npz, "jpg": self.write_jpg, "jpeg": self.write_jpg}
        if format not in methods:
            raise ValueError(f"Unknown export format: {format}")
        return methods[format](output, overwrite=overwrite, **kwargs)


def read_npz(path):
    """Read an exported NPZ without enabling Python object deserialization."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        arrays = {name: archive[name].copy() for name in archive.files if name not in {"mask", "metadata"}}
        mask = archive["mask"].copy()
    raw_header = metadata.pop('header')
    header = None if raw_header is None else fits.Header.fromstring(raw_header, sep='\n')
    return EvaluationResult(arrays, metadata.pop("units"), header, mask, metadata)
