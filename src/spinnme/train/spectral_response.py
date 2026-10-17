"""Explicit inline passbands for non-HMI instruments."""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from astropy import units as u
from torch import nn


class InstrumentSpectralResponse(nn.Module):
    """A global or per-filter quadrature, portable through state_dict.

    Offsets are relative to each observation's tuning wavelengths. Tabulated
    weights already include quadrature spacing; all weights must be nonnegative.
    The response plus optional unpolarized continuum contribution normalizes to
    one for every filter. No passband width is inferred from an instrument name.
    """

    def __init__(self, config=None):
        super().__init__()
        config = copy.deepcopy(config or {'type': 'monochromatic'})
        if not isinstance(config, dict):
            raise ValueError('spectral_response must be a mapping.')
        self.kind = config.pop('type', 'monochromatic')
        if self.kind == 'monochromatic':
            if config:
                raise ValueError('Monochromatic response accepts no additional options.')
            self.register_buffer('offsets', torch.empty(0))
            self.register_buffer('weights', torch.empty(0))
            self.register_buffer('continuum', torch.empty(0))
            return
        if 'unit' not in config:
            raise ValueError('Finite spectral_response requires an explicit wavelength unit.')
        unit = u.Unit(config.pop('unit'))
        continuum = np.asarray(config.pop('continuum_weights', 0.), dtype=float)
        if self.kind == 'gaussian':
            fwhm = np.atleast_1d(np.asarray(config.pop('fwhm'), dtype=float)) * unit.to(u.m)
            samples = config.pop('samples', 9)
            if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
                raise ValueError('Gaussian response samples must be an integer of at least three.')
            if fwhm.ndim != 1 or not np.isfinite(fwhm).all() or np.any(fwhm <= 0):
                raise ValueError('Gaussian FWHM must be finite and positive (scalar or per-filter).')
            roots, quadrature = np.polynomial.hermite.hermgauss(samples)
            sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
            offsets = np.sqrt(2) * sigma[:, None] * roots
            weights = np.broadcast_to(quadrature / np.sqrt(np.pi), offsets.shape).copy()
        elif self.kind == 'airy':
            # One order of an ideal Fabry-Perot. The prefilter is assumed to
            # reject adjacent orders; it is not multiplied into calibrated L2
            # Stokes data a second time. No acquisition calibration is implied.
            fwhm = float(config.pop('fwhm')) * unit.to(u.m)
            fsr = float(config.pop('free_spectral_range')) * unit.to(u.m)
            samples = config.pop('samples', 161)
            if not np.isfinite([fwhm, fsr]).all() or not 0 < fwhm < fsr:
                raise ValueError('Airy response requires 0 < fwhm < free_spectral_range, both finite.')
            if isinstance(samples, bool) or not isinstance(samples, int) or samples < 3:
                raise ValueError('Airy response samples must be an integer of at least three.')
            # Gauss-Legendre integration in a sinh coordinate resolves the
            # narrow peak while retaining wings out to +/- FSR/2. Include the
            # Jacobian before normalizing the quadrature weights.
            roots, quadrature = np.polynomial.legendre.leggauss(samples)
            extent = np.arcsinh(fsr / fwhm)
            offsets = .5 * fwhm * np.sinh(extent * roots)
            jacobian = .5 * fwhm * extent * np.cosh(extent * roots)
            coefficient = 1 / np.sin(np.pi * fwhm / (2 * fsr)) ** 2
            transmission = 1 / (1 + coefficient * np.sin(np.pi * offsets / fsr) ** 2)
            weights = (quadrature * jacobian * transmission)[None]
            weights /= weights.sum(axis=-1, keepdims=True)
            offsets = offsets[None]
        elif self.kind == 'tabulated':
            offsets = np.asarray(config.pop('offsets'), dtype=float) * unit.to(u.m)
            weights = np.asarray(config.pop('weights'), dtype=float)
            if offsets.ndim == 1:
                offsets = offsets[None]
            if weights.ndim == 1:
                weights = weights[None]
            try:
                offsets, weights = np.broadcast_arrays(offsets, weights)
            except ValueError as error:
                raise ValueError('Tabulated response offsets and weights must broadcast to [filter,quadrature].') from error
            if offsets.ndim != 2 or not offsets.shape[-1]:
                raise ValueError('Tabulated response needs a nonempty global or per-filter profile.')
        else:
            raise ValueError('spectral_response.type must be monochromatic, gaussian, airy, or tabulated.')
        if config:
            raise ValueError(f'Unknown spectral response settings: {sorted(config)}.')
        if not np.isfinite(offsets).all() or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError('Response offsets must be finite; quadrature weights must be finite and nonnegative.')
        try:
            continuum = np.broadcast_to(continuum, (offsets.shape[0],)).copy()
        except ValueError as error:
            raise ValueError('continuum_weights must be scalar or match the number of filters.') from error
        if not np.isfinite(continuum).all() or np.any(continuum < 0):
            raise ValueError('continuum_weights must be finite and nonnegative.')
        normalization = weights.sum(-1) + continuum
        if np.any(normalization <= 0):
            raise ValueError('Every spectral filter must have positive total weight.')
        self.register_buffer('offsets', torch.tensor(offsets.copy(), dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(weights / normalization[:, None], dtype=torch.float32))
        self.register_buffer('continuum', torch.tensor(continuum / normalization, dtype=torch.float32))

    def forward(self, n_profiles, n_wavelengths, device=None):
        if self.kind == 'monochromatic':
            return {}
        if self.offsets.shape[0] not in (1, n_wavelengths):
            raise ValueError(f'Response has {self.offsets.shape[0]} filters but observation has {n_wavelengths} wavelengths.')
        def expanded(value, shape):
            return value.to(device=device or value.device).expand(shape)
        return {
            'spectral_offsets': expanded(self.offsets[None], (n_profiles, n_wavelengths, self.offsets.shape[-1])),
            'spectral_weights': expanded(self.weights[None], (n_profiles, n_wavelengths, self.weights.shape[-1])),
            'continuum_weights': expanded(self.continuum[None], (n_profiles, n_wavelengths)),
        }
