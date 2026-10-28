"""Shared spherical observation synthesis, extracted without planar loaders."""
import numpy as np
import torch
from astropy import units as u
from tqdm import tqdm

from spinnme.train.spherical_synthesis import (
    correct_spherical_limb_effects,
    field_free_forward_parameters,
    mix_magnetic_filling_factor,
    scale_spherical_forward_parameters,
    transform_spherical_parameters,
)
from spinnme.train.stokes_loss import StokesLossModule


class SphericalSynthesis:
    def _instrument(self, instrument_id=None):
        instrument_id = self.instrument_id if instrument_id is None else instrument_id
        if self.forward_models is None:
            raise RuntimeError('This inversion does not contain saved spectral forward models.')
        if instrument_id not in self.forward_models:
            raise KeyError(
                f'Unknown instrument {instrument_id!r}; available instruments: {list(self.forward_models.keys())}'
            )
        return instrument_id, self.forward_models[instrument_id]

    @staticmethod
    def _flat_tensor(value, trailing_shape, name):
        value = torch.as_tensor(value, dtype=torch.float32)
        if tuple(value.shape[-len(trailing_shape):]) != tuple(trailing_shape):
            raise ValueError(f'{name} must end in shape {trailing_shape}, got {tuple(value.shape)}.')
        return value.reshape(-1, *trailing_shape)

    def synthesize_observation(self, coords, cartesian_to_spherical_transform,
                               rtp_to_img_transform, v_obs_los, mu=None,
                               wavelength_grid=None, instrument_id=None,
                               spectral_offsets=None, spectral_weights=None,
                               continuum_weights=None, detector_coords=None,
                               spectral_response_file=None, pix=None,
                               batch_size=2 ** 13, progress=True,
                               denormalize_stokes=False, image_p_angle=None):
        """Evaluate a spherical inversion and synthesize observer-frame Stokes profiles.

        Geometry and velocities follow the identical transformation used during
        training. HMI response maps are sampled per detector pixel and per batch,
        so a full-disk evaluation does not materialize the dense response cube.
        Wavelengths and response offsets are in meters; velocities are in m/s.
        mu is derived from rtp_to_img_transform. If supplied, it must agree with
        that projection; omit it to use the geometry directly.
        """
        instrument_id, forward_model = self._instrument(instrument_id)
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        if denormalize_stokes and self.continuum_normalization_mode == 'per_profile':
            raise ValueError(
                'Continuum-normalized profiles cannot be restored to absolute '
                'intensities using the instrument scalar alone. Evaluate with '
                'denormalize_stokes=False and multiply by the observed per-pixel continuum estimate '
                'in the desired intensity units.'
            )
        if denormalize_stokes and instrument_id not in self.stokes_normalization_by_instrument:
            raise KeyError(
                f'No saved Stokes normalization for {instrument_id!r}.'
            )
        coords = torch.as_tensor(coords, dtype=torch.float32)
        if coords.shape[-1] != 4:
            raise ValueError(f'coords must end in [x, y, z, t], got {tuple(coords.shape)}.')
        output_shape = tuple(coords.shape[:-1])
        n_points = int(np.prod(output_shape))
        if image_p_angle is not None:
            image_p_angle = torch.as_tensor(image_p_angle, dtype=torch.float32)
            if image_p_angle.ndim == 0:
                image_p_angle = image_p_angle.expand(n_points, 1)
            image_p_angle = self._flat_tensor(image_p_angle, (1,), 'image_p_angle')
            if image_p_angle.shape[0] != n_points:
                raise ValueError('image_p_angle must have one value per sample.')
        coords = coords.reshape(-1, 4)
        c2s = self._flat_tensor(cartesian_to_spherical_transform, (3, 3),
                                'cartesian_to_spherical_transform')
        rtp2img = self._flat_tensor(rtp_to_img_transform, (3, 3), 'rtp_to_img_transform')
        v_obs = self._flat_tensor(v_obs_los, (1,), 'v_obs_los')
        projected_mu = rtp2img[..., 2, 0:1]
        if mu is not None:
            supplied_mu = self._flat_tensor(mu, (1,), 'mu')
            if supplied_mu.shape != projected_mu.shape or not torch.allclose(
                    supplied_mu, projected_mu, atol=2e-6, rtol=2e-6, equal_nan=True):
                raise ValueError('mu must match the radial-to-LOS projection in rtp_to_img_transform.')
        mu = projected_mu
        for name, value in (('cartesian_to_spherical_transform', c2s),
                            ('rtp_to_img_transform', rtp2img), ('v_obs_los', v_obs), ('mu', mu)):
            if value.shape[0] != n_points:
                raise ValueError(f'{name} has {value.shape[0]} samples, expected {n_points}.')

        if wavelength_grid is None:
            wavelength_grid = self.wavelength_config[instrument_id]['wavelength_grid'].to_value(u.m)
        wavelength_grid = torch.as_tensor(wavelength_grid, dtype=torch.float32)
        if wavelength_grid.ndim == 1:
            wavelength_grid = wavelength_grid.unsqueeze(0).expand(n_points, -1)
        else:
            wavelength_grid = wavelength_grid.reshape(-1, wavelength_grid.shape[-1])
            if wavelength_grid.shape[0] == 1:
                wavelength_grid = wavelength_grid.expand(n_points, -1)
        if wavelength_grid.shape[0] != n_points:
            raise ValueError(f'wavelength_grid has {wavelength_grid.shape[0]} samples, expected {n_points}.')

        supplied_response = (spectral_offsets, spectral_weights, continuum_weights)
        if any(value is not None for value in supplied_response) and not all(
                value is not None for value in supplied_response):
            raise ValueError('spectral_offsets, spectral_weights, and continuum_weights are required together.')
        if spectral_response_file is not None and any(value is not None for value in supplied_response):
            raise ValueError('Provide either a response file or sampled response arrays, not both.')
        if spectral_response_file is not None and detector_coords is None:
            raise ValueError('detector_coords are required when spectral_response_file is provided.')
        if detector_coords is not None:
            detector_coords = self._flat_tensor(detector_coords, (2,), 'detector_coords')
            if detector_coords.shape[0] != n_points:
                raise ValueError(f'detector_coords has {detector_coords.shape[0]} samples, expected {n_points}.')
        if all(value is not None for value in supplied_response):
            spectral_offsets = torch.as_tensor(spectral_offsets, dtype=torch.float32).reshape(
                n_points, wavelength_grid.shape[-1], -1)
            spectral_weights = torch.as_tensor(spectral_weights, dtype=torch.float32).reshape_as(spectral_offsets)
            continuum_weights = torch.as_tensor(continuum_weights, dtype=torch.float32).reshape(
                n_points, wavelength_grid.shape[-1])
        is_hmi = forward_model.__class__.__name__ == 'HMIMEAtmosphere'
        if is_hmi and spectral_response_file is None \
                and spectral_offsets is None:
            raise ValueError('HMI synthesis requires a transmission response file or sampled response arrays.')

        if instrument_id in self.artifact_correction_models and pix is None:
            raise ValueError(f'Artifact correction for {instrument_id!r} requires normalized pix coordinates.')
        if pix is not None:
            pix = self._flat_tensor(pix, (2,), 'pix')
            if pix.shape[0] != n_points:
                raise ValueError(f'pix has {pix.shape[0]} samples, expected {n_points}.')

        finite = (torch.isfinite(coords).all(-1) & torch.isfinite(c2s).all((-2, -1)) &
                  torch.isfinite(rtp2img).all((-2, -1)) & torch.isfinite(v_obs).all(-1) &
                  torch.isfinite(mu).all(-1) & torch.isfinite(wavelength_grid).all(-1))
        if image_p_angle is not None:
            finite &= torch.isfinite(image_p_angle).all(-1)
        if detector_coords is not None:
            finite &= torch.isfinite(detector_coords).all(-1)
        if pix is not None:
            finite &= torch.isfinite(pix).all(-1)
        if spectral_offsets is not None:
            finite &= (torch.isfinite(spectral_offsets).all((-2, -1)) &
                       torch.isfinite(spectral_weights).all((-2, -1)) &
                       torch.isfinite(continuum_weights).all(-1))
        valid_indices = torch.nonzero(finite, as_tuple=False).flatten()
        collected = {}

        def append(name, value):
            collected.setdefault(name, []).append(value.detach().cpu())

        iterator = range(0, valid_indices.numel(), batch_size)
        iterator = tqdm(iterator, total=int(np.ceil(valid_indices.numel() / batch_size)),
                        disable=not progress)
        vector_potential = bool(getattr(getattr(self.parameter_model, 'module', self.parameter_model), 'vector_potential', False))
        for start in iterator:
            index = valid_indices[start:start + batch_size]
            batch_coords = coords[index].to(self.device)
            if vector_potential:
                batch_coords.requires_grad_(True)
            with torch.set_grad_enabled(vector_potential):
                raw = self.parameter_model(batch_coords, instrument_id=instrument_id)
                transformed = transform_spherical_parameters(
                    raw, batch_coords, c2s[index].to(self.device), rtp2img[index].to(self.device),
                    self.meters_per_ds, self.seconds_per_dt,
                )
                forward_params = scale_spherical_forward_parameters(
                    raw, transformed, v_obs[index].to(self.device), self.gauss_per_dB,
                    self.meters_per_ds, self.seconds_per_dt,
                )
                if image_p_angle is not None:
                    forward_params['image_p_angle'] = image_p_angle[index].to(self.device)
                if instrument_id in self.velocity_correction_models:
                    forward_params['vdop'] = forward_params['vdop'] + self.velocity_correction_models[
                        instrument_id](batch_coords[..., 3:4])
                limb_correction = None
                if instrument_id in self.limb_correction_models:
                    limb_correction = self.limb_correction_models[instrument_id](mu[index].to(self.device))
                    forward_params = correct_spherical_limb_effects(forward_params, limb_correction)

                response = {}
                if spectral_response_file is not None:
                    from spinnme.loader.spherical import sample_hmi_spectral_response
                    response = sample_hmi_spectral_response(
                        spectral_response_file, detector_coords[index]
                    )
                    response = {key: value.to(self.device) for key, value in response.items()}
                elif spectral_offsets is not None:
                    response = {
                        'spectral_offsets': spectral_offsets[index].to(self.device),
                        'spectral_weights': spectral_weights[index].to(self.device),
                        'continuum_weights': continuum_weights[index].to(self.device),
                    }
                elif not is_hmi and hasattr(self, 'module'):
                    response = self.module.default_spectral_response(
                        instrument_id, len(index), wavelength_grid.shape[-1], device=self.device)
                components = forward_model(
                    **forward_params, mu=mu[index].to(self.device),
                    wavelength_grid=wavelength_grid[index].to(self.device), **response,
                )
                stokes = torch.stack(components, dim=-2)
                if limb_correction is not None:
                    field_free_components = forward_model(
                        **field_free_forward_parameters(forward_params),
                        mu=mu[index].to(self.device),
                        wavelength_grid=wavelength_grid[index].to(self.device),
                        **response,
                    )
                    field_free_stokes = torch.stack(field_free_components, dim=-2)
                    stokes = mix_magnetic_filling_factor(
                        stokes,
                        field_free_stokes,
                        limb_correction['filling_factor'],
                    )
                artifact_correction = None
                if instrument_id in self.artifact_correction_models:
                    artifact_correction = self.artifact_correction_models[instrument_id](
                        pix[index].to(self.device), stokes
                    )
                    stokes = artifact_correction['stokes_corr']

            if self.continuum_normalization_mode == 'per_profile':
                stokes = StokesLossModule(**self.stokes_loss_config).normalize_profile(stokes)

            append('_index', index)
            for key, value in raw.items():
                if value is not None:
                    append(f'raw_{key}' if key in forward_params else key, value)
            for key, value in forward_params.items():
                append(key, value)
            for key in ('b_rtp', 'b_img'):
                append(key, transformed[key] * self.gauss_per_dB)
            for key in ('v_rtp', 'v_img'):
                append(key, transformed[key] * self.meters_per_ds / self.seconds_per_dt)
            append('stokes', stokes)
            for name, component in zip(('I', 'Q', 'U', 'V'), stokes.unbind(dim=-2)):
                append(name, component)
            if limb_correction is not None:
                for key, value in limb_correction.items():
                    output_key = 'limb_filling_factor' if key == 'filling_factor' else key
                    append(output_key, value)
            if artifact_correction is not None:
                append('artifact_correction_params', artifact_correction['correction_params'])
                if 'correction_gate' in artifact_correction:
                    append('artifact_gate', artifact_correction['correction_gate'])

        if not collected:
            raise ValueError('No finite coordinates were provided for synthesis.')
        indices = torch.cat(collected.pop('_index')).long()
        result = {}
        for key, chunks in collected.items():
            values = torch.cat(chunks)
            full = torch.full((n_points, *values.shape[1:]), torch.nan, dtype=values.dtype)
            full[indices] = values
            result[key] = full.numpy().reshape(*output_shape, *values.shape[1:])
        if denormalize_stokes:
            scale = self.stokes_normalization_by_instrument[instrument_id]
            for key in ('stokes', 'I', 'Q', 'U', 'V'):
                result[key] *= scale
        return result
