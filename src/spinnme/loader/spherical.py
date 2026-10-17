import copy
import base64
import glob
import os
import re
from multiprocessing import get_context
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
try:
    import wandb
except ImportError:
    wandb = None
from astropy import units as u
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import all_coordinates_from_map, Map
from torch.utils.data import DataLoader

from spinnme.data.fits import (read_fits_observation, observation_header, observation_times,
                              parse_time, observer_velocity_from_header)
from spinnme.data.phi import read_phi_observation
from spinnme.data.hmi_transmission import (
    load_hmi_transmission_profile,
    read_hmi_fits_acquisition,
    resolve_hmi_transmission_profile,
)
from spinnme.data.hmi_time import load_hmi_date_obs_map
from spinnme.data.util import spherical_to_cartesian, cartesian_to_spherical_matrix, image_to_spherical_matrix, observer_ray_and_mu, image_p_angle
from spinnme.instrument import hmi_wavelength_config
from spinnme.progress import progress
from spinnme.loader.util import MultiprocessingWrapper
from spinnme.loader.polarization import polarization_selection
from spinnme.train.data_loader import TensorsDataset, CombinedDataset, TimeStratifiedBatchSampler




def _summarize_physics_coordinates(spherical_coords):
    """Return compact, seam-safe bounds for continuous collocation sampling."""
    flat_coords = np.asarray(spherical_coords).reshape(-1, 3)
    flat_coords = flat_coords[np.isfinite(flat_coords).all(axis=-1)]
    if flat_coords.size == 0:
        raise ValueError('Cannot build a physics domain from an empty spherical footprint.')

    latitude = np.pi / 2 - flat_coords[:, 1]
    longitude = flat_coords[:, 2]
    sin_sum = np.sin(longitude).sum(dtype=np.float64)
    cos_sum = np.cos(longitude).sum(dtype=np.float64)
    longitude_center = float(np.arctan2(sin_sum, cos_sum))
    longitude_offset = np.arctan2(
        np.sin(longitude - longitude_center),
        np.cos(longitude - longitude_center),
    )

    return {
        'radius_Rs': (float(flat_coords[:, 0].min()), float(flat_coords[:, 0].max())),
        'latitude_rad': (float(latitude.min()), float(latitude.max())),
        'longitude_center_rad': longitude_center,
        'longitude_offset_rad': (
            float(longitude_offset.min()), float(longitude_offset.max()),
        ),
        'longitude_vector_sum': (float(sin_sum), float(cos_sum)),
        'n_points': int(flat_coords.shape[0]),
    }


class SphericalDataModule(LightningDataModule):

    def __init__(self, train_configs, valid_config, work_directory,
                 seconds_per_dt=86400, Rs_per_ds=1, gauss_per_dB=1e3,
                 stokes_normalization=1.0, ref_time=None,
                 batch_size=65536, dataset_batch_size=4096,
                 num_workers=4, num_dataset_workers=8, balance_train_configs=False,
                 preparation_seed=None):
        super().__init__()
        self.spectral_response_files = {}
        self.spectral_response_by_acquisition = {}
        self.batch_size, self.dataset_batch_size = batch_size, dataset_batch_size
        self.balance_train_configs = balance_train_configs
        self.preparation_seed = preparation_seed
        self.num_workers = num_workers if num_workers is not None else 0
        self.num_dataset_workers = 8 if num_dataset_workers is None else num_dataset_workers
        if min(self.num_workers, self.num_dataset_workers) < 0:
            raise ValueError('Worker counts must be nonnegative.')
        self._validate_batch_sizes()
        configs = train_configs if isinstance(train_configs, list) else [train_configs]
        ref_time = self._default_reference_time(configs) if ref_time is None else parse_time(ref_time)
        self.ref_time, self.seconds_per_dt = ref_time, seconds_per_dt
        self.Rs_per_ds, self.gauss_per_dB = Rs_per_ds, gauss_per_dB
        self.train_config_indices, self.train_datasets = [], []
        common = dict(seconds_per_dt=seconds_per_dt, Rs_per_ds=Rs_per_ds,
                      ref_time=ref_time, work_directory=work_directory)
        for group, source_config in enumerate(configs):
            config = copy.deepcopy(source_config)
            ds_type = config.pop('type').lower()
            dataset_class = self._dataset_class(ds_type)
            files = self._load_IQUV_files(config.pop('data_path')) if ds_type in ('hmi', 'test') else self._load_all_files(config.pop('data_path'))
            files = self._select_files(files, config, ds_type)
            if not files.__len__():
                raise ValueError(f'No training files remain for dataset {group}.')
            normalization = config.pop('stokes_normalization', stokes_normalization)
            wrapper = MultiprocessingWrapper(dataset_class, **common,
                       stokes_normalization=normalization, batch_size=dataset_batch_size, **config)
            items = [dict(data=f, ds_id=f'train_{group:02d}_{i:03d}', store_plot_image=False)
                     for i, f in enumerate(files)]
            if preparation_seed is not None:
                for i,item in enumerate(items):
                    item['_preparation_seed'] = (preparation_seed + 100000*group+i) % 2**32
            if self.num_dataset_workers > 1:
                # W&B, CUDA and Zarr can already own background threads here.
                # Spawn fresh interpreters instead of inheriting those locks.
                workers = min(self.num_dataset_workers, len(items))
                with get_context('spawn').Pool(workers) as pool:
                    datasets = list(progress(pool.imap(wrapper.run, items), total=len(items),
                                             desc=f'Loading training data {group} ({ds_type})'))
            else:
                datasets = [wrapper.run(item) for item in
                            progress(items, desc=f'Loading training data {group} ({ds_type})')]
            offset = len(self.train_datasets)
            self.train_config_indices.append(list(range(offset, offset + len(datasets))))
            self.train_datasets.extend(datasets)
            for dataset in datasets:
                self._register_hmi_dataset_response(dataset)
        self.wavelength_config, self.stokes_normalization_by_instrument = {}, {}
        for dataset in self.train_datasets:
            key = dataset.instrument_id
            self.wavelength_config.setdefault(key, dataset.wavelength_config)
            previous = self.stokes_normalization_by_instrument.setdefault(key, dataset.stokes_normalization)
            if previous != dataset.stokes_normalization:
                raise ValueError(f'All datasets for {key!r} must share stokes_normalization.')
            center = self.wavelength_config[key]['wavelength_center'].to_value(u.m)
            if self.wavelength_config[key]['wavelength_grid'].size != dataset.wavelength_config['wavelength_grid'].size:
                raise ValueError(f'Acquisitions for {key!r} have different wavelength counts. '
                                 'Use distinct instrument IDs for distinct spectral setups; '
                                 'different counts across instrument IDs are supported.')
            if not np.isclose(center, dataset.wavelength_config['wavelength_center'].to_value(u.m), rtol=1e-12, atol=0):
                raise ValueError(f'Use distinct instrument IDs for different atomic lines: {key!r}.')
        config = copy.deepcopy(valid_config)
        ds_type = config.pop('type').lower()
        dataset_class = self._dataset_class(ds_type)
        files = self._load_IQUV_files(config.pop('data_path')) if ds_type in ('hmi', 'test') else self._load_all_files(config.pop('data_path'))
        sample_idx = config.pop('sample_idx', None)
        files = self._select_files(files, config, ds_type, select_sample=False)
        if not files.__len__():
            raise ValueError('No validation files matched.')
        sample_idx = len(files)//2 if sample_idx is None else sample_idx
        if not isinstance(sample_idx, int) or not 0 <= sample_idx < len(files):
            raise ValueError(f'Validation sample_idx {sample_idx} exceeds {len(files)} files.')
        normalization = config.pop('stokes_normalization', stokes_normalization)
        config.pop('oversample_factor', None)
        config.pop('polarization_filter', None)
        for selected in progress(files[sample_idx:sample_idx+1], desc='Loading validation data'):
            self.valid_dataset = dataset_class(selected, ds_id='valid', **common,
                         stokes_normalization=normalization, batch_size=batch_size,
                         filter_nans=False, shuffle=False, **config)
        self._register_hmi_dataset_response(self.valid_dataset)
        key = self.valid_dataset.instrument_id
        if self.stokes_normalization_by_instrument.get(key) != self.valid_dataset.stokes_normalization:
            raise ValueError('Validation instrument and normalization must match training.')
        trained = self.wavelength_config[key]
        validation = self.valid_dataset.wavelength_config
        if (trained['wavelength_grid'].size != validation['wavelength_grid'].size or
                not np.isclose(trained['wavelength_center'].to_value(u.m),
                               validation['wavelength_center'].to_value(u.m),rtol=1e-12,atol=0)):
            raise ValueError('Validation wavelength count and line center must match the same instrument ID. '
                             'Use distinct instrument IDs for distinct spectral setups.')
        self.times = [d.time for d in self.train_datasets]
        self.image_shape = self.valid_dataset.image_shape
        self.value_range = self.valid_dataset.value_range
        self.data_range = self.valid_dataset.data_range
        self.observations = self.get_observations()

    @staticmethod
    def _dataset_class(ds_type):
        classes = {'hmi': HMISphericalDataset, 'test': TestSphericalDataset,
                   'phi-hrt': PHIHRTSphericalDataset, 'phi_hrt': PHIHRTSphericalDataset,
                   'phi-fdt': PHIFDTSphericalDataset, 'phi_fdt': PHIFDTSphericalDataset,
                   'fits': FITSSphericalDataset, 'hinode': HinodeSphericalDataset}
        if ds_type not in classes:
            raise ValueError(f'Unknown spherical dataset type: {ds_type}')
        return classes[ds_type]

    def _default_reference_time(self, configs):
        """Choose the earliest selected model epoch, including raster timing.

        This inspects metadata before constructing caches. The source WCS and
        dates remain unchanged and the resolved UTC origin is saved in .pme.
        """
        from datetime import timedelta
        times = []
        for source in configs:
            config = copy.deepcopy(source)
            ds_type = config.pop('type').lower()
            paths = config.pop('data_path')
            files = self._load_IQUV_files(paths) if ds_type in ('hmi','test') else self._load_all_files(paths)
            files = self._select_files(files, config, ds_type)
            for group in files:
                path = np.asarray(group, dtype=object).reshape(-1)[0]
                model_time = self._observation_time_for_file_group(group, ds_type, config)
                with fits.open(path, memmap=False) as hdus:
                    if config.get('time_hdu') is not None:
                        offset = np.asarray(hdus[config['time_hdu']].data, dtype=float)
                        if not offset.size or not np.isfinite(offset).all():
                            raise ValueError('time_hdu must contain finite offsets.')
                        model_time += timedelta(seconds=float(offset.min()))
                    times.append(model_time)
        if not times:
            raise ValueError('No selected training observations define an automatic reference time.')
        return min(times)

    def _select_files(self, files, config, ds_type, select_sample=True):
        if 'time_range' in config:
            files = self._filter_files_by_time_range(files, config.pop('time_range'), ds_type, config)
        excluded_indices = set(config.pop('exclude_samples', []))
        excluded_times = {parse_time(v) for v in config.pop('exclude_times', [])}
        files = [f for i, f in enumerate(files) if i not in excluded_indices and
                 (not excluded_times or self._observation_time_for_file_group(f, ds_type, config) not in excluded_times)]
        if select_sample and 'sample_idx' in config:
            index = config.pop('sample_idx')
            if not isinstance(index, int) or not 0 <= index < len(files):
                raise ValueError(f'sample_idx {index!r} exceeds {len(files)} selected files.')
            files = files[index:index+1]
        if 'n_samples' in config:
            files = self._select_evenly_spaced_samples(files, config.pop('n_samples'))
        if 'first_n_samples' in config:
            count = config.pop('first_n_samples')
            if not isinstance(count, int) or count < 1:
                raise ValueError('first_n_samples must be a positive integer.')
            files = files[:count]
        return files

    def get_observations(self):
        """Return a JSON-safe acquisition registry for snapshots and exports."""
        rows = []
        for role, datasets in [('train', self.train_datasets), ('validation', [self.valid_dataset])]:
            for dataset in datasets:
                row = copy.deepcopy(dataset.observation)
                row['role'] = role
                rows.append(row)
        return rows

    def _validate_batch_sizes(self):
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError('batch_size must be a positive integer.')
        if not isinstance(self.dataset_batch_size, int) or self.dataset_batch_size <= 0:
            raise ValueError('dataset_batch_size must be a positive integer.')
        if self.batch_size < self.dataset_batch_size:
            raise ValueError('batch_size must be greater than or equal to dataset_batch_size.')
        if self.batch_size % self.dataset_batch_size:
            raise ValueError(
                f'batch_size ({self.batch_size}) must be divisible by dataset_batch_size '
                f'({self.dataset_batch_size}).'
            )

    @staticmethod
    def _select_evenly_spaced_samples(files, n_samples):
        if not isinstance(n_samples, int) or isinstance(n_samples, bool):
            raise TypeError('n_samples must be an integer.')
        if n_samples <= 0:
            raise ValueError('n_samples must be positive.')
        if n_samples > len(files):
            raise ValueError(
                f'n_samples ({n_samples}) exceeds the number of available acquisitions ({len(files)}).'
            )
        if n_samples == len(files):
            return files
        indices = np.rint(np.linspace(0, len(files) - 1, n_samples)).astype(int)
        if len(np.unique(indices)) != n_samples:
            raise RuntimeError(f'Unable to select {n_samples} unique samples from {len(files)} files.')
        return files[indices] if isinstance(files, np.ndarray) else [files[index] for index in indices]

    def _register_hmi_dataset_response(self, dataset):
        if not isinstance(dataset, HMISphericalDataset):
            return
        instrument_id = dataset.instrument_id
        response_file = dataset.spectral_response_file
        registered_files = self.spectral_response_files.setdefault(instrument_id, [])
        response_shapes = getattr(self, '_spectral_response_shapes', {})
        self._spectral_response_shapes = response_shapes
        if response_file not in registered_files:
            response_shape = load_hmi_transmission_profile(response_file)['offsets'].shape
            expected_shape = response_shapes.setdefault(instrument_id, response_shape)
            if response_shape != expected_shape:
                raise ValueError(
                    f'HMI response files for {instrument_id!r} use incompatible quadrature shapes: '
                    f'{expected_shape} and {response_shape}.'
                )
            registered_files.append(response_file)

        registered_acquisitions = self.spectral_response_by_acquisition.setdefault(instrument_id, {})
        existing = registered_acquisitions.get(dataset.acquisition_key)
        if existing is not None and existing != response_file:
            raise ValueError(
                f'Conflicting HMI responses for {instrument_id!r}, acquisition '
                f'{dataset.acquisition_key!r}: {existing!r} and {response_file!r}.'
            )
        registered_acquisitions[dataset.acquisition_key] = response_file

    def plot_dataset(self, ds):
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))

        im = axs[0].imshow(ds.integrated_V, cmap='gray', origin='lower', norm='log')
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Integrated V')

        im = axs[1].imshow(ds.latitude, origin='lower', cmap='seismic', vmin=0, vmax=np.pi)
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Latitude [rad]')

        im = axs[2].imshow(ds.longitude % (2 * np.pi), origin='lower', cmap='twilight', vmin=0, vmax=2 * np.pi)
        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Longitude [rad]')

        v_min_max = np.nanmax(np.abs(ds.v_obs_los))
        im = axs[3].imshow(ds.v_obs_los, origin='lower', cmap='coolwarm_r', vmin=-v_min_max, vmax=v_min_max)
        divider = make_axes_locatable(axs[3])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical', label='Observer LOS Velocity [m/s]')

        axs[0].set_title(
            f'Time: {ds.time.isoformat(" ", timespec="hours")} - lat:{np.rad2deg(ds.obs_lat):.1f}°, lon:{np.rad2deg(ds.obs_lon):.1f}°')

        fig.tight_layout()
        if wandb is not None and wandb.run is not None:
            wandb.log({'Data Overview': wandb.Image(fig)})
        plt.close(fig)

    def _load_IQUV_files(self, data_path, num_wl=6):
        if isinstance(data_path, str):
            if os.path.isdir(data_path):
                candidate_files = sorted(glob.glob(os.path.join(data_path, '*.fits')))
            else:
                candidate_files = sorted(glob.glob(data_path))
        elif isinstance(data_path, Iterable):
            candidate_files = []
            for path in data_path:
                if os.path.isdir(path):
                    candidate_files.extend(sorted(glob.glob(os.path.join(path, '*.fits'))))
                else:
                    candidate_files.extend(sorted(glob.glob(path)))
        else:
            raise ValueError(f'Unknown data path type: {type(data_path)}. Expected str or Iterable[str].')

        if len(candidate_files) == 0:
            raise ValueError(f'No FITS files matched data_path={data_path!r}.')
        candidate_files = sorted(set(candidate_files))

        # Group by the complete acquisition prefix, rather than sorting each
        # Stokes/wavelength stream independently. This makes a missing segment
        # an explicit error instead of silently pairing different timestamps.
        segment_pattern = re.compile(r'^(?P<prefix>.+)\.(?P<stokes>[IQUV])(?P<wl>\d+)\.fits$')
        acquisitions = {}
        for file in candidate_files:
            match = segment_pattern.match(file)
            if match is None:
                continue
            wavelength_index = int(match.group('wl'))
            if wavelength_index >= num_wl:
                continue
            key = match.group('prefix')
            segment = (match.group('stokes'), wavelength_index)
            if segment in acquisitions.setdefault(key, {}):
                raise ValueError(f'Duplicate segment {segment} for acquisition {key!r}.')
            acquisitions[key][segment] = file

        if not acquisitions:
            raise ValueError(
                f'No I/Q/U/V FITS segments matched data_path={data_path!r}. '
                f'Expected filenames ending in .I0.fits ... .V{num_wl - 1}.fits.'
            )

        expected_segments = {(stokes_id, wavelength_index)
                             for stokes_id in 'IQUV' for wavelength_index in range(num_wl)}
        grouped_files = []
        for acquisition, segments in sorted(acquisitions.items()):
            missing = sorted(expected_segments - segments.keys())
            unexpected = sorted(segments.keys() - expected_segments)
            if missing or unexpected:
                raise ValueError(
                    f'Incomplete I/Q/U/V file set for acquisition {acquisition!r}: '
                    f'missing={missing}, unexpected={unexpected}.'
                )
            grouped_files.append([
                [segments[(stokes_id, wavelength_index)] for wavelength_index in range(num_wl)]
                for stokes_id in 'IQUV'
            ])

        return np.asarray(grouped_files)  # time, stokes, wavelength

    @staticmethod
    def _parse_observation_time(value):
        return parse_time(value)

    @classmethod
    def _observation_time_for_file_group(cls, files, ds_type, config=None):
        """Selected model epoch in UTC; selectors use the same clock as ingestion."""
        config = config or {}
        representative_file = np.asarray(files, dtype=object).reshape(-1)[0]
        with fits.open(representative_file, memmap=False) as hdus:
            data_hdu = config.get('data_hdu', 0)
            if ds_type in ('hmi', 'test'):
                data_hdu = next((i for i, h in enumerate(hdus) if 'DATE-OBS' in h.header), None)
                if data_hdu is None:
                    raise ValueError(f'Missing DATE-OBS in {representative_file!r}.')
            header = observation_header(hdus, data_hdu, config.get('spatial_header_hdu'))
        header.update(config.get('observer') or {})
        return observation_times(header, config.get('time_reference', 'observation'),
                                 config.get('time_scale'))[1]

    @classmethod
    def _hmi_date_obs(cls, path):
        return cls._observation_time_for_file_group(path, 'hmi')

    @classmethod
    def _filter_files_by_time_range(cls, files, time_range, ds_type, config=None):
        if not isinstance(time_range, (list, tuple)) or len(time_range) != 2:
            raise ValueError('time_range must contain exactly [start, end].')
        start_time, end_time = (cls._parse_observation_time(value) for value in time_range)
        if end_time < start_time:
            raise ValueError(f'time_range end {end_time} precedes start {start_time}.')

        selected = [
            file_group for file_group in files
            if start_time <= cls._observation_time_for_file_group(file_group, ds_type, config) <= end_time
        ]
        if not selected:
            raise ValueError(
                f'No {ds_type} observations fall within the inclusive time range '
                f'[{start_time}, {end_time}].'
            )
        if isinstance(files, np.ndarray):
            return np.asarray(selected, dtype=files.dtype)
        return selected

    def _load_all_files(self, data_path):
        paths = [data_path] if isinstance(data_path, (str, os.PathLike)) else data_path
        files = []
        for path in paths:
            path = os.fspath(path)
            if os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, '*.fits')))
                files.extend(glob.glob(os.path.join(path, '*.fits.gz')))
            else:
                files.extend(glob.glob(path))
        if not files:
            raise ValueError(f'No FITS files matched {data_path!r}.')
        return sorted(set(files))

    def train_dataloader(self):
        self._validate_batch_sizes()
        datasets = self.train_datasets
        balanced = self.balance_train_configs
        groups = self.train_config_indices if balanced else [list(range(len(datasets)))]
        chunks_per_batch = self.batch_size // self.dataset_batch_size
        if chunks_per_batch % len(groups):
            raise ValueError('batch_size must be divisible by dataset_batch_size * number of balanced train_configs.')
        for ds in datasets:
            ds.batch_size = self.dataset_batch_size
            ds.pad_last_batch = balanced
        combined_dataset = CombinedDataset(datasets)
        pools = [[combined_dataset.sample_indices_by_dataset[i] for i in group] for group in groups]
        sampler = TimeStratifiedBatchSampler(
            pools[0], chunks_per_batch // len(groups),
            additional_dataset_groups=pools[1:] if balanced else None,
        )
        loader_kwargs = {
            'dataset': combined_dataset,
            'batch_size': None,
            'sampler': sampler,
            'num_workers': self.num_workers,
            'pin_memory': False,
        }
        if self.num_workers > 0:
            loader_kwargs.update(prefetch_factor=2, persistent_workers=True, multiprocessing_context='spawn')
        loader = DataLoader(**loader_kwargs)
        return loader

    def val_dataloader(self):
        self.valid_dataset.batch_size = self.batch_size
        options = {'multiprocessing_context': 'spawn'} if self.num_workers > 0 else {}
        data_loader = DataLoader(self.valid_dataset, batch_size=None, num_workers=self.num_workers,
                                 pin_memory=False, shuffle=False, **options)
        return data_loader


class SphericalDataset(TensorsDataset):

    def __init__(self, stokes, map_data, wavelength_config, ds_id, instrument_id, seconds_per_dt, Rs_per_ds, ref_time,
                 stokes_normalization, work_directory, oversample_factor=1,
                 store_plot_image=False, mu_limit=1e-3, pixel_mask=None, line_id="fe-i-6173",
                 observation_metadata=None, polarization_filter=None, loss_weight=1.0, **kwargs):
        self.ds_id = ds_id
        self.line_id = line_id
        self.instrument_id = instrument_id
        if isinstance(oversample_factor, bool) or not isinstance(oversample_factor, (int, np.integer)) or oversample_factor < 1:
            raise ValueError('oversample_factor must be a positive integer.')
        self.oversample_factor = oversample_factor
        self.loss_weight = float(loss_weight)
        if not np.isfinite(self.loss_weight) or self.loss_weight <= 0:
            raise ValueError("loss_weight must be finite and positive.")

        self.wavelength_config = wavelength_config  # lambda grid and reference wavelength
        self.stokes_normalization = float(stokes_normalization)
        if not np.isfinite(self.stokes_normalization) or self.stokes_normalization <= 0:
            raise ValueError("stokes_normalization must be finite and positive.")
        if not np.isfinite(seconds_per_dt) or seconds_per_dt <= 0 or not np.isfinite(Rs_per_ds) or Rs_per_ds <= 0:
            raise ValueError('seconds_per_dt and Rs_per_ds must be finite and positive.')
        if not np.isfinite(mu_limit) or not 0 <= mu_limit < 1:
            raise ValueError('mu_limit must be finite and in [0,1).')

        # load coordinates
        self.num_wl = stokes.shape[-1]
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.ref_time = ref_time

        # normalize time
        time = map_data['time']  # datetime
        self.normalized_time = (time - ref_time).total_seconds() / seconds_per_dt

        # observer info
        self.obs_lat = map_data['obs_lat']
        self.obs_lon = map_data['obs_lon']
        self.pAng = map_data['pAng']
        self.observer_distance = map_data['observer_distance']
        self.observer_velocity = np.asarray(map_data['observer_velocity'], dtype=np.float64)
        self.detector_mapping = map_data.get('detector_mapping')
        self.mu_limit = mu_limit
        self.time = map_data['time']
        self.wcs = map_data['wcs']


        # Full-frame geometry is temporary: keep one mask for filtering and bounds.
        mu = map_data['mu']
        spherical_coords = map_data['spherical_coords']
        v_obs_los = map_data['v_obs_los']
        stokes = np.asarray(stokes, dtype=np.float32) / self.stokes_normalization
        valid = np.isfinite(stokes).all(axis=(-2, -1))
        valid &= np.isfinite(spherical_coords).all(axis=-1)
        valid &= np.isfinite(mu) & (mu >= mu_limit) & np.isfinite(v_obs_los)
        native_valid = valid.copy()
        if pixel_mask is not None and not hasattr(self, 'polarization_stats'):
            native_valid &= pixel_mask
        if polarization_filter is not None:
            radius = np.sqrt(np.maximum(0, 1 - np.asarray(mu) ** 2))
            selected, _, self.polarization_stats = polarization_selection(stokes, radius, valid, **polarization_filter)
            valid &= selected
        if pixel_mask is not None:
            valid &= pixel_mask
        stokes[~valid] = np.nan
        spherical_coords = np.where(valid[..., None], spherical_coords, np.nan)
        self.physics_bounds = _summarize_physics_coordinates(spherical_coords[valid])
        self.physics_bounds['normalized_time'] = float(self.normalized_time)
        if 'pixel_time_offsets' in map_data:
            offsets = np.asarray(map_data['pixel_time_offsets'])[native_valid] / seconds_per_dt
            self.physics_bounds['normalized_time_range'] = [float(self.normalized_time + offsets.min()),
                                                            float(self.normalized_time + offsets.max())]

        # Plot Data Overview
        self.store_plot_image = store_plot_image
        if store_plot_image:
            self.integrated_V = block_reduce(np.abs(stokes[:, :, -1]).sum(-1), (8, 8), np.mean)
            self.latitude = block_reduce(spherical_coords[..., 1], (8, 8), np.mean)
            self.longitude = block_reduce(spherical_coords[..., 2], (8, 8), np.mean)
            self.v_obs_los = block_reduce(v_obs_los, (8, 8), np.mean)

        # image info
        self.value_range = np.stack([np.nanmin(stokes, (0, 1, -1)), np.nanmax(stokes, (0, 1, -1))], -1)
        self.image_shape = stokes.shape[:2]  # y, x
        header = map_data.get('header', self.wcs.to_header())
        header_dict = {key: value if isinstance(value, (str, int, float, bool)) else str(value)
                       for key, value in header.items() if key not in ('', 'COMMENT', 'HISTORY')}
        metadata = copy.deepcopy(observation_metadata or {})
        self.observation = {
            **metadata, 'id': ds_id, 'instrument_id': instrument_id, 'line_id': line_id,
            'shape': list(self.image_shape), 'header': header_dict,
            'valid_mask_bits': base64.b64encode(np.packbits(native_valid).tobytes()).decode('ascii'),
            'time': metadata.get('time', {'model_time': self.time.isoformat(), 'time_reference': 'observation'}),
            'source_paths': metadata.get('source_paths', []),
            'obs_lon': float(self.obs_lon.to_value(u.rad)),
            'obs_lat': float(self.obs_lat.to_value(u.rad)), 'pAng': float(self.pAng.to_value(u.rad)),
            'observer_distance': float(self.observer_distance),
            'observer_velocity': self.observer_velocity.tolist(), 'normalized_time': float(self.normalized_time),
            'wavelength_center_m': float(wavelength_config['wavelength_center'].to_value(u.m)),
            'wavelength_grid_m': wavelength_config['wavelength_grid'].to_value(u.m).tolist(),
            'stokes_normalization': self.stokes_normalization,
            'detector_mapping': None if self.detector_mapping is None else [np.asarray(v).tolist() for v in self.detector_mapping],
            'pix_convention': 'image' if self.detector_mapping is None else 'detector',
            'pix_axes': 'yx', 'pix_range': [-1, 1],
        }
        self.observation['geometry'] = {key: self.observation[key] for key in ('obs_lon', 'obs_lat', 'pAng', 'observer_distance', 'observer_velocity', 'normalized_time')}
        if 'pixel_time_offsets' in map_data:
            offsets = np.asarray(map_data['pixel_time_offsets'])
            if np.all(offsets == offsets[0:1]):
                self.observation['time_offsets_seconds'] = offsets[0].tolist()
                self.observation['time_offset_axis'] = 'x'
            elif np.all(offsets == offsets[:, 0:1]):
                self.observation['time_offsets_seconds'] = offsets[:, 0].tolist()
                self.observation['time_offset_axis'] = 'y'
            else:
                self.observation['time_offsets_seconds'] = offsets.tolist()
                self.observation['time_offset_axis'] = 'yx'
        for name in ('spectral_response_file', 'acquisition_key'):
            if hasattr(self, name):
                self.observation[name] = str(getattr(self, name))
        self.data_range = np.array(
            [[np.nanmin(spherical_coords[..., i]), np.nanmax(spherical_coords[..., i])] for i in range(3)])

        # lambda grid
        self.wavelength_grid = torch.tensor(wavelength_config['wavelength_grid'].to_value(u.m),
                                            dtype=torch.float32).reshape(
            (1, -1))

        # Filter once. Integer pixel IDs preserve the original image location
        # after filtering/shuffling without storing any derived geometry.
        filter_nans = kwargs.pop('filter_nans', True)
        shuffle = kwargs.pop('shuffle', True)
        flat_stokes = stokes.reshape((-1, *stokes.shape[2:]))
        tensors = {'stokes': flat_stokes[valid.ravel()] if filter_nans else flat_stokes}
        if 'pixel_time_offsets' in map_data:
            offsets = np.asarray(map_data['pixel_time_offsets'], dtype=np.float32).reshape(-1, 1) / seconds_per_dt
            tensors['time_offset'] = offsets[valid.ravel()] if filter_nans else offsets
        if filter_nans or shuffle:
            count = int(np.prod(self.image_shape))
            index_dtype = np.int32 if count <= np.iinfo(np.int32).max else np.int64
            indices = np.arange(count, dtype=index_dtype)[:, None]
            tensors['pixel_index'] = indices[valid.ravel()] if filter_nans else indices
        super().__init__(tensors=tensors, work_directory=work_directory,
                         filter_nans=False, shuffle=shuffle, **kwargs)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out['instrument_id'] = self.instrument_id
        out['sample_weight'] = torch.full((len(out['stokes']), 1), self.loss_weight, dtype=torch.float32)
        pixel_index = out.pop('pixel_index', None)
        if pixel_index is None:
            pixel_index = np.arange(idx * self.batch_size, idx * self.batch_size + len(out['stokes']))
        else:
            pixel_index = pixel_index.numpy().reshape(-1)
        geometry = spherical_geometry_from_pixels(
            pixel_index, self.image_shape, self.wcs,
            self.obs_lon.to_value(u.rad), self.obs_lat.to_value(u.rad),
            self.pAng.to_value(u.rad), self.observer_distance, self.observer_velocity,
            self.Rs_per_ds, self.normalized_time,
        )
        time_offset = out.pop('time_offset', None)
        if time_offset is not None:
            geometry['coords'][..., -1] += time_offset.numpy().reshape(-1)
        valid = np.isfinite(out['stokes'].numpy()).all(axis=(-2, -1))
        valid &= np.isfinite(geometry['mu'][..., 0]) & (geometry['mu'][..., 0] >= self.mu_limit)
        for key, value in geometry.items():
            value[~valid] = np.nan
            out[key] = torch.from_numpy(value.astype(np.float32))
        out['wavelength_grid'] = self.wavelength_grid.expand(len(pixel_index), -1)
        if self.detector_mapping is not None:
            row, column = np.divmod(pixel_index, self.image_shape[1])
            origin, step = self.detector_mapping
            detector = np.stack([column, row], axis=-1) * step + origin
            detector = np.clip(detector, 0, 1).astype(np.float32)
            detector[~valid] = np.nan
            out['detector_coords'] = torch.from_numpy(detector)
            out['pix'] = torch.from_numpy((detector[..., ::-1].copy() * 2 - 1).astype(np.float32))
        return out


def spherical_geometry_from_pixels(pixel_index, image_shape, wcs, obs_lon, obs_lat,
                                   p_angle, observer_distance, observer_velocity,
                                   Rs_per_ds, normalized_time):
    """Reconstruct a batch's surface geometry from pixel IDs and frame metadata.

    WCS supplies helioprojective angles. Analytic ray/sphere intersection avoids
    constructing SunPy coordinate objects on every dataset read.
    """
    row, column = np.divmod(pixel_index, image_shape[1])
    if tuple(wcs.wcs.ctype) != ('HPLN-TAN', 'HPLT-TAN'):
        # Any valid solar WCS can use the SunPy coordinate path. TAN frames
        # retain the analytic ray/sphere fast path used by full-disk HMI.
        world = wcs.pixel_to_world(column, row)
        carrington = world.transform_to(frames.HeliographicCarrington)
        lon, lat = carrington.lon.to_value(u.rad), carrington.lat.to_value(u.rad)
        spherical = np.stack([np.ones_like(lon), np.pi/2-lat, lon], axis=-1)
        c2s = cartesian_to_spherical_matrix(spherical)
        rtp2img = np.swapaxes(image_to_spherical_matrix(lon, lat, obs_lon, obs_lat, p_angle,
                              observer_distance=observer_distance), -1, -2)
        ray, mu = observer_ray_and_mu(lon, lat, obs_lon, obs_lat, observer_distance)
        pixel_center = (np.asarray(image_shape)-1)/2
        pix = (np.stack([row,column],axis=-1)-pixel_center)/max(*pixel_center,1.)
        coords = np.concatenate([spherical_to_cartesian(spherical)/Rs_per_ds,
                                 np.full((*lon.shape,1),normalized_time)],axis=-1)
        return {'coords':coords, 'mu':mu[...,None],
                'cartesian_to_spherical_transform':c2s, 'rtp_to_img_transform':rtp2img,
                'v_obs_los':(ray @ observer_velocity)[...,None], 'pix':pix}
    tx, ty = wcs.pixel_to_world_values(column, row)
    tx = np.asarray(tx) * u.Unit(wcs.wcs.cunit[0]).to(u.rad)
    ty = np.asarray(ty) * u.Unit(wcs.wcs.cunit[1]).to(u.rad)
    dx, dy, dz = np.cos(ty) * np.sin(tx), np.sin(ty), np.cos(ty) * np.cos(tx)
    discriminant = 1 - observer_distance ** 2 * (dx ** 2 + dy ** 2)
    with np.errstate(invalid='ignore'):
        travel = observer_distance * dz - np.sqrt(discriminant)
    surface = np.stack([travel * dx, travel * dy, observer_distance - travel * dz], axis=-1)
    # Observer-centered west/north/radial axes expressed in Carrington Cartesian.
    sl, cl, sb, cb = np.sin(obs_lon), np.cos(obs_lon), np.sin(obs_lat), np.cos(obs_lat)
    observer_basis = np.array([[-sl, -sb * cl, cb * cl],
                               [cl, -sb * sl, cb * sl], [0., cb, sb]])
    xyz = surface @ observer_basis.T
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    lat = np.arctan2(xyz[..., 2], np.hypot(xyz[..., 0], xyz[..., 1]))
    spherical = np.stack([np.ones_like(lon), np.pi / 2 - lat, lon], axis=-1)
    c2s = cartesian_to_spherical_matrix(spherical)
    rtp2img = np.swapaxes(image_to_spherical_matrix(
        lon, lat, obs_lon, obs_lat, p_angle, observer_distance=observer_distance,
    ), -1, -2)
    ray = np.stack([-dx, -dy, dz], axis=-1)
    v_obs_los = ray @ observer_velocity
    pixel_center = (np.asarray(image_shape) - 1) / 2
    pix = (np.stack([row, column], axis=-1) - pixel_center) / max(*pixel_center, 1.)
    coords = np.concatenate([spherical_to_cartesian(spherical) / Rs_per_ds,
                             np.full((*lon.shape, 1), normalized_time)], axis=-1)
    return {'coords': coords, 'mu': rtp2img[..., 2, 0:1].copy(),
            'cartesian_to_spherical_transform': c2s, 'rtp_to_img_transform': rtp2img,
            'v_obs_los': v_obs_los[..., None], 'pix': pix}


def load_v_observer_LOS(s_map, ray=None):
    if ray is None:
        carrington = all_coordinates_from_map(s_map).transform_to(frames.HeliographicCarrington)
        ray, _ = observer_ray_and_mu(
            carrington.lon.to_value(u.rad), carrington.lat.to_value(u.rad),
            s_map.carrington_longitude.to_value(u.rad), s_map.carrington_latitude.to_value(u.rad),
            (s_map.dsun / s_map.rsun_meters).to_value(u.dimensionless_unscaled),
        )
    # Preserve the FITS OBS_VW/OBS_VN/OBS_VR sign convention: a receding
    # observer contributes positive Doppler velocity at disk center.
    return ray @ observer_velocity_from_header(s_map.meta)




_HMI_RESPONSE_CACHE = {}


def sample_hmi_spectral_response(profile_file, detector_coords):
    """Interpolate one CPU-side HMI response map for the current dataset batch."""
    profile_file = os.path.abspath(os.fspath(profile_file))
    stat = os.stat(profile_file)
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _HMI_RESPONSE_CACHE.get(profile_file)
    if cached is None or cached[0] != identity:
        arrays = load_hmi_transmission_profile(profile_file)
        response = {
            'spectral_offsets': torch.from_numpy(arrays['offsets'] * 1e-10),
            'spectral_weights': torch.from_numpy(arrays['weights']),
            'continuum_weights': torch.from_numpy(arrays['continuum_weights']),
        }
        _HMI_RESPONSE_CACHE[profile_file] = identity, response
    else:
        response = cached[1]

    flat_coords = detector_coords.reshape(-1, 2).to(dtype=torch.float32, device='cpu')
    weights = response['spectral_weights']
    continuum_weights = response['continuum_weights']
    height, width, n_filters, n_samples = weights.shape
    grid = flat_coords.reshape(1, -1, 1, 2) * 2 - 1
    profile_grid = weights.permute(2, 3, 0, 1).reshape(1, n_filters * n_samples, height, width)
    sampled_weights = F.grid_sample(
        profile_grid, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).reshape(n_filters, n_samples, -1).movedim(-1, 0)
    continuum_grid = continuum_weights.permute(2, 0, 1).unsqueeze(0)
    sampled_continuum = F.grid_sample(
        continuum_grid, grid, mode='bilinear', padding_mode='border', align_corners=True
    ).reshape(n_filters, -1).movedim(-1, 0)
    sampled_offsets = response['spectral_offsets'].unsqueeze(0).expand(flat_coords.shape[0], -1, -1)

    spatial_shape = detector_coords.shape[:-1]
    return {
        'spectral_offsets': sampled_offsets.reshape(*spatial_shape, n_filters, n_samples),
        'spectral_weights': sampled_weights.reshape(*spatial_shape, n_filters, n_samples),
        'continuum_weights': sampled_continuum.reshape(*spatial_shape, n_filters),
    }


class HMISphericalDataset(SphericalDataset):

    def __init__(self, data, transmission_profile_directory, resolution=None,
                 require_quality_zero=True, polarization_filter=None, *args, **kwargs):
        if polarization_filter is not None and resolution is not None:
            raise ValueError('Polarization sampling requires native-resolution training images.')
        wavelength_config = hmi_wavelength_config()
        stokes_i_files, _, _, _ = data
        ref_file = stokes_i_files[0]
        acquisition = read_hmi_fits_acquisition(ref_file)
        self.acquisition_key = acquisition['acquisition_key']
        if require_quality_zero:
            self._validate_quality(data)
        self._validate_segment_alignment(data)
        self.spectral_response_file = os.fspath(
            resolve_hmi_transmission_profile(transmission_profile_directory, ref_file)
        )
        source_map = load_hmi_date_obs_map(ref_file)
        s_map = source_map
        if resolution is not None:
            s_map = s_map.resample(resolution * u.pix)
        map_data = load_map_data(s_map)
        # The map frame and latent coordinate now share the same DATE-OBS time.
        # T_REC remains only the JSOC/calibration record identity.
        map_data['time'] = SphericalDataModule._hmi_date_obs(ref_file)
        map_data['detector_mapping'] = hmi_detector_mapping(source_map, output_shape=s_map.data.shape)
        stokes = load_stokes_data(data, resolution)

        if polarization_filter is not None:
            # Work on the image before flattening; training targets remain unsmoothed.
            yy, xx = np.ogrid[:stokes.shape[0], :stokes.shape[1]]
            radius = np.hypot(
                (xx + 1 - s_map.meta['CRPIX1']) * s_map.scale.axis1.to_value(u.arcsec/u.pix),
                (yy + 1 - s_map.meta['CRPIX2']) * s_map.scale.axis2.to_value(u.arcsec/u.pix),
            ) / s_map.rsun_obs.to_value(u.arcsec)
            valid = np.isfinite(map_data['spherical_coords']).all(axis=-1)
            valid &= np.isfinite(map_data['mu']) & (map_data['mu'] >= kwargs.get('mu_limit', 1e-3))
            valid &= np.isfinite(map_data['v_obs_los'])
            mask, _, stats = polarization_selection(stokes, radius, valid, **polarization_filter)

            self.polarization_stats = stats
            kwargs['pixel_mask'] = mask

        # The optional filter is applied before flattening and cache shuffling.
        with fits.open(ref_file, memmap=False) as hdus:
            native_header = next(h.header.copy() for h in hdus if 'DATE-OBS' in h.header)
        timing, _ = observation_times(native_header)
        kwargs['observation_metadata'] = {'source_paths': [str(os.path.abspath(p)) for p in np.asarray(data).ravel()],
                                          'dataset_type': 'hmi', 'time': timing,
                                          'original_header': native_header.tostring(sep='\n', endcard=False, padding=False)}
        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)

    def __getitem__(self, *args):
        out = super().__getitem__(*args)
        detector_coords = out.pop('detector_coords')
        out.update(sample_hmi_spectral_response(self.spectral_response_file, detector_coords))
        return out

    @staticmethod
    def _validate_quality(data):
        bad_segments = []
        missing_quality = []
        for path in np.asarray(data, dtype=object).reshape(-1):
            with fits.open(path, memmap=False) as hdus:
                header = next((hdu.header for hdu in hdus if 'T_REC' in hdu.header), None)
            if header is None or 'QUALITY' not in header:
                missing_quality.append(os.fspath(path))
                continue
            try:
                quality = int(header['QUALITY'])
            except (TypeError, ValueError) as error:
                raise ValueError(f'Invalid HMI QUALITY value in {path!r}: {header["QUALITY"]!r}.') from error
            if quality != 0:
                bad_segments.append((os.fspath(path), quality))
        if missing_quality:
            raise ValueError(f'HMI segments are missing the QUALITY keyword: {missing_quality[:3]}')
        if bad_segments:
            summary = ', '.join(f'{os.path.basename(path)}={quality:#x}' for path, quality in bad_segments[:5])
            raise ValueError(f'Refusing nonzero-quality HMI acquisition: {summary}')

    @staticmethod
    def _validate_segment_alignment(data, tolerance_pixels=1e-3):
        """Reject HMI segments that cannot be stacked on one pixel grid."""
        paths = np.asarray(data, dtype=object).reshape(-1)
        if paths.size == 0:
            raise ValueError('Cannot validate an empty HMI acquisition.')
        if tolerance_pixels < 0:
            raise ValueError('Segment-alignment tolerance must be non-negative.')

        def metadata(path):
            # HDU headers expose the image dimensions even for compressed FITS;
            # accessing .data here would decompress every plane before loading.
            with fits.open(path, memmap=False) as hdus:
                index = next((i for i, hdu in enumerate(hdus)
                              if hdu.header.get('NAXIS') == 2), None)
                if index is None:
                    raise ValueError(f'HMI segment has no 2D image HDU: {path}.')
                header = observation_header(hdus, index)
            shape = (header['NAXIS2'], header['NAXIS1'])
            origin = tuple(None if header.get(key) is None else float(header[key])
                           for key in ('CCD_X0', 'CCD_Y0'))
            return shape, observation_times(header)[1], origin, WCS(header, naxis=2)

        reference_path = paths[0]
        reference_shape, reference_time, reference_origin, reference_wcs = metadata(reference_path)
        height, width = reference_shape
        x = np.array([0., width-1., 0., width-1., (width-1)/2])
        y = np.array([0., 0., height-1., height-1., (height-1)/2])
        world = reference_wcs.pixel_to_world_values(x, y)
        errors = []
        for path in paths[1:]:
            shape, segment_time, segment_origin, wcs = metadata(path)
            segment_name = os.path.basename(os.fspath(path))
            if shape != reference_shape:
                errors.append(f'{segment_name}: shape {shape} != {reference_shape}')
                continue
            if segment_time != reference_time:
                errors.append(f'{segment_name}: DATE-OBS {segment_time.isoformat()} != {reference_time.isoformat()}')
            if (None in reference_origin) != (None in segment_origin):
                errors.append(f'{segment_name}: detector origin {segment_origin} != {reference_origin}')
            elif None not in reference_origin and not np.allclose(
                    segment_origin, reference_origin, rtol=0, atol=tolerance_pixels):
                errors.append(f'{segment_name}: detector origin {segment_origin} != {reference_origin}')
            sx, sy = wcs.world_to_pixel_values(*world)
            error = np.hypot(np.asarray(sx)-x, np.asarray(sy)-y)
            if not np.isfinite(error).all() or np.any(error > tolerance_pixels):
                errors.append(f'{segment_name}: spatial WCS differs by more than {tolerance_pixels} pixels')

        if errors:
            reference_name = os.path.basename(os.fspath(reference_path))
            details = '; '.join(errors[:5])
            if len(errors) > 5:
                details += f'; ... and {len(errors) - 5} more'
            raise ValueError(
                f'Misaligned HMI segments relative to {reference_name}: {details}'
            )



class FITSSphericalDataset(SphericalDataset):
    """A generic correctly-georeferenced FITS cube projected onto the Sun."""

    def __init__(self, data, *, data_hdu=0, spatial_header_hdu=None,
                 axis_order=None, stokes_order='IQUV', stokes_signs=None,
                 wavelength_center=None, wavelength_grid=None, wavelength_unit='Angstrom',
                 wavelength_grid_is_offsets=True, wavelength_hdu=None, wavelength_range=None,
                 spatial_wcs_axes=None, observer=None, time_reference='observation',
                 time_scale=None, line_id='custom', continuum_index=None,
                 time_hdu=None, **kwargs):
        spec = dict(data_hdu=data_hdu, spatial_header_hdu=spatial_header_hdu,
                    axis_order=axis_order, stokes_order=stokes_order, stokes_signs=stokes_signs,
                    wavelength_center=wavelength_center, wavelength_grid=wavelength_grid,
                    wavelength_unit=wavelength_unit, wavelength_grid_is_offsets=wavelength_grid_is_offsets,
                    wavelength_hdu=wavelength_hdu, wavelength_range=wavelength_range, spatial_wcs_axes=spatial_wcs_axes,
                    observer=observer, time_reference=time_reference, time_scale=time_scale,
                    line_id=line_id, continuum_index=continuum_index)
        observation = read_fits_observation(data, **spec)
        observation['metadata']['reader'].update({k: v for k, v in spec.items() if v is not None})
        observation['metadata']['dataset_type'] = 'fits'
        self._init_observation(observation, line_id, kwargs, data, time_hdu)

    def _init_observation(self, observation, line_id, kwargs, path, time_hdu=None):
        reference = Map(observation['stokes'][..., 0, 0], observation['header'])
        map_data = load_map_data(reference)
        map_data['time'] = observation['model_time']
        if time_hdu is not None:
            offsets = np.asarray(fits.getdata(path, time_hdu), dtype=np.float64)
            if offsets.shape != observation['stokes'].shape[:2] or not np.isfinite(offsets).all():
                raise ValueError('time_hdu must contain a finite per-pixel time-offset map in seconds.')
            map_data['pixel_time_offsets'] = offsets
            observation['metadata']['reader']['time_hdu'] = time_hdu
        detector_shape = kwargs.pop('detector_shape', None)
        detector_origin = kwargs.pop('detector_origin', [0, 0])
        detector_step = np.asarray(kwargs.pop('detector_step', [1, 1]), dtype=float)
        if detector_shape is not None:
            shape = np.asarray(detector_shape, dtype=float)
            origin = np.asarray(detector_origin, dtype=float)
            if shape.shape != (2,) or np.any(shape < 2) or origin.shape != (2,) or detector_step.shape != (2,) or np.any(detector_step <= 0):
                raise ValueError('detector_shape is [height,width] >=2, detector_origin is [x,y].')
            denominator = shape[::-1]-1
            extent = origin + (np.asarray(observation['stokes'].shape[1::-1])-1)*detector_step
            if np.any(origin < 0) or np.any(extent > denominator):
                raise ValueError('The observed image extends outside the configured physical detector.')
            map_data['detector_mapping'] = origin/denominator, detector_step/denominator
        elif (observation['metadata'].get('instrument') == 'PHI' and
              'PXBEG1' in observation['header'] and 'PXBEG2' in observation['header']):
            height, width = observation['stokes'].shape[:2]
            detector_size = 2048 if observation['metadata'].get('instrument') == 'PHI' else max(height, width)
            step = np.array([observation['header'].get('NBIN1',1),observation['header'].get('NBIN2',1)],dtype=float)
            origin = np.array([observation['header']['PXBEG1']-1, observation['header']['PXBEG2']-1]) + (step-1)/2
            map_data['detector_mapping'] = origin / (detector_size-1), step/(detector_size-1)
        SphericalDataset.__init__(self, observation['stokes'], map_data,
                       observation['wavelength_config'], line_id=line_id,
                       observation_metadata=observation['metadata'], **kwargs)


class HinodeSphericalDataset(FITSSphericalDataset):
    """Prepared Hinode/SP rasters use the same spherical WCS and geometry API.

    Use data.prepare_hinode.prepare_hinode_raster to combine slit products
    with an explicitly validated raster WCS, preserving their acquisition times.
    """
    def __init__(self, data, **kwargs):
        kwargs.setdefault('line_id', 'fe-i-6302')
        kwargs.setdefault('axis_order', ['stokes', 'wavelength', 'y', 'x'])
        super().__init__(data, **kwargs)
        self.observation['dataset_type'] = 'hinode'


class PHIHRTSphericalDataset(FITSSphericalDataset):
    def __init__(self, data, *, time_reference='observation', line_id='fe-i-6173', **kwargs):
        read_keys = {'data_hdu', 'spatial_header_hdu', 'axis_order', 'stokes_order', 'stokes_signs',
                     'wavelength_center', 'wavelength_grid', 'wavelength_unit',
                     'wavelength_grid_is_offsets', 'wavelength_hdu', 'wavelength_range', 'spatial_wcs_axes',
                     'observer', 'time_scale', 'continuum_index'}
        spec = {k: kwargs.pop(k) for k in list(kwargs) if k in read_keys}
        observation = read_phi_observation(data, telescope=self.telescope,
                      time_reference=time_reference, line_id=line_id, **spec)
        observation['metadata']['dataset_type'] = 'phi-' + self.telescope
        observation['metadata']['reader'].update(time_reference=time_reference,
                                                 continuum_index=observation['continuum_index'])
        self._init_observation(observation, line_id, kwargs, data)

    telescope = 'hrt'


class PHIFDTSphericalDataset(PHIHRTSphericalDataset):
    telescope = 'fdt'


class TestSphericalDataset(SphericalDataset):

    def __init__(self, data, noise=0, *args, **kwargs):
        wavelength_config = hmi_wavelength_config()

        stokes_i_files, _, _, _ = data
        ref_file = stokes_i_files[0]
        s_map = load_hmi_date_obs_map(ref_file)
        map_data = load_map_data(s_map)
        stokes = load_stokes_data(data)

        kwargs['observation_metadata'] = {'source_paths': [str(os.path.abspath(p)) for p in np.asarray(data).ravel()], 'dataset_type': 'test'}
        # add noise
        normal_noise = np.random.normal(size=stokes.shape, scale=noise)
        stokes += normal_noise

        super().__init__(stokes, map_data, wavelength_config, *args, **kwargs)


def load_stokes_data(files, resolution: Optional[tuple[int, int]] = None):
    """Read each plane once into the float32 training cube, without stack copies."""
    files = np.asarray(files)
    if files.ndim != 2 or files.shape[0] != 4 or not files.shape[1]:
        raise ValueError('Stokes files must contain four equally sampled I/Q/U/V rows.')
    stokes = None
    for index, path in enumerate(files.ravel()):
        image = (fits.getdata(path, memmap=False) if resolution is None else
                 load_hmi_date_obs_map(path).resample(np.asarray(resolution)*u.pix).data)
        image = np.asarray(image)
        if image.ndim != 2:
            raise ValueError(f'Stokes segment must be a 2D image: {path}.')
        if stokes is None:
            stokes = np.empty((*image.shape, 4, files.shape[1]), dtype=np.float32)
        elif image.shape != stokes.shape[:2]:
            raise ValueError(f'Misaligned Stokes segment shape {image.shape}: {path}.')
        component, wavelength = divmod(index, files.shape[1])
        stokes[..., component, wavelength] = image
    return stokes


def require_observer_metadata(s_map):
    """Reject incomplete geometry instead of letting SunPy infer an observer."""
    required = ('DSUN_OBS',)
    missing = [key for key in required if key not in s_map.meta]
    if not any(key in s_map.meta for key in ('HGLN_OBS', 'CRLN_OBS', 'HAEX_OBS', 'HEEX_OBS')):
        missing.append('observer longitude (HGLN_OBS/CRLN_OBS or Cartesian observer)')
    if not any(key in s_map.meta for key in ('HGLT_OBS', 'CRLT_OBS', 'HAEZ_OBS', 'HEEZ_OBS')):
        missing.append('observer latitude (HGLT_OBS/CRLT_OBS or Cartesian observer)')
    if missing:
        raise ValueError(f'Explicit observer geometry metadata are required: {missing}')


def load_map_data(s_map):
    require_observer_metadata(s_map)
    time = s_map.date.to_datetime()
    # convert world coordinates to cartesian
    map_coords = all_coordinates_from_map(s_map)

    carrington_coords = map_coords.transform_to(frames.HeliographicCarrington)
    lat, lon = carrington_coords.lat.to_value(u.rad), carrington_coords.lon.to_value(u.rad)
    r = np.ones_like(lon)  # carrington_coords.radius
    # r = r * u.solRad if r.unit == u.dimensionless_unscaled else r
    # convert latitude to colatitude
    spherical_coords = np.stack([r, np.pi / 2 - lat, lon], -1)

    # create observer transform
    pAng = image_p_angle(s_map) * u.rad
    obs_lat = s_map.carrington_latitude.to(u.rad)
    obs_lon = s_map.carrington_longitude.to(u.rad)

    observer_distance = (s_map.dsun / s_map.rsun_meters).to_value(u.dimensionless_unscaled)
    ray, mu = observer_ray_and_mu(
        lon, lat, obs_lon.to_value(u.rad), obs_lat.to_value(u.rad), observer_distance,
    )
    mu = mu.astype(np.float32)

    # load observer velocity
    v_obs_los = load_v_observer_LOS(s_map, ray=ray).astype(np.float32)

    return {'mu': mu, 'v_obs_los': v_obs_los,
            'obs_lat': obs_lat, 'obs_lon': obs_lon, 'pAng': pAng,
            'observer_distance': observer_distance,
            'observer_velocity': observer_velocity_from_header(s_map.meta).tolist(),
            'time': time, 'spherical_coords': spherical_coords, 'wcs': s_map.wcs,
            'instrument': s_map.instrument, 'header': s_map.fits_header}


def hmi_detector_mapping(s_map, output_shape=None, ccd_size=4096):
    """Compact affine map from output [column,row] to normalized CCD coordinates."""
    if ccd_size < 2:
        raise ValueError('ccd_size must be at least 2.')
    height, width = s_map.data.shape[-2:]
    output_height, output_width = (height, width) if output_shape is None else output_shape[-2:]
    if height == ccd_size and width == ccd_size:
        origin = np.zeros(2, dtype=np.float64)
    elif 'CCD_X0' in s_map.meta and 'CCD_Y0' in s_map.meta:
        origin = np.array([float(s_map.meta['CCD_X0']), float(s_map.meta['CCD_Y0'])])
    else:
        raise ValueError('HMI cutouts require CCD_X0 and CCD_Y0 metadata. '
                         'Prepare the cutout with validated WCS and original CCD detector coordinates.')
    if not np.isfinite(origin).all():
        raise ValueError('HMI detector origin must be finite.')
    # SunPy resample maps pixel centers, not the outermost native centers.
    # A two-pixel view of a 4096-pixel detector samples centers 1023.5 and
    # 3071.5 rather than the detector endpoints 0 and 4095.
    step = np.array([width/output_width, height/output_height], dtype=np.float64)
    origin += (step-1)/2
    return origin / (ccd_size - 1), step / (ccd_size - 1)


def load_hmi_detector_coords(s_map, output_shape=None, ccd_size=4096):
    """Map an HMI image or cutout to normalized full-detector [x,y] coordinates."""
    shape = s_map.data.shape[-2:] if output_shape is None else output_shape[-2:]
    origin, step = hmi_detector_mapping(s_map, output_shape, ccd_size)
    row, column = np.indices(shape)
    coords = np.stack([column, row], axis=-1) * step + origin
    return np.clip(coords, 0, 1).astype(np.float32)
