import json
from datetime import datetime

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.time import Time

from spinnme.data.hmi_time import load_hmi_date_obs_map
from spinnme.data.hmi_transmission import (
    MANIFEST_FORMAT,
    hmi_acquisition_key,
    resolve_hmi_transmission_profile,
)
from spinnme.instrument import hmi_wavelength_config
from spinnme.loader.spherical import (
    HMISphericalDataset,
    SphericalDataModule,
    _summarize_physics_coordinates,
    load_hmi_detector_coords,
    sample_hmi_spectral_response,
)
from spinnme.train.data_loader import TimeStratifiedBatchSampler


def _create_segment_files(directory, acquisition, num_wl=6):
    for stokes_id in 'IQUV':
        for wavelength_index in range(num_wl):
            (directory / f'{acquisition}.{stokes_id}{wavelength_index}.fits').touch()

def test_physics_coordinate_summary_is_seam_safe():
    longitude = np.deg2rad(np.array([179.0, -179.0]))
    spherical = np.stack([
        np.ones(2),
        np.full(2, np.pi / 2),
        longitude,
    ], axis=-1)

    bounds = _summarize_physics_coordinates(spherical)

    assert abs(abs(np.rad2deg(bounds['longitude_center_rad'])) - 180) < 1e-10
    assert np.rad2deg(bounds['longitude_offset_rad']) == pytest.approx((-1.0, 1.0))

def test_iquv_files_are_grouped_by_acquisition(tmp_path):
    _create_segment_files(tmp_path, 'hmi.S_720s.20240507_120000_TAI.1')
    _create_segment_files(tmp_path, 'hmi.S_720s.20240507_121200_TAI.1')

    loader = SphericalDataModule.__new__(SphericalDataModule)
    files = loader._load_IQUV_files(str(tmp_path))

    assert files.shape == (2, 4, 6)
    assert files[0, 0, 0].endswith('20240507_120000_TAI.1.I0.fits')
    assert files[1, 3, 5].endswith('20240507_121200_TAI.1.V5.fits')

def test_iquv_loader_rejects_a_missing_timestamp_segment(tmp_path):
    acquisition = 'hmi.S_720s.20240507_120000_TAI.1'
    _create_segment_files(tmp_path, acquisition)
    (tmp_path / f'{acquisition}.U3.fits').unlink()

    loader = SphericalDataModule.__new__(SphericalDataModule)
    with pytest.raises(ValueError, match=r"missing=.*\('U', 3\)"):
        loader._load_IQUV_files(str(tmp_path))

def test_training_time_range_is_inclusive(monkeypatch):
    files = np.asarray([[[name]] for name in ('before', 'start', 'center', 'end', 'after')], dtype=object)
    observation_times = {
        'before': datetime(2024, 5, 5, 0),
        'start': datetime(2024, 5, 5, 12),
        'center': datetime(2024, 5, 7, 12),
        'end': datetime(2024, 5, 9, 12),
        'after': datetime(2024, 5, 10, 0),
    }
    monkeypatch.setattr(
        SphericalDataModule,
        '_observation_time_for_file_group',
        lambda file_group, ds_type, config=None: observation_times[np.asarray(file_group).reshape(-1)[0]],
    )

    selected = SphericalDataModule._filter_files_by_time_range(
        files,
        ['2024-05-05T12:00:00', '2024-05-09T12:00:00'],
        'hmi',
    )

    assert [group.reshape(-1)[0] for group in selected] == ['start', 'center', 'end']

def test_hmi_tai_time_parser_accepts_fractional_seconds():
    assert SphericalDataModule._parse_observation_time(
        '2024.05.01_00:00:00.000_TAI'
    ) == datetime(2024, 4, 30, 23, 59, 23)
    assert SphericalDataModule._parse_observation_time(
        '20240501_000000.125_TAI'
    ) == datetime(2024, 4, 30, 23, 59, 23, microsecond=125000)

def test_hmi_training_time_comes_from_fits_header(tmp_path):
    segment_path = tmp_path / 'sample.I0.fits'
    fits.PrimaryHDU(
        header=fits.Header({
            'T_REC': '2024.05.07_12:00:00.000_TAI',
            'T_OBS': '2024.05.07_12:00:00.000_TAI',
            'DATE-OBS': '2024-05-07T11:58:09.400',
            'HCAMID': 3,
        })
    ).writeto(segment_path)

    observation_time = SphericalDataModule._observation_time_for_file_group(
        np.asarray([[segment_path]], dtype=object), 'hmi'
    )

    assert observation_time == datetime(2024, 5, 7, 11, 59, 23)

def test_hmi_latent_time_requires_t_obs(tmp_path):
    segment_path = tmp_path / 'sample.I0.fits'
    fits.PrimaryHDU(
        header=fits.Header({'T_REC': '2024.05.07_12:00:00.000_TAI', 'HCAMID': 3})
    ).writeto(segment_path)

    with pytest.raises(ValueError, match='T_OBS'):
        SphericalDataModule._hmi_date_obs(segment_path)

def test_hmi_map_coordinate_frame_uses_t_obs(tmp_path):
    segment_path = tmp_path / 'sample.I0.fits'
    header = fits.Header({
        'TELESCOP': 'SDO/HMI',
        'INSTRUME': 'HMI_SIDE1',
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CRPIX1': 1.5,
        'CRPIX2': 1.5,
        'CRVAL1': 0.0,
        'CRVAL2': 0.0,
        'CDELT1': 0.5,
        'CDELT2': 0.5,
        'DATE-OBS': '2024-05-07T11:58:09.400',
        'T_OBS': '2024.05.07_12:00:00.000_TAI',
        'DATE-AVG': '2024-05-07T11:59:00.000',
        'DSUN_OBS': 1.5e11,
        'RSUN_REF': 6.957e8,
        'HGLN_OBS': 0.0,
        'HGLT_OBS': 0.0,
    })
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32), header=header).writeto(segment_path)

    hmi_map = load_hmi_date_obs_map(segment_path)

    assert hmi_map.date.isot == '2024-05-07T11:58:09.400'
    assert hmi_map.reference_date.utc.isot == '2024-05-07T11:59:23.000'
    assert hmi_map.coordinate_frame.obstime.utc.isot == '2024-05-07T11:59:23.000'
    assert hmi_map.observer_coordinate.obstime.utc.isot == '2024-05-07T11:59:23.000'
    assert hmi_map.meta['T_OBS'] == header['T_OBS']
    assert hmi_map.meta['DATE-AVG'] == '2024-05-07T11:59:23.000'

def _write_hmi_wcs_segment(path, *, crpix1=1.5, date_obs='2024-05-07T11:58:09.400'):
    header = fits.Header({
        'TELESCOP': 'SDO/HMI',
        'INSTRUME': 'HMI_SIDE1',
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CRPIX1': crpix1,
        'CRPIX2': 1.5,
        'CRVAL1': 0.0,
        'CRVAL2': 0.0,
        'CDELT1': 0.5,
        'CDELT2': 0.5,
        'DATE-OBS': date_obs,
        'T_OBS': Time(date_obs).tai.isot + '_TAI',
        'DSUN_OBS': 1.5e11,
        'RSUN_REF': 6.957e8,
        'HGLN_OBS': 0.0,
        'HGLT_OBS': 0.0,
        'CCD_X0': 100.0,
        'CCD_Y0': 200.0,
    })
    fits.PrimaryHDU(np.zeros((2, 2), dtype=np.float32), header=header).writeto(path)

def test_hmi_segment_alignment_accepts_matching_wcs(tmp_path):
    reference_path = tmp_path / 'sample.I0.fits'
    segment_path = tmp_path / 'sample.Q0.fits'
    _write_hmi_wcs_segment(reference_path)
    _write_hmi_wcs_segment(segment_path)

    HMISphericalDataset._validate_segment_alignment([reference_path, segment_path])

def test_hmi_segment_alignment_rejects_date_obs_mismatch(tmp_path):
    reference_path = tmp_path / 'sample.I0.fits'
    segment_path = tmp_path / 'sample.Q0.fits'
    _write_hmi_wcs_segment(reference_path)
    _write_hmi_wcs_segment(segment_path, date_obs='2024-05-07T11:58:10.400')

    with pytest.raises(ValueError, match=r'Misaligned HMI segments.*T_OBS'):
        HMISphericalDataset._validate_segment_alignment([reference_path, segment_path])

def test_training_time_range_rejects_empty_selection(monkeypatch):
    files = np.asarray([[['sample']]], dtype=object)
    monkeypatch.setattr(
        SphericalDataModule,
        '_observation_time_for_file_group',
        lambda file_group, ds_type, config=None: datetime(2024, 5, 1),
    )

    with pytest.raises(ValueError, match='inclusive time range'):
        SphericalDataModule._filter_files_by_time_range(
            files,
            ['2024.05.05_12:00:00_TAI', '2024.05.09_12:00:00_TAI'],
            'hmi',
        )

@pytest.mark.parametrize(('available', 'requested', 'expected'), [
    (5, 2, [0, 4]),
    (7, 3, [0, 3, 6]),
    (4, 4, [0, 1, 2, 3]),
])
def test_n_samples_selects_exact_evenly_spaced_count(available, requested, expected):
    files = np.arange(available)
    selected = SphericalDataModule._select_evenly_spaced_samples(files, requested)
    np.testing.assert_array_equal(selected, expected)

def test_n_samples_rejects_more_than_available():
    with pytest.raises(ValueError, match='exceeds'):
        SphericalDataModule._select_evenly_spaced_samples(np.arange(2), 3)

@pytest.mark.parametrize('num_workers', [0, 4])
def test_validation_loader_uses_training_worker_count(num_workers):
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return index

    data_module = SphericalDataModule.__new__(SphericalDataModule)
    data_module.num_workers = num_workers
    data_module.num_dataset_workers = 16
    data_module.prefetch_factor = 2
    data_module.persistent_workers = True
    data_module.pin_memory = False
    data_module.batch_size = 4096
    data_module.valid_dataset = Dataset()
    assert data_module.val_dataloader().num_workers == num_workers

@pytest.mark.parametrize(('batch_size', 'dataset_batch_size'), [(511, 512), (1000, 512), (0, 512)])
def test_invalid_combined_batch_sizes_are_rejected(batch_size, dataset_batch_size):
    data_module = SphericalDataModule.__new__(SphericalDataModule)
    data_module.batch_size = batch_size
    data_module.dataset_batch_size = dataset_batch_size
    with pytest.raises(ValueError):
        data_module._validate_batch_sizes()

def test_time_stratified_sampler_regroups_chunks_each_epoch():
    sampler = TimeStratifiedBatchSampler(
        [[(0, index) for index in range(8)], [(1, index) for index in range(8)]],
        chunks_per_batch=4,
    )
    first_epoch = list(iter(sampler))
    second_epoch = list(iter(sampler))

    assert first_epoch != second_epoch
    for batch in first_epoch + second_epoch:
        assert len(batch) == 4
        assert {dataset_index for dataset_index, _ in batch} == {0, 1}
    assert sorted(index for batch in first_epoch for index in batch) == [
        *[(0, index) for index in range(8)],
        *[(1, index) for index in range(8)],
    ]

def test_hmi_quality_check_rejects_nonzero_quality(tmp_path):
    paths = []
    for index, quality in enumerate((0, 0x20)):
        path = tmp_path / f'segment_{index}.fits'
        image = fits.ImageHDU(np.zeros((1, 1)))
        image.header['T_REC'] = '2024.05.07_12:00:00_TAI'
        image.header['QUALITY'] = quality
        fits.HDUList([fits.PrimaryHDU(), image]).writeto(path)
        paths.append(path)

    with pytest.raises(ValueError, match='nonzero-quality'):
        HMISphericalDataset._validate_quality(np.asarray(paths, dtype=object))

def test_hmi_nominal_wavelength_configuration():
    config = hmi_wavelength_config()
    np.testing.assert_allclose(config['wavelength_grid'].to_value('Angstrom'),
                               [0.1720, 0.1032, 0.0344, -0.0344, -0.1032, -0.1720])
    assert config['wavelength_center'].to_value('Angstrom') == pytest.approx(6173.3433)



def test_hmi_cutout_without_detector_origin_is_rejected():
    data = np.zeros((3, 4), dtype=np.float32)
    header = fits.Header({'CRPIX1': -997.0, 'CRPIX2': -1997.0})
    hmi_map = type('MapStub', (), {'data': data, 'meta': header})()

    with pytest.raises(ValueError, match='CCD_X0 and CCD_Y0'):
        load_hmi_detector_coords(hmi_map)

def test_hmi_cutout_uses_preserved_detector_origin():
    data = np.zeros((2, 3), dtype=np.float32)
    header = fits.Header({'CRPIX1': 1.0, 'CRPIX2': 1.0, 'CCD_X0': 1200.0, 'CCD_Y0': 800.0})
    hmi_map = type('MapStub', (), {'data': data, 'meta': header})()

    coords = load_hmi_detector_coords(hmi_map)

    np.testing.assert_allclose(coords[0, 0], [1200 / 4095, 800 / 4095])
    np.testing.assert_allclose(coords[-1, -1], [1202 / 4095, 801 / 4095])

def test_full_disk_hmi_uses_exact_detector_pixels_despite_shifted_crpix():
    data = np.zeros((4096, 4096), dtype=np.uint8)
    header = fits.Header({'CRPIX1': 2037.2, 'CRPIX2': 2047.7})
    hmi_map = type('MapStub', (), {'data': data, 'meta': header})()

    coords = load_hmi_detector_coords(hmi_map, output_shape=(2, 2))

    np.testing.assert_allclose(coords[0, 0], [1023.5/4095, 1023.5/4095])
    np.testing.assert_allclose(coords[-1, -1], [3071.5/4095, 3071.5/4095])

def test_dataset_can_resolve_response_from_calibration_directory(tmp_path):
    profile_path = tmp_path / 'profile.npz'
    weights = np.ones((2, 2, 6, 1))
    np.savez(
        profile_path,
        offsets=np.zeros((6, 1)),
        weights=weights,
        continuum_weights=np.zeros(weights.shape[:-1]),
    )
    t_rec = '2024.05.07_12:00:00_TAI'
    acquisition_key = hmi_acquisition_key(t_rec, 2)
    manifest = {
        'format': MANIFEST_FORMAT,
        'profiles': {'calibration': {'file': profile_path.name}},
        'acquisitions': {acquisition_key: {'profile': 'calibration'}},
    }
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    segment_path = tmp_path / 'hmi.S_720s.20240507_120000_TAI.1.I0.fits'
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(np.zeros((2, 2)))
    image.header['T_REC'] = t_rec
    image.header['HCAMID'] = 2
    fits.HDUList([primary, image]).writeto(segment_path)

    resolved = resolve_hmi_transmission_profile(tmp_path, segment_path)

    assert resolved == profile_path

    dataset = HMISphericalDataset.__new__(HMISphericalDataset)
    dataset.instrument_id = 'HMI'
    dataset.acquisition_key = acquisition_key
    dataset.spectral_response_file = str(resolved)
    data_module = SphericalDataModule.__new__(SphericalDataModule)
    data_module.spectral_response_files = {}
    data_module.spectral_response_by_acquisition = {}
    data_module._register_hmi_dataset_response(dataset)

    assert data_module.spectral_response_files == {'HMI': [str(profile_path)]}
    assert data_module.spectral_response_by_acquisition == {
        'HMI': {acquisition_key: str(profile_path)}
    }

def test_dataset_samples_spatial_response_for_current_pixels(tmp_path):
    offsets = np.broadcast_to(np.array([-0.01, 0.01], dtype=np.float32), (6, 2)).copy()
    weights = np.zeros((2, 2, 6, 2), dtype=np.float32)
    first_weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    weights[..., 0] = first_weight[..., None]
    weights[..., 1] = 1 - weights[..., 0]
    profile_path = tmp_path / 'spatial_response.npz'
    np.savez(
        profile_path,
        offsets=offsets,
        weights=weights,
        continuum_weights=np.zeros(weights.shape[:-1], dtype=np.float32),
    )

    sampled = sample_hmi_spectral_response(
        profile_path,
        torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
    )

    assert sampled['spectral_offsets'].shape == (3, 6, 2)
    assert sampled['spectral_weights'].shape == (3, 6, 2)
    assert sampled['continuum_weights'].shape == (3, 6)
    torch.testing.assert_close(sampled['spectral_weights'][0, :, 0], torch.ones(6))
    torch.testing.assert_close(sampled['spectral_weights'][1], torch.full((6, 2), 0.5))
    torch.testing.assert_close(sampled['spectral_weights'][2, :, 1], torch.zeros(6))
    torch.testing.assert_close(
        sampled['spectral_weights'].sum(-1) + sampled['continuum_weights'],
        torch.ones(3, 6),
    )

def test_hmi_loader_needs_no_observer_velocity_keywords(tmp_path):
    acquisition = 'hmi.S_720s.20240507_120000_TAI.1'
    data = []
    for component in 'IQUV':
        paths = []
        for index in range(6):
            path = tmp_path/f'{acquisition}.{component}{index}.fits'
            _write_hmi_wcs_segment(path)
            with fits.open(path, mode='update') as hdus:
                hdus[0].header.update(T_REC='2024.05.07_12:00:00_TAI', HCAMID=2, QUALITY=0)
                hdus[0].data[:] = 1.
            paths.append(str(path))
        data.append(paths)
    calibration = tmp_path/'transmission'
    calibration.mkdir()
    np.savez(calibration/'profile.npz', offsets=np.zeros((6, 1)),
             weights=np.ones((1, 1, 6, 1)), continuum_weights=np.zeros((1, 1, 6)))
    key = hmi_acquisition_key('2024.05.07_12:00:00_TAI', 2)
    (calibration/'manifest.json').write_text(json.dumps({
        'format': MANIFEST_FORMAT,
        'profiles': {'profile': {'file': 'profile.npz'}},
        'acquisitions': {key: {'profile': 'profile'}},
    }))
    dataset = HMISphericalDataset(data, transmission_profile_directory=calibration,
                                  ds_id='hmi', instrument_id='HMI',
                                  seconds_per_dt=3600., Rs_per_ds=1.,
                                  ref_time=datetime(2024, 5, 7), stokes_normalization=1.,
                                  work_directory=str(tmp_path/'cache'), batch_size=4, shuffle=False)
    np.testing.assert_array_equal(dataset.observer_velocity, [0., 0., 0.])
    torch.testing.assert_close(dataset[0]['v_obs_los'], torch.zeros((4, 1)))
    assert torch.isfinite(dataset[0]['stokes']).all()


def test_hmi_response_cache_reloads_replaced_calibration(tmp_path, monkeypatch):
    import os

    from spinnme.loader import spherical
    profile = tmp_path/'response.npz'
    def write(values):
        np.savez(profile, offsets=np.tile([-.1, .1], (6, 1)).astype('f4'),
                 weights=np.tile(values, (6, 1)).astype('f4'),
                 continuum_weights=np.zeros(6, dtype='f4'))
    write([1, 0])
    original_stat = profile.stat()
    load = spherical.load_hmi_transmission_profile
    calls = []
    def counted(path):
        calls.append(path)
        return load(path)
    monkeypatch.setattr(spherical, 'load_hmi_transmission_profile', counted)
    coords = torch.tensor([[.5, .5]])
    first = sample_hmi_spectral_response(profile, coords)
    sample_hmi_spectral_response(profile, coords)
    assert len(calls) == 1  # Batches reuse the tensors without rereading the NPZ.
    write([0, 1])
    os.utime(profile, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = sample_hmi_spectral_response(profile, coords)
    assert len(calls) == 2
    torch.testing.assert_close(first['spectral_weights'][0, 0], torch.tensor([1., 0.]))
    torch.testing.assert_close(second['spectral_weights'][0, 0], torch.tensor([0., 1.]))


@pytest.mark.parametrize('compressed', [False, True])
def test_hmi_alignment_checks_only_headers(tmp_path, monkeypatch, compressed):
    from test_fits_observations import header

    from spinnme.loader import spherical
    paths = []
    for index in range(2):
        path = tmp_path/f'plane{index}.fits'
        meta = header()
        meta['T_OBS'] = Time(meta['DATE-OBS']).tai.isot + '_TAI'
        if compressed:
            fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(np.ones((3, 5)), meta)]).writeto(path)
        else:
            fits.writeto(path, np.ones((3, 5)), meta)
        paths.append(path)
    def unexpected(*args, **kwargs):
        raise AssertionError('Alignment must not load Stokes pixels')
    monkeypatch.setattr(spherical, 'load_hmi_map', unexpected)
    monkeypatch.setattr(fits.hdu.image._ImageBaseHDU, '_get_scaled_image_data', unexpected)
    HMISphericalDataset._validate_segment_alignment(paths)


def test_stokes_loading_preserves_component_and_wavelength_order(tmp_path):
    from spinnme.loader.spherical import load_stokes_data
    files = []
    expected = np.empty((3, 5, 4, 2), dtype='float32')
    for component in range(4):
        row = []
        for wavelength in range(2):
            image = np.arange(15).reshape(3, 5)+component*100+wavelength*10
            path = tmp_path/f'{component}_{wavelength}.fits'
            fits.writeto(path, image.astype('float64'))
            row.append(path)
            expected[..., component, wavelength] = image
        files.append(row)
    actual = load_stokes_data(files)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
