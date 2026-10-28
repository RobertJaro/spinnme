"""Native Hinode scans assemble during inversion without intermediate FITS files."""
import copy
import json

import numpy as np
import pytest
import torch
from astropy.io import fits

from spinnme import load, run
from spinnme.configuration import resolve_config
from spinnme.data.hinode import read_hinode_slits
from spinnme.grids import GridGeometry, prepare_grid
from spinnme.loader.spherical import SphericalDataModule


def scan(tmp_path, descending=False):
    folder = tmp_path/'slits'
    folder.mkdir()
    for column, index in enumerate((0, 2, 5)):
        raw = np.full((4, 4, 112), column+1., dtype=np.float32)
        raw[0] += 100
        h = fits.Header({'SLITINDX': index, 'CRVAL1': 6303. if descending else 6300.9,
                         'CRPIX1': 1., 'CDELT1': -.0215 if descending else .0215,
                         'CUNIT1': 'Angstrom', 'XCEN': -200+index*.3, 'YCEN': -110.,
                         'XSCALE': .3, 'YSCALE': .32, 'CROTA2': .4, 'FGSPROT': .2636,
                         'SPCCDIX0': 510, 'SPCCDIX1': 513, 'CAMSSUM': 1,
                         'DATE_OBS': f'2024-03-23T02:00:{35+column*4:02d}'})
        fits.writeto(folder/f'slit_{index:03d}.fits', raw, h)
    return folder


@pytest.mark.parametrize('descending', [False, True])
def test_native_scan_preserves_gaps_spectra_calibration_and_timing(tmp_path, descending):
    folder = scan(tmp_path, descending)
    obs = read_hinode_slits(reversed(sorted(folder.glob('*.fits'))), line_id='fe-i-6302')
    assert obs['stokes'].shape[:3] == (4, 3, 4)
    assert obs['metadata']['slit_indices'] == [0, 2, 5]
    np.testing.assert_array_equal(obs['stokes'][0, :, 0, 0], [101, 102, 103])
    np.testing.assert_array_equal(obs['pixel_time_offsets'][0], [0, 4, 8])
    waves = obs['wavelength_config']['wavelength_grid'].to_value('Angstrom')+6302.4931
    assert np.all(waves >= 6301.99715)
    assert np.all(np.diff(waves)*(-1 if descending else 1) > 0)
    assert sorted(p.name for p in folder.iterdir()) == ['slit_000.fits', 'slit_002.fits', 'slit_005.fits']


def config(folder, tmp_path):
    c = resolve_config('hinode/sp', input_path=str(folder), output=str(tmp_path/'run'))
    c['work_path'] = str(tmp_path/'cache')
    c['data'].update(num_dataset_workers=0, num_workers=0, batch_size=12, dataset_batch_size=12)
    for d in c['data']['train_configs']:
        d.update(filter_nans=False, shuffle=False)
    return c


def test_native_geometry_matches_snapshot_and_resized_grid(tmp_path):
    folder = scan(tmp_path)
    c = config(folder, tmp_path)
    dm = SphericalDataModule(work_directory=str(tmp_path/'cache'), **c['data'])
    assert len(dm.train_datasets) == 2
    assert dm.image_shape == (4, 3)
    # The cache retains all channels; each line view selects native samples.
    assert dm.train_datasets[0].data['stokes'].shape[-1] == 112
    calibrated = 6300.9 + np.arange(112)*.0215
    for dataset, entry in zip(dm.train_datasets, c['data']['train_configs']):
        lower, upper = entry['wavelength_range']
        expected = calibrated[(calibrated >= lower) & (calibrated < upper)]
        actual = (dataset.wavelength_config['wavelength_grid'] +
                  dataset.wavelength_config['wavelength_center']).to_value('Angstrom')
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    dataset = dm.train_datasets[0]
    batch = dataset[0]
    assert torch.isfinite(batch['coords']).all()
    np.testing.assert_allclose(batch['coords'][:, -1].reshape(4, 3)[0], [0, 4/3600, 8/3600])
    row = dataset.observation
    json.dumps(row, allow_nan=False)
    geometry = GridGeometry(row).pixels(np.arange(12), Rs_per_ds=1, normalized_time=0, seconds_per_dt=3600)
    np.testing.assert_allclose(geometry['coords'], batch['coords'], atol=1e-7)
    # Native slit coefficients follow crop/resize pixel mappings as well.
    resized = prepare_grid(row, shape=(2, 2))
    assert np.asarray(resized['slit_pixel_world']).shape == (4, 2)
    assert np.isfinite(GridGeometry(resized).pixels(np.arange(4), Rs_per_ds=1,
                        normalized_time=0, seconds_per_dt=3600)['coords']).all()


@pytest.mark.integration
def test_native_scan_inversion_cache_and_source_free_export(tmp_path, capsys):
    folder = scan(tmp_path)
    c = config(folder, tmp_path)
    c['model'].update(dim=8, n_layers=2)
    c['training'].update(max_steps=1, accelerator='cpu', devices=1, plot_validation=False,
                         val_check_interval=1, enable_progress_bar=False, enable_model_summary=False)
    module, dm, trainer = run(copy.deepcopy(c))
    assert trainer.global_step == 1
    assert all(torch.isfinite(p).all() for p in module.parameters())
    from spinnme.data.cache import load_or_prepare_data_module, _LOADED_MODULES
    _LOADED_MODULES.clear()
    restored = load_or_prepare_data_module(c['data'], c['work_path'])
    assert restored.cache_reused
    np.testing.assert_allclose(restored.train_datasets[0][0]['coords'], dm.train_datasets[0][0]['coords'])
    expected_spectra = {}
    for dataset in dm.train_datasets:
        batch = dataset[0]
        response = module._get_spectral_response(batch, dataset.instrument_id)
        assert response['spectral_offsets'].shape == (12, dataset.num_wl, 9)
        with torch.no_grad():
            expected_spectra[dataset.ds_id] = module.validation_step(batch, 0)['stokes_pred'].numpy()
    for source in folder.glob('*.fits'):
        source.unlink()
    with load(tmp_path/'run/inversion.pme', device='cpu') as snapshot:
        for row in snapshot.observations:
            result = snapshot.evaluate(row['id'], parameters=['B', 'Blos', 'Vdop'])
            assert result.mask.all()
            assert np.asarray(result.metadata['slit_pixel_world']).shape == (4, 3)
            assert all(np.isfinite(a).all() for a in result.arrays.values())
            if row['id'] in expected_spectra:
                spectra = snapshot.evaluate(row['id'], parameters=['I', 'Q', 'U', 'V'])
                actual = np.stack([spectra.arrays[key] for key in 'IQUV'], axis=-2).reshape(12, 4, -1)
                actual = module.normalization(torch.from_numpy(actual)).detach().numpy()
                np.testing.assert_allclose(actual, expected_spectra[row['id']], rtol=2e-5, atol=2e-6)
                spectra.write_fits(tmp_path/'spectral-export'/row['id'])
                with fits.open(tmp_path/'spectral-export'/row['id']/'I.fits') as hdus:
                    assert 'X_COORDINATE' in hdus and 'Y_COORDINATE' in hdus
                    assert hdus[0].header['CTYPE3'] == 'WAVE-TAB'

        paths = result.write_fits(tmp_path/'export')
        with fits.open(paths[0]) as hdus:
            provenance = json.loads(hdus['PROVENANCE'].data['JSON'][0])
            assert np.asarray(provenance['slit_pixel_world']).shape == (4, 3)
            assert 'nominal WCS' in hdus[0].header['SPINNME GRID']
            coefficients = np.asarray(provenance['slit_pixel_world'])
            rows = np.arange(result.mask.shape[0])[:, None]
            np.testing.assert_allclose(hdus['X_COORDINATE'].data,
                                       coefficients[0]+rows*coefficients[2])
            np.testing.assert_allclose(hdus['Y_COORDINATE'].data,
                                       coefficients[1]+rows*coefficients[3])
            assert 'TIME_OFFSETS' in hdus

    from spinnme.cli import main
    assert main(['export', str(tmp_path/'run/inversion.pme'), '--observation', 'train_01_000',
                 '--instrument', 'HINODE_6302', '--parameters', 'B', 'B_los', 'B_transverse', 'azimuth',
                 '--device', 'cpu', '--output', str(tmp_path/'cli-export')]) == 0
    log = capsys.readouterr().out
    assert 'Observation: train_01_000; instrument: HINODE_6302; line: fe-i-6302' in log
    assert 'shape (y,x): (4, 3)' in log
    assert 'coordinates: native slit' in log
    assert 'Evaluation UTC:' in log and 'Valid pixels: 12/12' in log
    for file in (tmp_path/'cli-export').glob('*.fits'):
        with fits.open(file) as hdus:
            assert 'X_COORDINATE' in hdus and 'Y_COORDINATE' in hdus
            metadata = json.loads(hdus['PROVENANCE'].data['JSON'][0])
            assert metadata['snapshot_path'] == str((tmp_path/'run/inversion.pme').resolve())
            assert metadata['instrument_id'] == 'HINODE_6302'
    assert list(folder.iterdir()) == []


def test_each_slit_samples_its_own_physical_frame_without_resampling(tmp_path, monkeypatch):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from sunpy.coordinates import frames
    from sunpy.coordinates.ephemeris import get_earth
    from sunpy.map.mapbase import GenericMap

    folder = scan(tmp_path)
    sources = sorted(folder.glob('*.fits'))
    rolls, scales = [.4, 3.1, -1.2], [.32, .35, .31]
    xs, ys = [-200., -198.7, -195.3], [-110., -108., -113.5]
    dates = ['2024-03-23T02:00:35', '2024-03-23T02:30:35', '2024-03-23T03:00:35']
    for i, source in enumerate(sources):
        with fits.open(source, mode='update') as hdus:
            hdus[0].header.update(CROTA2=rolls[i], YSCALE=scales[i], XCEN=xs[i], YCEN=ys[i],
                                  DATE_OBS=dates[i], OBS_VR=1000.*(i+1))

    def no_resampling(*args, **kwargs):
        pytest.fail('Native Hinode observations must not be interpolated or resampled.')

    monkeypatch.setattr(GenericMap, 'resample', no_resampling)
    c = config(folder, tmp_path)
    dm = SphericalDataModule(work_directory=str(tmp_path/'cache'), **c['data'])
    ds = dm.train_datasets[0]
    batch = ds[0]
    # Independent SunPy reference, derived directly from each original slit header.
    y, x = np.indices((4, 3))
    along = (y-1.5)*np.asarray(scales)[None, :]
    tx = np.asarray(xs)[None, :]+np.sin(np.deg2rad(np.asarray(rolls)+.2636))[None, :]*along
    ty = np.asarray(ys)[None, :]+np.cos(np.deg2rad(np.asarray(rolls)+.2636))[None, :]*along
    times = Time(np.asarray(dates)[x.ravel()])
    observers = get_earth(times)
    world = SkyCoord(tx.ravel()*u.arcsec, ty.ravel()*u.arcsec,
                     frame=frames.Helioprojective(observer=observers, obstime=times))
    physical = world.transform_to(frames.HeliographicCarrington(observer=observers, obstime=times))
    expected = physical.cartesian.xyz.to_value(u.Rsun).T
    np.testing.assert_allclose(batch['coords'][:, :3], expected, atol=1e-7)
    np.testing.assert_allclose(batch['coords'][:, 3].reshape(4, 3)[0], [0, .5, 1.])
    # Each original profile is retained unchanged, including gaps in slit indices.
    np.testing.assert_array_equal(batch['stokes'][:, 0, 0].reshape(4, 3),
                                  np.tile([101, 102, 103], (4, 1)))
    np.testing.assert_allclose(ds.slit_geometry['pAng'], np.deg2rad(np.asarray(rolls)+.2636))
    assert np.ptp(ds.slit_geometry['obs_lon']) > .001
    assert np.ptp(batch['v_obs_los'].numpy()) > 1000
    # Raster indices and the nominal display WCS cannot replace the physical coordinates.
    row = copy.deepcopy(ds.observation)
    row['header'].update(CRVAL1=400., CRVAL2=300., CDELT1=5., CDELT2=5.)
    native = GridGeometry(row).pixels(np.arange(12), Rs_per_ds=1, normalized_time=0, seconds_per_dt=3600)
    np.testing.assert_allclose(native['coords'], batch['coords'], atol=1e-7)
    for key in ('mu', 'rtp_to_img_transform', 'v_obs_los'):
        np.testing.assert_allclose(native[key], batch[key], rtol=1e-6, atol=1e-7)


def test_old_native_cache_requires_rebuilding_physical_frames(tmp_path):
    from spinnme.data.cache import _restore_dataset
    old = {'class': 'HinodeSphericalDataset', 'attributes': {'slit_pixel_world': [[0], [0], [0], [1]]},
           'arrays': {}}
    with pytest.raises(ValueError, match='--reload'):
        _restore_dataset(old, tmp_path)
