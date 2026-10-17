import json
from datetime import datetime

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunpy.coordinates import frames
from sunpy.map import make_fitswcs_header

from spinnme.data.fits import (read_fits_observation, observation_times, parse_time,
                              wavelength_configuration)
from spinnme.data.phi import read_phi_observation
from spinnme.data.prepare_hinode import prepare_hinode_raster
from spinnme.loader.spherical import (FITSSphericalDataset, HinodeSphericalDataset,
    SphericalDataModule, HMISphericalDataset, spherical_geometry_from_pixels, load_map_data)


def header(shape=(3, 5)):
    observer = SkyCoord(lon=0*u.deg, lat=0*u.deg, radius=1*u.AU,
                        obstime='2024-03-23T00:00:00', frame=frames.HeliographicStonyhurst)
    reference = SkyCoord(0*u.arcsec, 0*u.arcsec, observer=observer,
                         obstime=observer.obstime, frame=frames.Helioprojective)
    h = fits.Header(make_fitswcs_header(shape, reference, scale=[2,2]*u.arcsec/u.pix))
    for key in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
        h[key] = 0.
    h['DATE-OBS'] = '2024-03-23T00:00:00'
    h['DATE-AVG'] = '2024-03-23T00:00:10'
    h['DATE_EAR'] = '2024-03-23T00:05:00'
    h['DATE_SUN'] = '2024-03-22T23:51:40'
    h['WAVELNTH'] = 6173.341
    h['WAVEUNIT'] = 'Angstrom'
    return h


def cube(tmp_path, count=7):
    y,x,s,w = np.indices((3,5,4,count))
    values = 1 + (y*1000+x*100+s*10+w)/10000
    h = header()
    for i, offset in enumerate(np.linspace(-.2,.3,count)):
        h[f'WAVELN{i+1:02d}'] = h['WAVELNTH']+offset
    path = tmp_path/'cube.fits'
    fits.writeto(path,values.transpose(2,3,0,1).astype('float32'),h)
    return path, values.astype('float32')


def test_explicit_axes_units_and_unchanged_time(tmp_path):
    path, expected = cube(tmp_path)
    observation = read_fits_observation(path, axis_order=['stokes','wavelength','y','x'], time_reference='earth')
    np.testing.assert_array_equal(observation['stokes'],expected)
    assert observation['header']['DATE-OBS'] == '2024-03-23T00:00:00'
    assert observation['model_time'] == datetime(2024,3,23,0,5)
    assert observation['wavelength_config']['wavelength_grid'].shape == (7,)
    assert observation['header']['NAXIS1'] == 5
    assert observation['header']['NAXIS2'] == 3
    with pytest.raises(ValueError,match='axis_order'):
        read_fits_observation(path)


def test_science_extension_inherits_global_observer_metadata(tmp_path):
    path,expected=cube(tmp_path)
    with fits.open(path) as hs:
        global_header=hs[0].header.copy()
        image=hs[0].data.copy()
    science=fits.ImageHDU(image,name='SCIENCE')
    for key in ('CTYPE1','CTYPE2','CUNIT1','CUNIT2','CDELT1','CDELT2',
                'CRPIX1','CRPIX2','CRVAL1','CRVAL2'):
        science.header[key]=global_header[key]
    for key in ['NAXIS1','NAXIS2','NAXIS3','NAXIS4']:
        global_header.remove(key,ignore_missing=True)
    new=tmp_path/'extensions.fits'
    fits.HDUList([fits.PrimaryHDU(header=global_header),science]).writeto(new)
    observation=read_fits_observation(new,data_hdu='SCIENCE',axis_order=['stokes','wavelength','y','x'])
    np.testing.assert_array_equal(observation['stokes'],expected)
    assert observation['header']['DSUN_OBS'] > 1e11


def test_auxiliary_wcs_distortions_are_rejected_explicitly(tmp_path):
    path,_=cube(tmp_path)
    with fits.open(path,mode='update') as hs:
        hs[0].header['CPDIS1']='LOOKUP'
    with pytest.raises(ValueError,match='auxiliary distortion'):
        read_fits_observation(path,axis_order=['stokes','wavelength','y','x'])


def test_time_scales_and_references():
    assert parse_time('2024.03.23_01:00:00_TAI') == datetime(2024,3,23,0,59,23)
    times, average = observation_times(header(), 'average')
    assert average == datetime(2024,3,23,0,0,10)
    assert times['original']['DATE_EAR'].endswith('00:05:00')


@pytest.mark.parametrize('options', [
    {'wavelength_center': 6173.3433},
    {'wavelength_grid': [6173.2, 6173.5], 'wavelength_grid_is_offsets': False},
    {'wavelength_grid': [-.1433, .1567]},
    {'wavelength_center': 617.33433*u.nm},
    {'wavelength_grid': np.array([-.01433,.01567])*u.nm},
    {'wavelength_center': 6173.3433*u.AA,
     'wavelength_grid': np.array([617.32,617.35])*u.nm,
     'wavelength_grid_is_offsets': False},
])
def test_header_and_configured_wavelength_units_are_independent(options):
    h = fits.Header({'WAVELNTH':617.33433, 'WAVEUNIT':'nm',
                     'WAVELN01':617.32, 'WAVELN02':617.35})
    configuration = wavelength_configuration(h, 2, wavelength_unit='Angstrom', **options)
    assert configuration['wavelength_center'].to_value(u.AA) == pytest.approx(6173.3433)
    np.testing.assert_allclose(configuration['wavelength_grid'].to_value(u.AA),
                               [-.1433,.1567], rtol=0, atol=2e-12)


def test_generic_spherical_dataset_registry_and_weight(tmp_path):
    path, _ = cube(tmp_path)
    dataset = FITSSphericalDataset(path, ds_id='custom', instrument_id='A', line_id='fe-i-6173',
                 axis_order=['stokes','wavelength','y','x'], work_directory=str(tmp_path/'cache'),
                 stokes_normalization=1, seconds_per_dt=60, Rs_per_ds=1,
                 ref_time=datetime(2024,3,23), batch_size=4, shuffle=False, loss_weight=2.)
    item = dataset[0]
    assert item['coords'].shape == (4,4)
    assert item['stokes'].shape == (4,4,7)
    assert np.isfinite(item['coords'].numpy()).all()
    assert np.all(item['sample_weight'].numpy() == 2)
    assert dataset.observation['shape'] == [3,5]
    assert dataset.observation['geometry']['observer_distance'] > 200
    json.dumps(dataset.observation, allow_nan=False)


@pytest.mark.parametrize('options,message', [
    ({'mu_limit': np.nan}, 'mu_limit'),
    ({'mu_limit': 1.}, 'mu_limit'),
    ({'oversample_factor': 0}, 'oversample_factor'),
    ({'oversample_factor': 1.5}, 'oversample_factor'),
    ({'oversample_factor': True}, 'oversample_factor'),
])
def test_spherical_dataset_sampling_options_are_validated(tmp_path, options, message):
    path, _ = cube(tmp_path)
    with pytest.raises(ValueError, match=message):
        FITSSphericalDataset(path, ds_id='custom', instrument_id='A',
                 axis_order=['stokes','wavelength','y','x'], work_directory=str(tmp_path/'cache'),
                 stokes_normalization=1, seconds_per_dt=60, Rs_per_ds=1,
                 ref_time=datetime(2024,3,23), batch_size=4, **options)


def test_datamodule_selection_and_registry(tmp_path):
    path,_ = cube(tmp_path)
    config = dict(type='fits', data_path=str(path), instrument_id='A',
                  axis_order=['stokes','wavelength','y','x'])
    module = SphericalDataModule([config],config,str(tmp_path/'cache'),
                 ref_time='2024-03-23T00:00:00',batch_size=8,dataset_batch_size=4,
                 num_workers=0,num_dataset_workers=0)
    assert len(module.get_observations()) == 2
    assert module.get_observations()[1]['role'] == 'validation'
    json.dumps(module.get_observations(),allow_nan=False)


def test_automatic_time_origin_uses_selected_model_epoch(tmp_path):
    path,_=cube(tmp_path)
    config=dict(type='fits',data_path=str(path),instrument_id='A',
                axis_order=['stokes','wavelength','y','x'],time_reference='earth')
    module=SphericalDataModule([config],config,str(tmp_path/'cache'),
                batch_size=8,dataset_batch_size=4,num_workers=0,num_dataset_workers=0)
    assert module.ref_time == datetime(2024,3,23,0,5)
    assert module.train_datasets[0].normalized_time == 0.
    assert module.train_datasets[0].observation['header']['DATE-OBS'] == '2024-03-23T00:00:00'


def test_phi_calibrated_schema_and_continuum(tmp_path):
    path, canonical = cube(tmp_path,6)
    with fits.open(path,mode='update') as hs:
        hs[0].data = canonical.transpose(3,2,0,1)
        waves = 6173.341+np.array([-.14,-.07,0,.07,.14,.3])
        for i,wave in enumerate(waves):
            hs[0].header[f'WAVELN{i+1:02d}']=wave
        hs[0].header['CONTPOS']=(6,'Index to WAVELNnn')
    observation = read_phi_observation(path)
    np.testing.assert_array_equal(observation['stokes'],canonical)
    assert observation['continuum_index'] == 5
    with fits.open(path,mode='update') as hs:
        del hs[0].header['WAVELN01']
    with pytest.raises(ValueError, match='Supply wavelength_grid'):
        read_phi_observation(path)
    observation = read_phi_observation(path, wavelength_grid=waves,
                                       wavelength_center=6173.341,
                                       wavelength_grid_is_offsets=False)
    assert observation['continuum_index'] == 5
    with fits.open(path,mode='update') as hs:
        hs[0].header['CTYPE1']='WAVE'
    with pytest.raises(ValueError,match='Unrecognized PHI'):
        read_phi_observation(path)


def test_phi_continuum_fallback_requires_an_isolated_endpoint():
    from spinnme.data.phi_util import phi_continuum_index
    assert phi_continuum_index({}, [-.14,-.07,0,.07,.14,.4]) == 5
    assert phi_continuum_index({}, [-.4,-.14,-.07,0,.07,.14]) == 0
    with pytest.raises(ValueError, match='Ambiguous PHI continuum'):
        phi_continuum_index({}, [-1.,-.9,-.8,.8,.9,1.])
    for invalid in (0, 999, 2.5, True, 'unknown'):
        with pytest.raises(ValueError, match='CONTPOS must'):
            phi_continuum_index({'CONTPOS': invalid}, [-.14,-.07,0,.07,.14,.4])


def test_hinode_slit_wavelength_sign_and_scan_times(tmp_path):
    files=[]
    for i in [1,0]:
        h=fits.Header({'SLITINDX':i,'CRVAL1':6302.5,'CRPIX1':2.,'CDELT1':-.02,
                       'CUNIT1':'Angstrom','DATE_OBS':f'2024-03-23T00:00:{i*3:02d}'})
        p=tmp_path/f'slit{i}.fits'
        fits.writeto(p,np.ones((4,3,4),dtype='float32')*(i+1),h)
        files.append(p)
    raster_header = header((3, 2))
    for key in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
        del raster_header[key]
    path=prepare_hinode_raster(files,tmp_path/'raster.fits',raster_header,wavelength_center=6302.4931)
    with fits.open(path) as hs:
        assert hs[0].data.shape == (4,4,3,2)
        assert hs[0].header['WAVELN01'] > hs[0].header['WAVELN04']
        np.testing.assert_array_equal(hs['TIME_OFFSETS'].data[0],[0,3])
    dataset=HinodeSphericalDataset(path,ds_id='hinode',instrument_id='H',
                 work_directory=str(tmp_path/'cache'),time_hdu='TIME_OFFSETS',
                 stokes_normalization=1,seconds_per_dt=3,Rs_per_ds=1,
                 ref_time=datetime(2024,3,23),batch_size=6,shuffle=False)
    np.testing.assert_array_equal(dataset[0]['coords'][:,-1],[0,1,0,1,0,1])
    assert dataset.observation['time_offsets_seconds'] == [0,3]
    assert dataset.observation['time_offset_axis'] == 'x'
    np.testing.assert_array_equal(dataset.observer_velocity, [0., 0., 0.])
    np.testing.assert_array_equal(dataset[0]['v_obs_los'], 0.)


def test_hinode_preparation_canonicalizes_polarization_and_time_scale(tmp_path):
    h = fits.Header({'CRVAL1':6302.5, 'CRPIX1':1., 'CDELT1':.02,
                     'CUNIT1':'Angstrom', 'DATE_OBS':'2024-03-23T00:00:40',
                     'TIMESYS':'TAI'})
    slit = tmp_path/'slit.fits'
    raw = np.broadcast_to(np.array([2.,3.,4.,1.])[:,None,None], (4,3,4))
    fits.writeto(slit, raw, h)
    raster_header = header((3,1))
    raster_header['DATE-OBS'] = '2024-03-23T00:00:37'
    raster_header['TIMESYS'] = 'TAI'
    path = prepare_hinode_raster([slit],tmp_path/'raster.fits',raster_header,
                                stokes_order='QUVI', wavelength_center=6302.4931)
    observation = read_fits_observation(path, axis_order=['stokes','wavelength','y','x'])
    np.testing.assert_array_equal(observation['stokes'][0,0,:,0], [1.,2.,3.,4.])
    assert observation['model_time'] == datetime(2024,3,23)
    with fits.open(path) as hs:
        assert hs[0].header['SPNSTOK'] == 'IQUV'
        assert hs[0].header['TIMESYS'] == 'UTC'
        np.testing.assert_array_equal(hs['TIME_OFFSETS'].data, 3.)
    with pytest.raises(ValueError, match='stokes_order'):
        prepare_hinode_raster([slit], tmp_path/'invalid.fits', raster_header,
                              stokes_order='IIUV')


def test_non_tan_solar_wcs_uses_generic_geometry(tmp_path):
    from sunpy.map import Map
    h=header()
    h['CTYPE1']='HPLN-SIN'
    h['CTYPE2']='HPLT-SIN'
    map_=Map(np.ones((3,5)),h)
    data=load_map_data(map_)
    geometry=spherical_geometry_from_pixels(np.arange(15),(3,5),map_.wcs,
        data['obs_lon'].to_value(u.rad),data['obs_lat'].to_value(u.rad),
        data['pAng'].to_value(u.rad),data['observer_distance'],data['observer_velocity'],1,0)
    assert np.isfinite(geometry['coords']).all()
    np.testing.assert_allclose(geometry['mu'][:,0],data['mu'].reshape(-1),atol=1e-7)


def test_tiny_circular_generator_has_generic_wavelength_count(tmp_path):
    from spinnme.data.synthetic import generate_circular
    paths=generate_circular(tmp_path,shape=(8,10),n_time_steps=1,
                            pattern_resolution=(16,16),wavelength_grid=[-.1,0,.1])
    observation=read_fits_observation(paths[0],axis_order=['stokes','wavelength','y','x'])
    assert observation['stokes'].shape == (8,10,4,3)
    assert np.isfinite(observation['stokes'][4,5]).all()
    assert (tmp_path/'parameters_000.npz').is_file()
    with pytest.raises(FileExistsError, match='new directory'):
        generate_circular(tmp_path, shape=(8,10), n_time_steps=2,
                          pattern_resolution=(16,16))
    assert not (tmp_path/'circular_001.fits').exists()


def test_download_manifest_reuses_verified_offline_collection(tmp_path, monkeypatch):
    from spinnme.data import download
    calls=[]
    def dispatch(*args,**kwargs):
        calls.append(1)
        path,_=cube(tmp_path)
        return [path]
    monkeypatch.setattr(download,'_dispatch',dispatch)
    manifest=download.download('phi',output=tmp_path,start='2024-03-23',end='2024-03-24')
    content=json.loads(manifest.read_text())
    assert content['files'][0]['metadata']['WAVELNTH'] == 6173.341
    assert len(content['files'][0]['sha256']) == 64
    assert download.download('phi',output=tmp_path,start='2024-03-23',end='2024-03-24') == manifest
    assert len(calls) == 1
    with pytest.raises(ValueError,match='different download collection'):
        download.download('phi',output=tmp_path,start='2024-03-24',end='2024-03-25')


@pytest.mark.parametrize('relative_to_cwd', [False, True])
def test_download_archive_relative_paths(tmp_path, monkeypatch, relative_to_cwd):
    from spinnme.data import download
    monkeypatch.chdir(tmp_path)
    output = tmp_path/'observations'
    def dispatch(*args, **kwargs):
        path, _ = cube(output)
        return [str(path.relative_to(tmp_path) if relative_to_cwd else path.name)]
    monkeypatch.setattr(download, '_dispatch', dispatch)
    manifest = download.download('phi', output='observations')
    records = json.loads(manifest.read_text())['files']
    assert records[0]['path'] == 'cube.fits'


def test_download_urls_reject_filename_collisions_before_transfer(tmp_path, monkeypatch):
    from spinnme.data import download
    def unexpected_transfer(*args, **kwargs):
        pytest.fail('URL collisions must be detected before downloading.')
    monkeypatch.setattr('urllib.request.urlopen', unexpected_transfer)
    with pytest.raises(ValueError, match='distinct filenames'):
        download.download('hinode', output=tmp_path,
                          urls=['https://example.org/a/slit.fits',
                                'https://example.org/b/slit.fits'])
    assert not list(tmp_path.iterdir())


def test_hmi_segment_wcs_alignment_is_checked(tmp_path):
    h=header()
    paths=[]
    for component in 'IQUV':
        group=[]
        for wave in range(6):
            p=tmp_path/f'hmi.{component}{wave}.fits'
            fits.writeto(p,np.ones((3,5)),h)
            group.append(str(p))
        paths.append(group)
    HMISphericalDataset._validate_segment_alignment(paths)
    with fits.open(paths[2][3],mode='update') as hs:
        hs[0].header['CRPIX1']+=1
    with pytest.raises(ValueError,match='spatial WCS'):
        HMISphericalDataset._validate_segment_alignment(paths)


def test_holdout_exclusion_preserves_acquisition_order():
    module=SphericalDataModule.__new__(SphericalDataModule)
    assert module._select_files(list(range(12)), {'exclude_samples': [4]}, 'fits') == [0,1,2,3,5,6,7,8,9,10,11]


@pytest.mark.parametrize('descending', [False, True])
def test_hinode_preset_trains_both_lines_from_one_raster(tmp_path, descending):
    import torch
    from spinnme.configuration import resolve_config
    from spinnme.train.me_spherical_module import MESphericalModule

    # Unequal sample counts exercise independent line grids, including a
    # descending detector dispersion. Slit timing must survive both selections.
    waves = np.linspace(6300.9, 6303.1, 113)
    if descending:
        waves = waves[::-1]
    slit_header = fits.Header({'CRVAL1': waves[0], 'CRPIX1': 1.,
                              'CDELT1': waves[1]-waves[0], 'CUNIT1': 'Angstrom',
                              'DATE_OBS': '2024-03-23T00:00:00'})
    slit = tmp_path/'slit.fits'
    raw = np.zeros((4, 3, len(waves)), dtype=np.float32)
    raw[0] = 1.
    fits.writeto(slit, raw, slit_header)
    raster = prepare_hinode_raster([slit], tmp_path/'raster.fits', header((3, 1)))
    config = resolve_config('hinode/sp', input_path=str(raster), output=str(tmp_path/'run'))
    data = dict(config['data'], batch_size=3, dataset_batch_size=3,
                num_dataset_workers=0, ref_time='2024-03-23T00:00:00')
    module = SphericalDataModule(work_directory=str(tmp_path/'cache'), **data)
    assert set(module.wavelength_config) == {'HINODE_6301', 'HINODE_6302'}
    spectra = [dataset[0]['stokes'] for dataset in module.train_datasets]
    assert sum(s.shape[-1] for s in spectra) == len(waves)
    observations = module.get_observations()
    assert {o['line_id'] for o in observations} == {'fe-i-6301', 'fe-i-6302'}
    assert all(o['reader']['wavelength_range'] is not None for o in observations)
    json.dumps(observations, allow_nan=False)
    training = MESphericalModule(
        image_shape=(3, 1), gauss_per_dB=1000, Rs_per_ds=1, seconds_per_dt=3600,
        wavelength_config=module.wavelength_config, instrument_config=config['instrument'],
        model_config={'type': 'mlp', 'dim': 8, 'n_layers': 2,
                      'encoding_config': {'type': 'identity'}},
        stokes_loss_config=config['stokes_loss'])
    batch = {dataset.instrument_id: dataset[0] for dataset in module.train_datasets}
    loss = training.training_step(batch, 0)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    for head in training.parameter_model.line_heads.values():
        assert head.weight.grad is not None
        assert torch.isfinite(head.weight.grad).all()
    import copy
    restored = MESphericalModule(**copy.deepcopy(training.snapshot_init))
    restored.load_state_dict(training.state_dict())
    assert restored.instrument_lines == {'HINODE_6301': 'fe-i-6301', 'HINODE_6302': 'fe-i-6302'}
    coords = next(iter(batch.values()))['coords']
    first = training.parameter_model(coords, instrument_id='HINODE_6301')
    second = training.parameter_model(coords, instrument_id='HINODE_6302')
    for key in ('b_x', 'b_y', 'b_z', 'v_x', 'v_y', 'v_z'):
        torch.testing.assert_close(first[key], second[key])


def test_spectral_window_units_continuum_and_empty_selection(tmp_path):
    path, expected = cube(tmp_path)
    options = dict(axis_order=['stokes', 'wavelength', 'y', 'x'],
                   wavelength_unit='nm', wavelength_range=[617.33, None])
    observation = read_fits_observation(path, continuum_index=6, **options)
    absolute = (observation['wavelength_config']['wavelength_grid'] +
                observation['wavelength_config']['wavelength_center']).to_value(u.nm)
    assert np.all(absolute >= 617.33)
    np.testing.assert_array_equal(observation['stokes'], expected[..., -len(absolute):])
    assert observation['continuum_index'] == len(absolute)-1
    with pytest.raises(ValueError, match='outside wavelength_range'):
        read_fits_observation(path, continuum_index=0, **options)
    with pytest.raises(ValueError, match='no spectral samples'):
        read_fits_observation(path, **dict(options, wavelength_range=[630., 631.]))


def test_empty_archive_download_never_writes_success_manifest(tmp_path, monkeypatch):
    from spinnme.data import download
    monkeypatch.setattr(download, '_dispatch', lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match='archive returned no files'):
        download.download('phi', output=tmp_path)
    assert not (tmp_path/'download-manifest.json').exists()


def test_phi_continuum_index_is_remapped_after_wavelength_selection(tmp_path):
    path, canonical = cube(tmp_path, 6)
    waves = 6173.341+np.array([-.3, -.14, -.07, 0, .07, .14])
    with fits.open(path, mode='update') as hdus:
        hdus[0].data = canonical.transpose(3, 2, 0, 1)
        for index, wave in enumerate(waves, start=1):
            hdus[0].header[f'WAVELN{index:02d}'] = wave
        hdus[0].header['CONTPOS'] = 1
    result = read_phi_observation(path, wavelength_range=[None, 6173.4])
    assert result['continuum_index'] == 0
    assert result['stokes'].shape[-1] < 6
    with pytest.raises(ValueError, match='outside wavelength_range'):
        read_phi_observation(path, wavelength_range=[6173.2, None])


def test_lambda_axis_names_wavelengths_without_reusing_weights(tmp_path):
    path, expected = cube(tmp_path)
    result = read_fits_observation(path, axis_order=['stokes', 'lambda', 'y', 'x'])
    np.testing.assert_array_equal(result['stokes'], expected)
    with pytest.raises(ValueError, match='axis_order'):
        read_fits_observation(path, axis_order=['stokes', 'weights', 'y', 'x'])


def test_separate_spatial_hdu_preserves_science_metadata_and_model_clock(tmp_path):
    path, expected = cube(tmp_path)
    with fits.open(path) as hdus:
        science = fits.ImageHDU(hdus[0].data.copy(), hdus[0].header.copy(), name='SCI')
    science.header['TIMESYS'] = 'TAI'
    spatial = fits.ImageHDU(np.zeros((3, 5)), name='SPATIAL')
    for key in ('CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2', 'CDELT1', 'CDELT2',
                'CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2'):
        spatial.header[key] = science.header[key]
    spatial.header['CRVAL1'] = .01
    spatial.header['DATE-OBS'] = '2000-01-01T00:00:00'
    spatial.header['WAVELNTH'] = 6302.5
    output = tmp_path/'extensions.fits'
    fits.HDUList([fits.PrimaryHDU(), science, spatial]).writeto(output)
    options = dict(data_hdu='SCI', spatial_header_hdu='SPATIAL',
                   axis_order=['stokes', 'wavelength', 'y', 'x'])
    observation = read_fits_observation(output, **options)
    np.testing.assert_array_equal(observation['stokes'], expected)
    assert (observation['header']['CRVAL1']*u.Unit(observation['header']['CUNIT1'])).to_value(u.arcsec) == pytest.approx(.01)
    assert observation['header']['WAVELNTH'] == 6173.341
    assert observation['header']['DATE-OBS'] == '2024-03-23T00:00:00'
    assert observation['model_time'] == datetime(2024, 3, 22, 23, 59, 23)
    config = {**options, 'time_reference': 'average'}
    selected_time = SphericalDataModule._observation_time_for_file_group(output, 'fits', config)
    assert selected_time == datetime(2024, 3, 22, 23, 59, 33)
    loader = SphericalDataModule.__new__(SphericalDataModule)
    select = {**config, 'time_range': [selected_time.isoformat(), selected_time.isoformat()]}
    assert loader._select_files([output], select, 'fits') == [output]
    assert loader._select_files([output], {**config, 'exclude_times': [selected_time.isoformat()]}, 'fits') == []
    source = dict(type='fits', data_path=str(output), **config)
    assert loader._default_reference_time([source]) == selected_time
    # Explicit scale overrides must follow the same path as ingestion.
    config['time_scale'] = 'utc'
    assert loader._observation_time_for_file_group(output, 'fits', config) == datetime(2024, 3, 23, 0, 0, 10)
