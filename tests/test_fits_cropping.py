"""Generic FITS crops preserve cube axes, HDUs, WCS and detector coordinates."""
import json
from datetime import datetime

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from test_fits_observations import header

from spinnme.cli import main
from spinnme.configuration import resolve_config
from spinnme.data.crop import crop_fits
from spinnme.data.hmi_time import load_hmi_date_obs_map
from spinnme.data.hmi_transmission import MANIFEST_FORMAT, hmi_acquisition_key
from spinnme.loader.spherical import HMISphericalDataset, SphericalDataModule, load_hmi_detector_coords


def acquisition(tmp_path, kind='primary', roll=0.):
    source = tmp_path/'source'
    source.mkdir()
    h = header((10, 12))
    for key in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
        del h[key]
    h.update(T_REC='2024.03.23_00:00:37_TAI', HCAMID=2, QUALITY=0,
             TELESCOP='SDO/HMI', INSTRUME='HMI_SIDE1', CCD_X0=100., CCD_Y0=200.)
    angle = np.deg2rad(roll)
    h.update(PC1_1=np.cos(angle), PC1_2=-np.sin(angle),
             PC2_1=np.sin(angle), PC2_2=np.cos(angle))
    for channel, component in enumerate('IQUV'):
        for wavelength in range(6):
            data = np.arange(120, dtype=np.float32).reshape(10, 12)+10*(6*channel+wavelength)
            path = source/f'hmi.S_720s.20240323_000037_TAI.1.{component}{wavelength}.fits'
            if kind == 'compressed':
                fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data, h)]).writeto(path)
            else:
                image = fits.PrimaryHDU(data, h)
                if kind == 'scaled':
                    image.scale('int16', bscale=.25, bzero=100)
                image.writeto(path)
    return source


@pytest.mark.parametrize('kind', ['primary', 'compressed', 'scaled'])
def test_cli_pixel_crop_preserves_stokes_wcs_and_detector_origin(tmp_path, kind):
    source = acquisition(tmp_path, kind)
    output = tmp_path/'crop'
    assert main(['crop', '--input', str(source), '--output', str(output),
                 '--pixels', '2', '3', '9', '8']) == 0
    assert len(list(output.glob('*.fits'))) == 24
    for original in source.glob('*.fits'):
        reference = fits.getdata(original)
        cropped = output/original.name
        np.testing.assert_array_equal(fits.getdata(cropped), reference[3:8, 2:9])
        meta = fits.getheader(cropped, 1 if kind == 'compressed' else 0)
        assert meta['CCD_X0'] == 102.
        assert meta['CCD_Y0'] == 203.
        assert meta['T_REC'] == '2024.03.23_00:00:37_TAI'
        assert meta['HCAMID'] == 2
        assert meta['QUALITY'] == 0
        assert 'OBS_VR' not in meta
        # Compressed inputs store image WCS in extension 1.
        original_header = fits.getheader(original, 1 if kind == 'compressed' else 0)
        x, y = np.array([0., 3., 6.]), np.array([0., 2., 4.])
        expected = WCS(original_header).celestial.pixel_to_world_values(x+2, y+3)
        actual = WCS(meta).celestial.pixel_to_world_values(x, y)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    grouped = SphericalDataModule.__new__(SphericalDataModule)._load_IQUV_files(str(output))
    assert grouped.shape == (1, 4, 6)
    image = load_hmi_date_obs_map(next(output.glob('*.I0.fits')))
    detector = load_hmi_detector_coords(image)
    np.testing.assert_allclose(detector[0, 0], np.array([102, 203])/4095, atol=1e-7)


def test_nested_crop_accumulates_original_detector_origin(tmp_path):
    source = acquisition(tmp_path)
    first = crop_fits(source, tmp_path/'first', pixels=[2, 3, 9, 8])
    output = tmp_path/'second'
    crop_fits(first, output, pixels=[1, 1, 5, 4])
    path = next(output.glob('*.I0.fits'))
    h = fits.getheader(path)
    assert (h['CCD_X0'], h['CCD_Y0']) == (103., 204.)
    original = next(source.glob('*.I0.fits'))
    np.testing.assert_array_equal(fits.getdata(path), fits.getdata(original)[4:7, 3:7])


def test_arcsec_crop_handles_rotated_wcs(tmp_path):
    source = acquisition(tmp_path, roll=31.)
    output = tmp_path/'sky'
    assert main(['crop', '--input', str(source), '--output', str(output),
                 '--bounds', '-3', '-3', '3', '3']) == 0
    path = next(output.glob('*.I0.fits'))
    h = fits.getheader(path)
    shape = fits.getdata(path).shape
    assert 0 < shape[0] < 10 and 0 < shape[1] < 12
    x, y = WCS(h).celestial.world_to_pixel_values(np.array([-3, 3, 3, -3])/3600,
                                                np.array([-3, -3, 3, 3])/3600)
    assert x.min() >= -1e-6 and x.max() <= shape[1]-1+1e-6
    assert y.min() >= -1e-6 and y.max() <= shape[0]-1+1e-6


@pytest.mark.parametrize('problem', ['outside', 'occupied'])
def test_invalid_collections_never_write_partial_crops(tmp_path, problem):
    source = acquisition(tmp_path)
    output = tmp_path/'crop'
    pixels, bounds = [2, 3, 9, 8], None
    if problem == 'outside':
        pixels, bounds = None, [500, 500, 600, 600]
    else:
        output.mkdir()
        (output/'keep.txt').write_text('preserve')
    with pytest.raises((ValueError, FileExistsError)):
        crop_fits(source, output, pixels=pixels, bounds=bounds)
    assert not list(output.glob('*.fits'))
    assert len(list(source.glob('*.fits'))) == 24


def test_subframe_can_train_with_acquisition_transmission_profiles(tmp_path):
    source = acquisition(tmp_path)
    output = tmp_path/'crop'
    crop_fits(source, output, pixels=[2, 3, 9, 8])
    config = resolve_config('hmi/subframe', output=str(tmp_path/'run'), bindings={'data_path': str(tmp_path)})
    assert all('resolution' not in d for d in [*config['data']['train_configs'], config['data']['valid_config']])
    calibration = tmp_path/'transmission'
    calibration.mkdir()
    np.savez(calibration/'profile.npz', offsets=np.zeros((6, 1)),
             weights=np.ones((1, 1, 6, 1)), continuum_weights=np.zeros((1, 1, 6)))
    (calibration/'manifest.json').write_text(json.dumps({
        'format': MANIFEST_FORMAT, 'profiles': {'profile': {'file': 'profile.npz'}},
        'acquisitions': {hmi_acquisition_key('2024.03.23_00:00:37_TAI', 2): {'profile': 'profile'}},
    }))
    group = SphericalDataModule.__new__(SphericalDataModule)._load_IQUV_files(str(output))[0]
    dataset = HMISphericalDataset(group, transmission_profile_directory=calibration,
                                  ds_id='crop', instrument_id='HMI', seconds_per_dt=3600, Rs_per_ds=1,
                                  ref_time=datetime(2024, 3, 23), stokes_normalization=1,
                                  work_directory=str(tmp_path/'cache'), batch_size=8, shuffle=False)
    assert dataset.image_shape == (5, 7)
    assert dataset[0]['spectral_weights'].shape == (8, 6, 1)
    np.testing.assert_allclose((dataset[0]['pix'][0].numpy()[::-1]+1)/2, np.array([102, 203])/4095, atol=1e-7)


def test_unlabelled_single_fits_needs_no_instrument_or_wcs(tmp_path):
    source = tmp_path/'image.fits'
    values = np.arange(80).reshape(8, 10)
    fits.writeto(source, values)
    output = tmp_path/'cropped'
    assert main(['crop', '--input', str(source), '--output', str(output), '--pixels', '1', '2', '7', '6']) == 0
    np.testing.assert_array_equal(fits.getdata(output/source.name), values[2:6, 1:7])
    assert 'CCD_X0' not in fits.getheader(output/source.name)


def test_cube_preserves_spectral_grid_table_mask_and_scan_times(tmp_path):
    source = tmp_path/'raster.fits'
    values = np.arange(4*6*8*10).reshape(4, 6, 8, 10).astype(np.float32)
    h = header((8, 10))
    for index in range(6):
        h[f'WAVELN{index+1:02d}'] = 6301.+index*.02
    times = np.broadcast_to(np.arange(10)*3, (8, 10)).copy()
    mask = np.ones((8, 10), dtype=np.uint8)
    wave = np.arange(6)*.02
    table = fits.BinTableHDU.from_columns([fits.Column(name='VALUE', format='D', array=[1., 2.])], name='CALIBRATION')
    fits.HDUList([fits.PrimaryHDU(values, h), fits.ImageHDU(times, name='TIME_OFFSETS'),
                  fits.ImageHDU(mask, name='VALID_MASK'), fits.ImageHDU(wave, name='WAVELENGTH'), table]).writeto(source)
    output = tmp_path/'cropped'
    crop_fits(source, output, pixels=[2, 1, 8, 7])
    with fits.open(output/source.name) as hdus:
        assert [hdu.name for hdu in hdus] == ['PRIMARY', 'TIME_OFFSETS', 'VALID_MASK', 'WAVELENGTH', 'CALIBRATION']
        np.testing.assert_array_equal(hdus[0].data, values[:, :, 1:7, 2:8])
        np.testing.assert_array_equal(hdus['TIME_OFFSETS'].data, times[1:7, 2:8])
        np.testing.assert_array_equal(hdus['VALID_MASK'].data, mask[1:7, 2:8])
        np.testing.assert_array_equal(hdus['WAVELENGTH'].data, wave)
        np.testing.assert_array_equal(hdus['CALIBRATION'].data['VALUE'], [1., 2.])
        for index in range(6):
            assert hdus[0].header[f'WAVELN{index+1:02d}'] == h[f'WAVELN{index+1:02d}']


def test_nondefault_spatial_axes_preserve_full_cube_wcs(tmp_path):
    # NumPy y,x,wavelength,Stokes -> spatial FITS axes 3,4.
    values = np.arange(8*10*6*4).reshape(8, 10, 6, 4).astype(np.float32)
    h = fits.Header({'CTYPE1': 'STOKES', 'CRPIX1': 1., 'CRVAL1': 1., 'CDELT1': 1.,
                     'CTYPE2': 'WAVE', 'CUNIT2': 'Angstrom', 'CRPIX2': 1., 'CRVAL2': 6301., 'CDELT2': .02,
                     'CTYPE3': 'HPLN-TAN', 'CUNIT3': 'arcsec', 'CRPIX3': 5., 'CRVAL3': 0., 'CDELT3': 1.,
                     'CTYPE4': 'HPLT-TAN', 'CUNIT4': 'arcsec', 'CRPIX4': 4., 'CRVAL4': 0., 'CDELT4': 1.})
    source = tmp_path/'unusual.fits'
    fits.writeto(source, values, h)
    output = tmp_path/'cropped'
    main(['crop', '--input', str(source), '--output', str(output), '--spatial-axes', '3', '4',
          '--pixels', '2', '1', '8', '7'])
    actual = fits.getdata(output/source.name)
    np.testing.assert_array_equal(actual, values[1:7, 2:8, :, :])
    new_header = fits.getheader(output/source.name)
    assert new_header['CRPIX1'] == h['CRPIX1']
    assert new_header['CRPIX2'] == h['CRPIX2']
    expected = WCS(h).pixel_to_world_values(1., 2., 4., 3.)
    result = WCS(new_header).pixel_to_world_values(1., 2., 2., 2.)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_sky_image_in_degrees_and_named_science_extension(tmp_path):
    h = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN', 'CUNIT1': 'deg', 'CUNIT2': 'deg',
                     'CRPIX1': 20., 'CRPIX2': 20., 'CRVAL1': 150., 'CRVAL2': 30.,
                     'CDELT1': -.01, 'CDELT2': .01})
    source = tmp_path/'sky.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((40, 40)), h, name='SCI')]).writeto(source)
    output = tmp_path/'cropped'
    main(['crop', '--input', str(source), '--output', str(output), '--hdu', 'SCI', '--unit', 'deg',
          '--bounds', '149.95', '29.95', '150.05', '30.05'])
    with fits.open(output/source.name) as hdus:
        assert hdus[0].data is None
        assert hdus['SCI'].data.shape[0] < 40
        assert hdus['SCI'].data.shape[1] < 40
        assert 'DATE-OBS' not in hdus['SCI'].header


def test_generic_crop_accepts_a_single_hmi_segment(tmp_path):
    source = acquisition(tmp_path)
    path = next(source.glob('*.I0.fits'))
    files = crop_fits(path, tmp_path/'one', pixels=[1, 1, 5, 5])
    assert len(files) == 1
    assert fits.getdata(files[0]).shape == (4, 4)


@pytest.mark.parametrize('telescope', ['hrt', 'fdt'])
@pytest.mark.parametrize('binning', [1, 2])
def test_phi_crop_preserves_detector_coordinates_across_nested_crops(tmp_path, telescope, binning):
    from test_fits_observations import cube

    from spinnme.loader.spherical import PHIFDTSphericalDataset, PHIHRTSphericalDataset
    source, values = cube(tmp_path)
    with fits.open(source, mode='update') as hdus:
        hdus[0].data = values.transpose(3, 2, 0, 1)
        hdus[0].header.update(PXBEG1=101, PXBEG2=201, NBIN1=binning, NBIN2=binning, CONTPOS=7)
    first = crop_fits(source, tmp_path/'first', pixels=[1, 0, 5, 3])[0]
    second = crop_fits(first, tmp_path/'second', pixels=[1, 1, 4, 3])[0]
    cropped = fits.getheader(second)
    assert (cropped['PXBEG1'], cropped['PXBEG2']) == (101+2*binning, 201+binning)
    cls = PHIHRTSphericalDataset if telescope == 'hrt' else PHIFDTSphericalDataset
    dataset = cls(second, ds_id='phi', instrument_id='PHI', work_directory=str(tmp_path/'cache'),
                  stokes_normalization=1, seconds_per_dt=60, Rs_per_ds=1,
                  ref_time=datetime(2024, 3, 23), batch_size=4, shuffle=False)
    expected = np.array([100+2*binning, 200+binning])+(binning-1)/2
    np.testing.assert_allclose(np.asarray(dataset.detector_mapping[0])*2047, expected)
