import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from spinnme.data.hmi_transmission import (
    DEFAULT_BLOCKER_FWHM,
    DEFAULT_BLOCKER_TAIL_SIGMA,
    DEFAULT_CONTINUUM_PANEL_NODES,
    DEFAULT_CONTINUUM_PANEL_WIDTH,
    DEFAULT_PROFILE_HALF_WIDTH,
    DEFAULT_PROFILE_SAMPLES,
    HMI_UNTUNED_CENTER,
    MANIFEST_FORMAT,
    build_hmi_transmission_profiles,
    discover_hmi_acquisitions,
    download_phase_map,
    format_jsoc_time,
    load_hmi_transmission_manifest,
    prepare_hmi_transmission_directory,
    resolve_hmi_phase_map,
    write_transmission_file,
)

PHASE_METADATA = {
    'T_REC': '2023.10.10_19:26:26_TAI',
    'FSN_REC': 230562565,
    'HCAMID': 2,
    'NX': 128,
    'HCMNB': 78,
    'HCMWB': 46,
    'HCMPOL': 0,
    'HCME1': 38,
    'FSRNB': 0.1689,
    'FSRWB': 0.33685,
    'FSRE1': 0.695,
    'FSRE2': 1.417,
    'FSRE3': 2.779,
    'FSRE4': 5.682,
    'FSRE5': 11.354,
    'CBLOCKER': 2.7,
    'PHASENBM': -116.0,
    'PHASEWBM': 77.0,
    'PHASELYO': -124.0,
    'COMMENT': 'test phase map',
}


def test_format_jsoc_time():
    assert format_jsoc_time('2024-05-07T12:00:00') == '2024.05.07_12:00:00_TAI'
    assert format_jsoc_time('2024.05.07_12:00:00_TAI') == '2024.05.07_12:00:00_TAI'


def test_constructed_profiles_are_normalized_and_wavelength_ordered():
    phases = np.array([-116.0, 77.0, -124.0])
    profiles = build_hmi_transmission_profiles(phases, PHASE_METADATA, samples=41)

    assert profiles['offsets'].shape == (6, 41)
    assert profiles['weights'].shape == (6, 41)
    assert profiles['continuum_weights'].shape == (6,)
    np.testing.assert_allclose(
        profiles['weights'].sum(axis=-1) + profiles['continuum_weights'], 1.0, atol=1e-6
    )
    assert np.all(profiles['continuum_weights'] > 0)
    assert np.all(profiles['weights'] >= 0)

    wavelength = profiles['wavelength_grid'][:, None] + profiles['offsets']
    # Every filter must synthesize the same global interval around the line
    # center. Offsets differ because each is relative to its nominal channel.
    np.testing.assert_allclose(wavelength, np.broadcast_to(wavelength[0], wavelength.shape), atol=1e-7)
    assert wavelength.min() > -0.65
    assert wavelength.max() < 0.65
    assert wavelength.min() < -0.64
    assert wavelength.max() > 0.64
    assert not np.allclose(profiles['offsets'][0], profiles['offsets'][-1])

    inner_weights = profiles['weights'] / profiles['weights'].sum(axis=-1, keepdims=True)
    centroids = (wavelength * inner_weights).sum(axis=-1)
    assert np.all(np.diff(centroids) < 0)


def test_calibrated_profile_defaults():
    assert DEFAULT_PROFILE_HALF_WIDTH == pytest.approx(0.65)
    assert DEFAULT_PROFILE_SAMPLES >= 81
    assert DEFAULT_BLOCKER_FWHM == pytest.approx(8.43)
    assert DEFAULT_BLOCKER_TAIL_SIGMA >= 7
    assert DEFAULT_CONTINUUM_PANEL_WIDTH <= 0.1
    assert DEFAULT_CONTINUUM_PANEL_NODES >= 8
    assert HMI_UNTUNED_CENTER == pytest.approx(0.016)


def test_default_quadrature_converges_to_dense_reference():
    phases = np.array([-116.0, 77.0, -124.0])
    profiles = build_hmi_transmission_profiles(phases, PHASE_METADATA)
    reference = build_hmi_transmission_profiles(
        phases, PHASE_METADATA, samples=401, blocker_tail_sigma=8,
        continuum_panel_width=0.05, continuum_panel_nodes=12,
    )

    wavelength = profiles['wavelength_grid'][:, None] + profiles['offsets']
    reference_wavelength = reference['wavelength_grid'][:, None] + reference['offsets']

    # Representative absorption and polarization-like profiles. Stokes I gets
    # the far-wing continuum contribution; polarization has zero continuum.
    intensity = 1 - 0.6 * np.exp(-0.5 * ((wavelength - 0.12) / 0.09) ** 2)
    reference_intensity = 1 - 0.6 * np.exp(-0.5 * ((reference_wavelength - 0.12) / 0.09) ** 2)
    polarization = 0.08 * (wavelength / 0.08) * np.exp(-0.5 * (wavelength / 0.08) ** 2)
    reference_polarization = (
        0.08 * (reference_wavelength / 0.08)
        * np.exp(-0.5 * (reference_wavelength / 0.08) ** 2)
    )

    integrated_intensity = (
        (profiles['weights'] * intensity).sum(-1) + profiles['continuum_weights']
    )
    reference_integrated_intensity = (
        (reference['weights'] * reference_intensity).sum(-1) + reference['continuum_weights']
    )
    integrated_polarization = (profiles['weights'] * polarization).sum(-1)
    reference_integrated_polarization = (
        reference['weights'] * reference_polarization
    ).sum(-1)

    np.testing.assert_allclose(integrated_intensity, reference_integrated_intensity, atol=2e-4)
    np.testing.assert_allclose(integrated_polarization, reference_integrated_polarization, atol=2e-5)


def test_outer_passband_integral_converges_and_includes_secondary_orders():
    phases = np.array([-116.0, 77.0, -124.0])
    six_sigma = build_hmi_transmission_profiles(phases, PHASE_METADATA, blocker_tail_sigma=6)
    seven_sigma = build_hmi_transmission_profiles(phases, PHASE_METADATA, blocker_tail_sigma=7)
    eight_sigma = build_hmi_transmission_profiles(phases, PHASE_METADATA, blocker_tail_sigma=8)

    np.testing.assert_allclose(
        seven_sigma['continuum_weights'], eight_sigma['continuum_weights'], rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        six_sigma['continuum_weights'], seven_sigma['continuum_weights'], rtol=2e-4, atol=1e-6
    )
    assert np.all(seven_sigma['continuum_weights'] > 0.1)
    assert seven_sigma['continuum_bounds'][0] < -20
    assert seven_sigma['continuum_bounds'][1] > 25


def test_transmission_file_contains_profiles_and_provenance(tmp_path):
    profiles = build_hmi_transmission_profiles(
        np.array([-116.0, 77.0, -124.0]), PHASE_METADATA, samples=17
    )
    path = write_transmission_file(
        tmp_path / 'profiles.npz', profiles, {'format': 'spinnme.hmi_transmission.v1', 'HCAMID': 2}
    )

    with np.load(path, allow_pickle=False) as data:
        assert data['offsets'].shape == (6, 17)
        assert data['continuum_weights'].shape == (6,)
        metadata = json.loads(str(data['metadata']))
    assert metadata['format'] == 'spinnme.hmi_transmission.v1'
    assert metadata['HCAMID'] == 2


class _FakeExportRequest:
    def __init__(self, source):
        self.source = Path(source)

    def download(self, directory, fname_from_rec=False):
        target = Path(directory) / 'phases.fits'
        target.write_bytes(self.source.read_bytes())
        return pd.DataFrame([{'record': 'record', 'url': 'url', 'download': str(target)}])


class _FakeClient:
    def __init__(self, phase_fits=None):
        self.phase_fits = phase_fits
        self.export_call = None

    def query(self, record, key=None, seg=None):
        if record.startswith('hmi.ME_720s_fd10'):
            return pd.DataFrame([{'T_REC': '2024.05.07_12:00:00_TAI', 'INVPHMAP': '230562565'}])
        if record == 'hmi.phasemaps_extended[230562565]':
            keys = pd.DataFrame([
                {**PHASE_METADATA, 'HCAMID': 2},
                {**PHASE_METADATA, 'HCAMID': 3},
            ])
            segments = pd.DataFrame([
                {'phases': '/SUM/cam2/phases.fits'},
                {'phases': '/SUM/cam3/phases.fits'},
            ])
            return keys, segments
        raise AssertionError(f'Unexpected query: {record}')

    def export(self, record, method, protocol):
        self.export_call = (record, method, protocol)
        return _FakeExportRequest(self.phase_fits)


def test_resolve_phase_map_follows_invphmap_and_camera():
    client = _FakeClient()
    metadata = resolve_hmi_phase_map(client, '2024-05-07T12:00:00', hcamid=2)

    assert metadata['phase_map_fsn'] == 230562565
    assert metadata['HCAMID'] == 2
    assert metadata['segment'] == '/SUM/cam2/phases.fits'
    assert metadata['record'] == 'hmi.phasemaps_extended[230562565][][2][128]'


def test_download_phase_map_uses_drms_export(tmp_path):
    source = tmp_path / 'source.fits'
    fits.writeto(source, np.ones((2, 2, 5), dtype=np.float32))
    client = _FakeClient(source)
    download_directory = tmp_path / 'download'

    downloaded = download_phase_map(client, 'hmi.phasemaps_extended[1][][2][128]', download_directory)

    assert downloaded.is_file()
    assert client.export_call == (
        'hmi.phasemaps_extended[1][][2][128]{phases}', 'url_quick', 'as-is'
    )


def _write_hmi_segment(path, t_rec, hcamid=2):
    primary = fits.PrimaryHDU()
    image = fits.ImageHDU(np.zeros((2, 2), dtype=np.int16))
    image.header['T_REC'] = t_rec
    image.header['HCAMID'] = hcamid
    fits.HDUList([primary, image]).writeto(path)


def test_series_pipeline_deduplicates_profiles_and_writes_manifest(tmp_path):
    input_directory = tmp_path / 'fits'
    input_directory.mkdir()
    _write_hmi_segment(
        input_directory / 'hmi.S_720s.20240507_120000_TAI.1.I0.fits',
        '2024.05.07_12:00:00_TAI',
    )
    _write_hmi_segment(
        input_directory / 'hmi.S_720s.20240507_121200_TAI.1.I0.fits',
        '2024.05.07_12:12:00_TAI',
    )
    phase_cube = np.zeros((128, 128, 5), dtype=np.float32)
    phase_cube[..., :3] = np.array([-116.0, 77.0, -124.0])
    phase_source = tmp_path / 'source.fits'
    fits.writeto(phase_source, phase_cube)
    output_directory = tmp_path / 'profiles'

    manifest_path = prepare_hmi_transmission_directory(
        inputs=[str(input_directory)],
        output_directory=output_directory,
        email='registered@example.com',
        client=_FakeClient(phase_source),
        samples=17,
        continuum_panel_width=0.2,
        continuum_panel_nodes=4,
    )

    assert manifest_path == output_directory / 'manifest.json'
    manifest = load_hmi_transmission_manifest(output_directory)
    assert len(manifest['acquisitions']) == 2
    assert len(manifest['profiles']) == 1
    assert len(discover_hmi_acquisitions([input_directory])) == 2


def test_discovery_requires_standard_i0_segment(tmp_path):
    segment = tmp_path / 'nonstandard.fits'
    _write_hmi_segment(segment, '2024.05.07_12:00:00_TAI')

    with pytest.raises(ValueError, match='I0 FITS segments'):
        discover_hmi_acquisitions([tmp_path])


def test_manifest_rejects_unknown_profile_reference(tmp_path):
    (tmp_path / 'profile.npz').write_bytes(b'profile')
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'format': MANIFEST_FORMAT,
        'profiles': {'valid': {'file': 'profile.npz'}},
        'acquisitions': {'acquisition': {'profile': 'missing'}},
    }))

    with pytest.raises(ValueError, match='Invalid HMI acquisition'):
        load_hmi_transmission_manifest(tmp_path)
