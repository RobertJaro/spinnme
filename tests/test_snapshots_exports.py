import json
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from sunpy.map import make_fitswcs_header

import spinnme
from spinnme.exports import EvaluationResult, read_npz
from spinnme.snapshots import save_snapshot
from spinnme.train.me_spherical_module import MESphericalModule


@pytest.mark.parametrize('device', [None, 'auto'])
@pytest.mark.parametrize('cuda_available', [False, True])
def test_automatic_snapshot_device_prefers_cuda_with_cpu_fallback(saved_snapshot, monkeypatch, device, cuda_available):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda_available)
    # Verify selection without allocating CUDA tensors on a CPU-only test host.
    transfers = []
    def transfer(module, target, *args, **kwargs):
        transfers.append(str(target))
        return module
    monkeypatch.setattr(torch.nn.Module, 'to', transfer)
    with spinnme.load(saved_snapshot[0], device=device) as snapshot:
        expected = 'cuda' if cuda_available else 'cpu'
        assert str(snapshot.device) == expected
        assert expected in transfers


def test_explicit_cpu_snapshot_override_is_retained_when_cuda_available(saved_snapshot, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        assert snapshot.device.type == 'cpu'


@pytest.fixture
def saved_snapshot(tmp_path):
    torch.manual_seed(3)
    shape = (7, 9)
    reference = SkyCoord(0 * u.arcsec, 0 * u.arcsec, observer="earth", obstime="2024-03-23T01:00:00", frame="helioprojective")
    header = fits.Header(make_fitswcs_header(np.zeros(shape), reference, scale=[100, 100] * u.arcsec / u.pix))
    header["OBS_VR"] = header["OBS_VW"] = header["OBS_VN"] = 0.
    wavelengths = {"USER": {"wavelength_center": 6173.3433 * u.AA, "wavelength_grid": np.array([-.2, -.07, .01, .1, .3]) * u.AA}}
    config = {"model": {"type": "mlp", "dim": 16, "n_layers": 2, "encoding_config": {"type": "identity"}},
              "instrument": [{"type": "generic", "instrument_id": "USER", "line_id": "fe-i-6173", "artifact_correction": True}],
              "stokes_loss": {"type": "mse", "continuum_normalization": True}, "data": {}, "training": {"max_steps": 2}}
    module = MESphericalModule(image_shape=shape, wavelength_config=wavelengths, gauss_per_dB=1000,
                               Rs_per_ds=1., seconds_per_dt=3600, model_config=config["model"],
                               instrument_config=config["instrument"], stokes_loss_config=config["stokes_loss"])
    observation = {"id": "example", "instrument_id": "USER", "line_id": "fe-i-6173", "shape": list(shape),
                   "header": dict(header), "time": {"model_time": "2024-03-23T01:00:00"}}
    calibration = tmp_path / 'response.npz'
    np.savez(calibration, example=np.arange(6))
    data = SimpleNamespace(image_shape=shape, wavelength_config=wavelengths, gauss_per_dB=1000,
                           Rs_per_ds=1., seconds_per_dt=3600, ref_time=datetime(2024, 3, 23, 1),
                           times=[datetime(2024, 3, 23, 1)], observations=[observation],
                           spectral_response_files={'USER': [str(calibration)]}, spectral_response_by_acquisition={})
    path = save_snapshot(module, data, config, tmp_path / "inversion.pme")
    calibration.unlink()
    return path, module, header


def test_portable_snapshot_maps_and_all_writers(saved_snapshot, tmp_path):
    path, module, header = saved_snapshot
    with spinnme.load(path, device="cpu") as snapshot:
        with np.load(snapshot.spectral_response_files['USER'][0]) as asset:
            np.testing.assert_array_equal(asset['example'], np.arange(6))
        for key, tensor in module.state_dict().items():
            torch.testing.assert_close(snapshot.module.state_dict()[key], tensor)
        result = snapshot.evaluate(parameters=["Brtp", "Blos", "Vdop", "vmac", "damping"])
        assert result.arrays["Br"].shape == (7, 9)
        assert result.mask.all()
        assert np.isfinite(result.arrays["Br"]).all()
        first = snapshot.evaluate(parameters=["Br"], batch_size=8)
        np.testing.assert_allclose(first.arrays["Br"], result.arrays["Br"], atol=1e-4)
        files = result.write_fits(tmp_path / "fits")
        assert len(files) == len(result.arrays)
        with fits.open(files[0], checksum=True) as hdus:
            assert hdus[0].header["BUNIT"] == "G"
            assert hdus["VALID_MASK"].data.all()
            original = WCS(header).celestial.pixel_to_world_values([0, 4, 8], [0, 3, 6])
            exported = WCS(hdus[0].header).celestial.pixel_to_world_values([0, 4, 8], [0, 3, 6])
            np.testing.assert_allclose(original, exported, rtol=0, atol=1e-10)
        npz = result.write_npz(tmp_path / "npz")[0]
        recovered = read_npz(npz)
        np.testing.assert_array_equal(recovered.arrays["Br"], result.arrays["Br"])
        assert recovered.units == result.units
        assert first.write_jpg(tmp_path / "jpg")[0].is_file()
        with pytest.raises(FileExistsError):
            result.write_npz(npz)


def test_spectral_fits_tab_wcs_and_artifact_gate(saved_snapshot, tmp_path):
    path, _, _ = saved_snapshot
    with spinnme.load(path) as snapshot:
        result = snapshot.evaluate(parameters=["I", "Q", "U", "V", "artifact_gate", "stokes"])
        assert result.arrays["I"].shape == (7, 9, 5)
        np.testing.assert_allclose(result.arrays["artifact_gate"], .5)
        # A correction parameter vector is not a spectral axis; export scalar gate separately.
        result.arrays.pop("artifact_gate")
        result.write_fits(tmp_path / "spectra")
        with fits.open(tmp_path / "spectra" / "I.fits") as hdus:
            wcs = WCS(hdus[0].header, hdus)
            _, _, wave = wcs.pixel_to_world_values(np.zeros(5), np.zeros(5), np.arange(5))
            np.testing.assert_allclose(wave, (6173.3433 + np.array([-.2, -.07, .01, .1, .3])) * 1e-10, rtol=1e-12)
        with fits.open(tmp_path / "spectra" / "stokes.fits") as hdus:
            assert hdus[0].data.shape == (4, 5, 7, 9)
            assert WCS(hdus[0].header, hdus).pixel_n_dim == 4


def test_reject_snapshot_without_format_metadata(tmp_path):
    path = tmp_path / "unsupported.pme"
    torch.save({"coordinate_order": "xyzt"}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        spinnme.load(path)


@pytest.mark.parametrize('payload', [None, [], torch.zeros(1)])
def test_reject_nonmapping_snapshot_payload(tmp_path, payload):
    path = tmp_path / 'invalid.pme'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='Unsupported .pme payload'):
        spinnme.load(path)


@pytest.mark.parametrize('options', [
    {'batch_size': 0}, {'batch_size': True}, {'batch_size': 2.5}, {'batch_size': np.nan},
    {'mu_limit': True}, {'mu_limit': np.nan}, {'mu_limit': np.inf}, {'mu_limit': '0.1'},
])
def test_invalid_snapshot_evaluation_controls_fail_before_geometry(saved_snapshot, options):
    with (spinnme.load(saved_snapshot[0], device='cpu') as snapshot,
          pytest.raises(ValueError, match='batch_size|mu_limit')):
        snapshot.evaluate(parameters=['Br'], **options)


def test_synthesis_numpy_pixels_without_artifact_model(saved_snapshot):
    with spinnme.load(saved_snapshot[0]) as snapshot:
        snapshot.artifact_correction_models = torch.nn.ModuleDict()
        result = snapshot.evaluate(parameters=['I', 'Q', 'U', 'V'])
        assert result.arrays['I'].shape == (7, 9, 5)
        assert np.isfinite(result.arrays['I']).all()


def test_external_grid_requires_explicit_observer_metadata(saved_snapshot, tmp_path):
    path = tmp_path/'grid.fits'
    header = saved_snapshot[2].copy()
    fits.writeto(path, np.zeros((7,9), dtype=np.float32), header)
    with spinnme.load(saved_snapshot[0]) as snapshot:
        result = snapshot.evaluate(path, parameters=['Br'])
        assert result.mask.all()
        del header['DSUN_OBS']
        fits.writeto(path, np.zeros((7,9), dtype=np.float32), header, overwrite=True)
        with pytest.raises(ValueError, match='Explicit observer geometry'):
            snapshot.evaluate(path, parameters=['Br'])


def test_native_raster_times_masks_and_explicit_epoch(saved_snapshot, tmp_path):
    import base64

    from astropy.time import Time
    payload = torch.load(saved_snapshot[0], weights_only=True)
    metadata = json.loads(payload['metadata_json'])
    observation = metadata['observations'][0]
    offsets = np.arange(9) * 120.
    observation.update(time_offsets_seconds=offsets.tolist(), time_offset_axis='x')
    mask = np.ones((7, 9), dtype=bool)
    mask[2, 4] = False
    observation['valid_mask_bits'] = base64.b64encode(np.packbits(mask).tobytes()).decode()
    payload['metadata_json'] = json.dumps(metadata)
    path = tmp_path / 'raster.pme'
    torch.save(payload, path)
    with spinnme.load(path) as snapshot:
        raster = snapshot.evaluate(parameters=['B_los'])
        assert not raster.mask[2, 4]
        assert np.isnan(raster.arrays['B_los'][2, 4])
        for column in [0, 3, 8]:
            epoch = Time('2024-03-23T01:00:00') + offsets[column] * u.s
            simultaneous = snapshot.evaluate(parameters=['B_los'], time=epoch)
            np.testing.assert_allclose(raster.arrays['B_los'][:, column], simultaneous.arrays['B_los'][:, column], atol=1e-4)
        assert raster.metadata['raster_time_offsets_applied']
        raster.write_fits(tmp_path/'native-raster')
        with fits.open(tmp_path/'native-raster/B_los.fits') as hdus:
            np.testing.assert_array_equal(hdus['TIME_OFFSETS'].data, np.broadcast_to(offsets, (7, 9)))
            assert hdus['TIME_OFFSETS'].header['BUNIT'] == 's'
            assert Time(hdus['TIME_OFFSETS'].header['DATEREF']) == Time('2024-03-23T01:00:00')
        simultaneous.write_fits(tmp_path/'simultaneous')
        with fits.open(tmp_path/'simultaneous/B_los.fits') as hdus:
            assert 'TIME_OFFSETS' not in hdus


def test_physical_parameters_do_not_depend_on_stokes_selection(saved_snapshot):
    with spinnme.load(saved_snapshot[0]) as snapshot:
        intrinsic = snapshot.evaluate(parameters=['b0', 'vdop'])
        with_spectra = snapshot.evaluate(parameters=['b0', 'vdop', 'I', 'Artifact', 'forward_vdop', 'forward_vmac', 'forward_b0'])
        np.testing.assert_array_equal(intrinsic.arrays['b0'], with_spectra.arrays['b0'])
        np.testing.assert_array_equal(intrinsic.arrays['vdop'], with_spectra.arrays['vdop'])
        assert with_spectra.arrays['artifact_alpha_Q'].shape == (7, 9)
        assert with_spectra.units['forward_vdop'] == 'm / s'


def test_vector_potential_and_physics_domain_serialization(saved_snapshot, tmp_path):
    from spinnme.snapshots import decode
    from spinnme.train.collocation import LongitudeInterval, SphericalCollocationDomain
    payload = torch.load(saved_snapshot[0], weights_only=True)
    metadata = decode(json.loads(payload['metadata_json']))
    init = dict(metadata['init'])
    init['model_config'] = {**init['model_config'], 'vector_potential': True}
    init['physics_domain'] = SphericalCollocationDomain((0, 1), (-.1, .1), LongitudeInterval(0, .2), (1, 1.1))
    module = MESphericalModule(**init)
    data = SimpleNamespace(ref_time=metadata['ref_time'], times=metadata['times'], observations=metadata['observations'],
                           spectral_response_files={}, spectral_response_by_acquisition={})
    path = save_snapshot(module, data, metadata['config'], tmp_path / 'vector.pme')
    with spinnme.load(path) as snapshot:
        result = snapshot.evaluate(parameters=['Brtp', 'I'])
        assert np.isfinite(result.arrays['Br']).all()
        assert isinstance(snapshot.metadata['init']['physics_domain'], SphericalCollocationDomain)


def test_reference_metrics_match_and_wrap_angles(saved_snapshot):
    from spinnme.evaluation import evaluate_result
    with spinnme.load(saved_snapshot[0]) as snapshot:
        result = snapshot.evaluate(parameters=["B_los", "azimuth"])
    reference = EvaluationResult({name: array.copy() for name, array in result.arrays.items()}, result.units, result.header, result.mask, {})
    reference.arrays["B_los"] -= 5
    reference.arrays["azimuth"] += 180
    metrics = evaluate_result(result, reference)["metrics"]
    assert metrics[0]["bias"] == pytest.approx(5, abs=1e-5)
    assert metrics[1]["mae"] == pytest.approx(0, abs=2e-5)


def test_fits_reference_units_and_observer_guard(saved_snapshot, tmp_path):
    from spinnme.evaluation import evaluate_result, read_reference
    with spinnme.load(saved_snapshot[0]) as snapshot:
        result = snapshot.evaluate(parameters=['Br'])
    header = result.header.copy()
    header['BUNIT'] = 'T'
    path = tmp_path/'reference.fits'
    fits.writeto(path, result.arrays['Br'] * 1e-4, header)
    reference = read_reference(path, result)
    assert evaluate_result(result, reference)['metrics'][0]['rmse'] < 1e-4
    header['DSUN_OBS'] *= .5
    fits.writeto(path, result.arrays['Br'] * 1e-4, header, overwrite=True)
    with pytest.raises(ValueError, match='observer geometry'):
        read_reference(path, result)


def test_fits_reference_preserves_independent_valid_mask(saved_snapshot, tmp_path):
    from spinnme.evaluation import evaluate_result, read_reference
    with spinnme.load(saved_snapshot[0]) as snapshot:
        result = snapshot.evaluate(parameters=['Br'])
    mask = result.mask.copy()
    mask[0, 0] = False
    values = result.arrays['Br'].copy()
    values[0, 0] = 1e6
    reference = EvaluationResult({'Br': values}, result.units, result.header, mask, result.metadata)
    path = reference.write_fits(tmp_path/'reference')[0]
    recovered = read_reference(path, result)
    np.testing.assert_array_equal(recovered.mask, mask)
    metrics = evaluate_result(result, recovered)['metrics'][0]
    assert metrics['count'] == int(mask.sum())
    assert metrics['rmse'] == 0.


@pytest.mark.parametrize('group', ['LOS', 'ME', 'ME_forward'])
def test_broad_parameter_groups_require_explicit_keys(group):
    from spinnme.exports import expand_parameters
    with pytest.raises(ValueError, match='select explicit parameter keys'):
        expand_parameters([group])


def test_explicit_blos_vdop_keys_match_physical_maps_and_export_names(saved_snapshot, tmp_path):
    path, _, _ = saved_snapshot
    with spinnme.load(path, device='cpu') as snapshot:
        result = snapshot.evaluate(parameters=['Brtp', 'Vdop', 'Blos', 'vmac', 'damping'])
        reference = snapshot.evaluate(parameters=['B_los', 'v_los'])
    assert set(result.arrays) == {'Br', 'Btheta', 'Bphi', 'Vdop', 'Blos', 'vmac', 'damping'}
    np.testing.assert_array_equal(result.arrays['Blos'], reference.arrays['B_los'])
    np.testing.assert_array_equal(result.arrays['Vdop'], reference.arrays['v_los'])
    assert result.units['Blos'] == 'G'
    assert result.units['Vdop'] == 'm / s'
    files = result.write_fits(tmp_path/'fits')
    assert {p.stem for p in files} == set(result.arrays)
    with fits.open(tmp_path/'fits/Vdop.fits') as hdus:
        assert hdus[0].header['BUNIT'] == 'm / s'
        assert 'positive redshift' in str(hdus[0].header['HISTORY'])
    recovered = read_npz(result.write_npz(tmp_path/'maps.npz')[0])
    np.testing.assert_array_equal(recovered.arrays['Vdop'], result.arrays['Vdop'])
    previews = result.write_jpg(tmp_path/'jpg')
    assert any(p.name == 'Blos.jpg' for p in previews)
    assert any(p.name == 'Vdop.jpg' for p in previews)


def test_external_grid_exports_without_observer_velocity(saved_snapshot, tmp_path):
    path = tmp_path/'no-velocity.fits'
    header = saved_snapshot[2].copy()
    for key in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
        del header[key]
    fits.writeto(path, np.zeros((7, 9), dtype=np.float32), header)
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        result = snapshot.evaluate(path, parameters=['Vdop', 'I'])
        solver = snapshot.evaluate(path, parameters=['vdop'])
    np.testing.assert_array_equal(result.arrays['Vdop'], solver.arrays['vdop'])
    assert not result.metadata['observer_velocity_correction_applied']
    assert result.metadata['observer_velocity'] == [0., 0., 0.]
    assert result.mask.all()
    result.write_fits(tmp_path/'no-velocity-export')


@pytest.mark.parametrize('format', ['fits', 'jpg'])
def test_case_only_export_names_never_overwrite_each_other(saved_snapshot, tmp_path, format):
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        result = snapshot.evaluate(parameters=['Vdop', 'vdop'])
    directory = tmp_path/format
    with pytest.raises(ValueError, match='filenames collide'):
        result.write(directory, format=format, overwrite=True)
    assert not list(directory.iterdir())


def test_public_grid_map_and_coordinate_interfaces_agree(saved_snapshot, tmp_path):
    from sunpy.map import Map
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        result = snapshot.evaluate(parameters=['Brtp', 'Vrtp', 'Vlos'])
        assert snapshot.config == snapshot.metadata['config']
        assert snapshot.times == snapshot.metadata['times']
        for name, map_ in result.maps.items():
            np.testing.assert_array_equal(map_.data, result.arrays[name])
            assert map_.unit == u.Unit(result.units[name])
        grid = snapshot.grid(Map(np.zeros((7, 9)), saved_snapshot[2]), shape=(5, 6))
        resized = snapshot.evaluate(grid, parameters=['Brtp', 'Vrtp', 'Vlos', 'I'])
        assert resized.mask.shape == (5, 6)
        assert resized.arrays['I'].shape == (5, 6, 5)
        context = Map(np.zeros((5, 6)), grid['header'])
        y, x = np.indices((5, 6))
        coordinates = context.pixel_to_world(x*u.pix, y*u.pix)
        points = snapshot.sample(coordinates, observation=grid, parameters=['Brtp', 'Vrtp', 'Vlos', 'I'], batch_size=11)
        assert points.header is None
        for name in resized.arrays:
            np.testing.assert_allclose(points.arrays[name], resized.arrays[name], atol=.02, rtol=2e-5)
        restored = read_npz(points.write_npz(tmp_path/'points.npz')[0])
        np.testing.assert_array_equal(restored.arrays['Br'], points.arrays['Br'])
        assert restored.header is None
        assert restored.metadata['coordinates']['shape'] == [5, 6]
        with pytest.raises(ValueError, match='no image WCS'):
            restored.write_fits(tmp_path/'invalid')
        single = snapshot.sample(coordinates[2, 3], observation=grid, parameters=['B'])
        assert single.mask.shape == () and single.arrays['B'].shape == ()


def test_coordinate_scan_times_across_batches(saved_snapshot, tmp_path, monkeypatch):
    from astropy.time import Time
    from sunpy.map import Map
    import spinnme.progress

    bars = []
    def progress(batches, **kwargs):
        bars.append(list(batches))
        return batches
    monkeypatch.setattr(spinnme.progress, 'progress', progress)
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        grid = Map(np.zeros((7, 9)), saved_snapshot[2])
        y, x = np.indices((3, 5))
        coordinates = grid.pixel_to_world((x+2)*u.pix, (y+2)*u.pix)
        times = Time('2024-03-23T01:00:00') + np.array([.123, 20.456, 67.891, 92.345, 130.678])*u.s
        combined = snapshot.sample(coordinates, time=times, parameters=['B_los'], batch_size=4, progress=True)
        assert bars == [[0, 4, 8, 12]]
        xyzt = np.asarray(combined.metadata['coordinates']['values']).reshape(3, 5, 4)
        np.testing.assert_allclose(xyzt[..., -1], np.broadcast_to(snapshot._normalize_time(times), (3, 5)))
        for column in range(5):
            slit = snapshot.sample(coordinates[:, column], time=times[column], parameters=['B_los'])
            np.testing.assert_allclose(combined.arrays['B_los'][:, column], slit.arrays['B_los'], atol=.001)
            np.testing.assert_array_equal(combined.mask[:, column], slit.mask)
        restored = read_npz(combined.write_npz(tmp_path/'scan.npz')[0])
        assert restored.metadata['model_time'] == times.utc.isot.tolist()
        with pytest.raises(ValueError, match='broadcast'):
            snapshot.sample(coordinates, time=times[:2], parameters=['B_los'])
        with pytest.raises(ValueError, match='Array times require'):
            snapshot.evaluate(time=times, parameters=['B_los'])


def test_coordinate_scan_times_with_default_batch_size(saved_snapshot):
    from astropy.time import Time
    from sunpy.map import Map

    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        grid = Map(np.zeros((7, 9)), saved_snapshot[2])
        indices = np.arange(8193)
        coordinates = grid.pixel_to_world((2+indices % 5)*u.pix, (2+indices % 3)*u.pix)
        times = Time('2024-03-23T01:00:00') + (indices*.125)*u.s
        combined = snapshot.sample(coordinates, time=times, parameters=['B_los'])
        assert combined.mask.all()
        assert combined.arrays['B_los'].shape == (8193,)
        xyzt = np.asarray(combined.metadata['coordinates']['values'])
        np.testing.assert_allclose(xyzt[:, -1], snapshot._normalize_time(times))
        for index in (0, 8191, 8192):
            point = snapshot.sample(coordinates[index], time=times[index], parameters=['B_los'])
            np.testing.assert_allclose(combined.arrays['B_los'][index], point.arrays['B_los'], atol=.001)


def test_crop_preserves_native_scan_support_and_calibration(saved_snapshot):
    import base64

    from sunpy.map import Map
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        row = snapshot.grid()
        mask = np.ones((7, 9), dtype=bool)
        mask[3, 4] = False
        row.update(valid_mask_bits=base64.b64encode(np.packbits(mask).tobytes()).decode(),
                   time_offsets_seconds=(np.arange(9)*30.).tolist(), time_offset_axis='x',
                   detector_mapping=[[.1, .2], [.01, .02]], pix_convention='detector',
                   acquisition_key='test-acquisition')
        crop = snapshot.grid(row, bounds=(-210, -210, 210, 210), shape=(5, 5))
        assert row['shape'] == [7, 9]  # No mutation of caller or snapshot metadata.
        assert crop['acquisition_key'] == row['acquisition_key']
        assert crop['time_offset_axis'] == 'yx'
        assert not snapshot.evaluate(crop, parameters=['B']).mask[2, 2]
        geometry = snapshot.geometry(crop)
        native_map = Map(np.zeros((7, 9)), row['header'])
        crop_map = Map(np.zeros((5, 5)), crop['header'])
        y, x = np.indices((5, 5))
        nx, ny = native_map.wcs.world_to_pixel_values(*crop_map.wcs.pixel_to_world_values(x, y))
        expected_detector = np.stack([nx, ny], -1)*[.01, .02]+[.1, .2]
        np.testing.assert_allclose(geometry['detector_coords'], expected_detector.reshape(-1, 2), atol=1e-10)
        simultaneous = snapshot.geometry(crop, time='2024-03-23T01:00:00')
        expected_offsets = np.asarray(crop['time_offsets_seconds']).reshape(-1)/snapshot.seconds_per_dt
        np.testing.assert_allclose(geometry['coords'][:, -1]-simultaneous['coords'][:, -1], expected_offsets)
        native = snapshot.geometry(row)
        np.testing.assert_allclose(native['pix'], 2*native['detector_coords'][:, ::-1]-1)


def test_full_state_strict_loading_and_immutable_grids(saved_snapshot, tmp_path):
    payload = torch.load(saved_snapshot[0], weights_only=True)
    payload['state_dict'].pop(next(iter(payload['state_dict'])))
    incomplete = tmp_path/'incomplete.pme'
    torch.save(payload, incomplete)
    with pytest.raises(RuntimeError, match='Missing key'):
        spinnme.load(incomplete, device='cpu')
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        grid = snapshot.grid()
        grid['header']['DSUN_OBS'] = 0
        assert snapshot.observations[0]['header']['DSUN_OBS'] > 0
        for shape in ((1, 2), (True, 2), (2.5, 3)):
            with pytest.raises(ValueError, match='shape'):
                snapshot.grid(shape=shape)
        with pytest.raises(ValueError, match='bounds'):
            snapshot.grid(bounds=(1, 2, -1, -2))
        with pytest.raises(ValueError, match='SkyCoord'):
            snapshot.sample(np.zeros((3, 3)), parameters=['B'])


def test_public_training_cadence_includes_scan_offsets(saved_snapshot):
    from astropy.time import Time
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        row = snapshot.observations[0]
        row.update(role='train', time_offsets_seconds=[0., 120.], time_offset_axis='x')
        times = snapshot.cadence_times(30)
        np.testing.assert_allclose((times-Time('2024-03-23T01:00:00')).to_value(u.s), [0, 30, 60, 90, 120])


def test_exported_fits_grid_preserves_mask_timing_and_model_epoch(saved_snapshot, tmp_path):
    import base64
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        row = snapshot.grid()
        mask = np.ones((7, 9), dtype=bool)
        mask[0, 0] = False
        row.update(valid_mask_bits=base64.b64encode(np.packbits(mask).tobytes()).decode(),
                   time_offsets_seconds=(np.arange(9)*30.).tolist(), time_offset_axis='x')
        raster = snapshot.evaluate(row, parameters=['Br'])
        path = raster.write_fits(tmp_path/'raster')[0]
        loaded = snapshot.evaluate(path, parameters=['Br'])
        np.testing.assert_allclose(loaded.arrays['Br'], raster.arrays['Br'], equal_nan=True)
        assert loaded.metadata['raster_time_offsets_applied']
        instant = snapshot.evaluate(row, time='2024-03-23T01:15:00', parameters=['Br'])
        reloaded = snapshot.evaluate(instant.write_fits(tmp_path/'instant')[0], parameters=['Br'])
        np.testing.assert_allclose(reloaded.arrays['Br'], instant.arrays['Br'], equal_nan=True)
        assert reloaded.metadata['model_time'] == instant.metadata['model_time']


def test_multinstrument_state_and_explicit_instrument_selection(saved_snapshot, tmp_path):
    from copy import deepcopy
    with spinnme.load(saved_snapshot[0], device='cpu') as original:
        init = deepcopy(original.metadata['init'])
        init['wavelength_config']['OTHER'] = deepcopy(init['wavelength_config']['USER'])
        init['instrument_config'].append({'type': 'generic', 'instrument_id': 'OTHER', 'line_id': 'fe-i-6173'})
        module = MESphericalModule(**init)
        data = SimpleNamespace(ref_time=original.ref_time, times=original.times,
            observations=original.observations, spectral_response_files=original.spectral_response_files,
            spectral_response_by_acquisition={})
        path = save_snapshot(module, data, original.config, tmp_path/'multi.pme')
    with spinnme.load(path, device='cpu') as snapshot:
        assert set(snapshot.instrument_ids) == {'USER', 'OTHER'}
        assert snapshot.instrument_id is None
        for key, value in module.state_dict().items():
            torch.testing.assert_close(snapshot.module.state_dict()[key], value)
        result = snapshot.evaluate('example', instrument_id='OTHER', parameters=['Brtp', 'IQUV'])
        assert result.metadata['instrument_id'] == 'OTHER'
        assert result.arrays['I'].shape == (7, 9, 5)


@pytest.mark.parametrize('explicit_distance', [False, True])
def test_scanned_point_geometry_matches_sunpy(saved_snapshot, explicit_distance, monkeypatch):
    from astropy.time import Time
    from sunpy.coordinates import frames
    from spinnme.grids import GridGeometry
    from spinnme.data.util import (cartesian_to_spherical_matrix, image_to_spherical_matrix,
                                   observer_ray_and_mu, spherical_to_cartesian)

    times = Time('2024-03-23T01:00:00') + np.array([0., 0., 37.123, 37.123, 37.123, 100., 100.])*u.s
    # Include distinct observers at the same epoch, repeated slits and an
    # off-disk ray. Explicit 3-D points also exercise non-surface radii.
    observer = SkyCoord(np.array([0., 0., .01, .01, .02, .03, .03])*u.deg,
                        np.full(7, 6.)*u.deg, np.full(7, 1.49e11)*u.m,
                        frame=frames.HeliographicStonyhurst(obstime=times))
    kwargs = {'distance': np.linspace(1.48e11, 1.485e11, 7)*u.m} if explicit_distance else {}
    coordinates = SkyCoord(np.array([-200., -50., 0., 50., 300., 1200., -300.])*u.arcsec,
                           np.linspace(-200, 200, 7)*u.arcsec,
                           frame=frames.Helioprojective(observer=observer, obstime=times), **kwargs)
    physical = SkyCoord(coordinates.make_3d())
    central = observer.transform_to(frames.HeliographicCarrington(observer=observer, obstime=times))
    carrington = physical.transform_to(frames.HeliographicCarrington(observer=observer, obstime=times))
    lon, lat = carrington.lon.to_value(u.rad), carrington.lat.to_value(u.rad)
    radius = carrington.radius.to_value(u.Rsun)
    spherical = np.stack([radius, np.pi/2-lat, lon], axis=-1)
    with spinnme.load(saved_snapshot[0], device='cpu') as snapshot:
        context = GridGeometry(snapshot.grid())
        normalized_time = snapshot._normalize_time(times)
        actual = context.points(coordinates, Rs_per_ds=1., normalized_time=normalized_time, include_pixels=False)
        expected_xyz = spherical_to_cartesian(spherical)
        np.testing.assert_allclose(actual['coords'][:, :3], expected_xyz, atol=2e-12)
        np.testing.assert_array_equal(actual['coords'][:, 3], normalized_time)
        np.testing.assert_allclose(actual['cartesian_to_spherical_transform'],
                                   cartesian_to_spherical_matrix(spherical), atol=2e-12)
        distance = (observer.radius/context.map.rsun_meters).to_value(u.one)
        expected_matrix = image_to_spherical_matrix(lon, lat, central.lon.to_value(u.rad),
                                                    central.lat.to_value(u.rad), context.angle,
                                                    observer_distance=distance, radius=radius)
        np.testing.assert_allclose(actual['rtp_to_img_transform'], np.swapaxes(expected_matrix, -1, -2), atol=2e-12)
        _, expected_mu = observer_ray_and_mu(lon, lat, central.lon.to_value(u.rad),
                                            central.lat.to_value(u.rad), distance, radius=radius)
        np.testing.assert_allclose(actual['mu'][:, 0], expected_mu, atol=2e-12)
        assert 'pix' not in actual and 'detector_coords' not in actual

        maps_only = snapshot.sample(physical, time=times, parameters=['B_los'])
        with_spectra = snapshot.sample(physical, time=times, parameters=['B_los', 'I'])
        np.testing.assert_array_equal(maps_only.mask, with_spectra.mask)
        np.testing.assert_allclose(maps_only.arrays['B_los'], with_spectra.arrays['B_los'], atol=1e-4)

        transform = SkyCoord.transform_to
        transformed_observers = []
        def track(self, frame, *args, **kwargs):
            if isinstance(self.frame, frames.Helioprojective):
                pytest.fail('Magnetic scan sampling must not reproject every point with SunPy.')
            if isinstance(getattr(frame, 'frame', frame), frames.HeliographicCarrington):
                transformed_observers.append(self.size)
            return transform(self, frame, *args, **kwargs)
        monkeypatch.setattr(SkyCoord, 'transform_to', track)
        sampled = snapshot.sample(physical, time=times, parameters=['B', 'B_los', 'azimuth'])
        assert sampled.mask.sum() == (7 if explicit_distance else 6)
        assert [size for size in transformed_observers if size > 1] == [4]


def test_legacy_snapshot_retains_original_predictions(saved_snapshot, tmp_path):
    path, _, _ = saved_snapshot
    with spinnme.load(path, device='cpu') as current:
        expected = current.evaluate(parameters=['I', 'Q', 'U', 'V', 'azimuth'])
    payload = torch.load(path, weights_only=True)
    payload['format_version'] = 1
    metadata = json.loads(payload['metadata_json'])
    metadata['conventions'].pop('version')
    metadata['init'].pop('coordinate_convention_version')
    payload['metadata_json'] = json.dumps(metadata)
    legacy = tmp_path/'legacy.pme'
    torch.save(payload, legacy)
    with spinnme.load(legacy, device='cpu') as model:
        assert model.module.coordinate_convention_version == 1
        actual = model.evaluate(parameters=list(expected.arrays))
    for key in expected.arrays:
        np.testing.assert_array_equal(actual.arrays[key], expected.arrays[key])


def test_checkpoint_rejects_legacy_coordinate_conventions(saved_snapshot):
    _, module, _ = saved_snapshot
    checkpoint = {'state_dict': module.state_dict()}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint['spinnme_coordinate_convention_version'] == 2
    module.on_load_checkpoint(checkpoint)
    checkpoint.pop('spinnme_coordinate_convention_version')
    with pytest.raises(ValueError, match='coordinate conventions'):
        module.on_load_checkpoint(checkpoint)


def test_snapshot_rejects_mismatched_coordinate_metadata(saved_snapshot, tmp_path):
    payload = torch.load(saved_snapshot[0], weights_only=True)
    metadata = json.loads(payload['metadata_json'])
    metadata['init']['coordinate_convention_version'] = 1
    payload['metadata_json'] = json.dumps(metadata)
    path = tmp_path/'inconsistent.pme'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='convention metadata'):
        spinnme.load(path, device='cpu')


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('date_source', ['t_obs', 'date_obs', 'explicit'])
def test_hmi_snapshot_uses_actual_dates_independent_of_version(saved_snapshot, tmp_path, legacy, date_source):
    import warnings
    from astropy.time import Time
    from sunpy.map import Map
    path, _, header = saved_snapshot
    if legacy:
        payload = torch.load(path, weights_only=True)
        payload['format_version'] = 1
        metadata = json.loads(payload['metadata_json'])
        metadata['conventions'].pop('version')
        metadata['init'].pop('coordinate_convention_version')
        payload['metadata_json'] = json.dumps(metadata)
        path = tmp_path/'old-time.pme'
        torch.save(payload, path)
    with warnings.catch_warnings(record=True) as caught, spinnme.load(path, device='cpu') as snapshot:
        warnings.simplefilter('always')
        header.update(INSTRUME='HMI_SIDE1', TELESCOP='SDO/HMI')
        if date_source != 'date_obs':
            header['T_OBS'] = '2024.03.23_01:01:22.043_TAI'
        timing = {'model_time': '2024-03-23T01:02:00'}
        if date_source == 'explicit':
            timing['geometry_time'] = '2024-03-23T00:59:30'
        row = snapshot.grid({'id': 'hmi', 'instrument_id': 'USER', 'shape': [7, 9],
                             'header': dict(header), 'time': timing})
        expected = Time({'date_obs': '2024-03-23T01:00:00', 't_obs': '2024-03-23T01:00:45.043',
                         'explicit': '2024-03-23T00:59:30'}[date_source])
        coordinate_epoch = Map(np.zeros((7, 9)), row['header']).reference_date
        assert abs((coordinate_epoch-expected).to_value(u.s)) < 1e-6
        assert abs((Time(row['time']['geometry_time'])-expected).to_value(u.s)) < 1e-6
        assert row['time']['model_time'] == timing['model_time']
        assert row['header']['DATE-OBS'] == header['DATE-OBS']
        result = snapshot.evaluate(row, parameters=['B'], time='2024-03-23T01:03:00')
        assert Time(result.metadata['model_time']) == Time('2024-03-23T01:03:00')
        assert abs((Time(result.metadata['geometry_time'])-expected).to_value(u.s)) < 1e-6
        file = result.write_fits(tmp_path/'dated-export')[0]
        restored = snapshot.grid(file)
        assert restored['time']['model_time'] == result.metadata['model_time']
        assert restored['time']['geometry_time'] == result.metadata['geometry_time']
    assert not any('legacy' in str(w.message).lower() for w in caught)
