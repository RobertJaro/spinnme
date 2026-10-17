from datetime import datetime

import numpy as np
import pytest

from spinnme.loader.polarization import polarization_selection


def scene():
    rng = np.random.default_rng(42)
    stokes = rng.normal(0, .002, (41, 41, 4, 6))
    stokes[..., 0, :] = 1
    stokes[18:23, 18:23, 3, :] += .1
    y, x = np.ogrid[:41, :41]
    radius = np.hypot(x-20, y-20)/20
    return stokes, radius


def test_filter_excludes_limb_before_smoothing_and_is_scale_invariant():
    stokes, radius = scene()
    original = stokes.copy()
    mask, score, stats = polarization_selection(stokes, radius)
    np.testing.assert_array_equal(stokes, original)
    stokes[radius >= .95] = 1e8
    other, smoothed, _ = polarization_selection(stokes, radius)
    np.testing.assert_array_equal(mask, other)
    np.testing.assert_allclose(score, smoothed, equal_nan=True)
    scaled, _, _ = polarization_selection(original * 50000, radius)
    np.testing.assert_array_equal(mask, scaled)
    assert np.isnan(score[radius >= .95]).all()
    assert not mask[radius >= .95].any()
    assert abs(mask.sum()/stats['eligible_pixels'] - .1) < .002
    assert mask[20, 20]


def test_invalid_pixels_do_not_contaminate_neighbors():
    stokes, radius = scene()
    stokes[20, 20] = np.nan
    stokes[21, 20, 0] = 0
    mask, score, _ = polarization_selection(stokes, radius)
    assert np.isnan(score[20, 20]) and np.isnan(score[21, 20])
    assert not mask[20, 20] and not mask[21, 20]
    assert np.isfinite(score[20, 21])


@pytest.mark.parametrize('kwargs', [{'fraction': 0}, {'fraction': 1},
                                   {'smoothing_sigma': 0}, {'radius_limit': 1.1}])
def test_filter_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        polarization_selection(*scene(), **kwargs)


def test_empty_filter_fails_clearly():
    stokes, radius = scene()
    stokes[:] = np.nan
    with pytest.raises(ValueError, match='no eligible'):
        polarization_selection(stokes, radius)


def test_two_config_sampler_keeps_half_batches_and_ddp_sharding(monkeypatch):
    from spinnme.train.data_loader import TimeStratifiedBatchSampler
    primary = [[(0, j) for j in range(12)], [(1, j) for j in range(12)]]
    secondary = [[(2, 0)], [(3, j) for j in range(2)]]
    def make():
        return TimeStratifiedBatchSampler(primary, 4, additional_dataset_groups=[secondary])
    all_batches = list(make())
    assert len(all_batches) == 6
    for batch in all_batches:
        assert len(batch) == 8
        assert all(i < 2 for i, _ in batch[:4])
        assert all(i >= 2 for i, _ in batch[4:])
        assert sum(i == 2 for i, _ in batch[4:]) == 2
    assert len({index for b in all_batches for index in b[:4]}) == 24
    for rank in range(2):
        monkeypatch.setattr(TimeStratifiedBatchSampler, '_distributed_context', staticmethod(lambda rank=rank: (rank, 2)))
        sampler = make()
        assert len(sampler) == 3
        assert list(sampler) == all_batches[rank::2]


def test_hmi_filter_precedes_shuffle_and_preserves_profiles(tmp_path, monkeypatch):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames
    from sunpy.map import Map, make_fitswcs_header

    import spinnme.loader.spherical as loader
    from spinnme.train.data_loader import TensorsDataset

    stokes, _ = scene()
    observer = SkyCoord(0*u.deg, 7*u.deg, 1*u.AU, frame=frames.HeliographicStonyhurst,
                        obstime='2024-03-23')
    center = SkyCoord(0*u.arcsec, 0*u.arcsec,
                      frame=frames.Helioprojective(observer=observer, obstime=observer.obstime))
    header = make_fitswcs_header(stokes[..., 0, 0], center, scale=[50, 50]*u.arcsec/u.pix)
    header.update({'obs_vr': 0., 'obs_vw': 0., 'obs_vn': 0., 'ccd_x0': 0, 'ccd_y0': 0})
    solar_map = Map(stokes[..., 0, 0], header)
    segment_path = tmp_path / 'segment.fits'
    solar_map.save(segment_path)
    monkeypatch.setattr(loader, 'load_hmi_date_obs_map', lambda _: solar_map)
    monkeypatch.setattr(loader, 'load_stokes_data', lambda *args: stokes.copy())
    monkeypatch.setattr(loader, 'read_hmi_fits_acquisition', lambda _: {'acquisition_key': 'test'})
    monkeypatch.setattr(loader, 'resolve_hmi_transmission_profile', lambda *args: 'response.npz')
    monkeypatch.setattr(loader.HMISphericalDataset, '_validate_quality', lambda *args: None)
    monkeypatch.setattr(loader.HMISphericalDataset, '_validate_segment_alignment', lambda *args: None)
    monkeypatch.setattr(loader.SphericalDataModule, '_hmi_date_obs', lambda _: datetime(2024, 3, 23))
    before_shuffle = []
    real_init = TensorsDataset.__init__
    def capture(self, tensors, **kwargs):
        before_shuffle.append({k: v.copy() for k, v in tensors.items()})
        real_init(self, tensors=tensors, **kwargs)
    monkeypatch.setattr(TensorsDataset, '__init__', capture)
    datasets = []
    for config in [None, {'smoothing_sigma': 5, 'fraction': .1, 'radius_limit': .95}]:
        datasets.append(loader.HMISphericalDataset(
            np.full((4, 6), str(segment_path)), 'calibration', polarization_filter=config,
            ds_id='test', instrument_id='HMI', seconds_per_dt=3600, Rs_per_ds=1,
            ref_time=datetime(2024, 3, 23), stokes_normalization=2,
            work_directory=str(tmp_path), batch_size=8,
        ))
    all_ds, filtered = datasets
    assert before_shuffle[1]['stokes'].shape[0] < before_shuffle[0]['stokes'].shape[0]
    assert before_shuffle[1]['stokes'].shape[0] == filtered.polarization_stats['selected_pixels']
    for ds, tensors in zip(datasets, before_shuffle):
        indices = tensors['pixel_index'][:, 0]
        np.testing.assert_allclose(tensors['stokes'], stokes.reshape(-1, 4, 6)[indices]/2)
        stored_indices = ds.data['pixel_index'][:].reshape(-1)
        np.testing.assert_allclose(ds.data['stokes'][:], stokes.reshape(-1, 4, 6)[stored_indices]/2)
    # Exercise the real combined loader and its short-chunk padding.
    module = loader.SphericalDataModule.__new__(loader.SphericalDataModule)
    module.train_datasets = datasets
    module.train_config_indices = [[0], [1]]
    module.balance_train_configs = True
    module.batch_size, module.dataset_batch_size, module.num_workers = 16, 4, 0
    monkeypatch.setattr(loader, 'sample_hmi_spectral_response', lambda *args: {})
    train_loader = module.train_dataloader()
    for batch in train_loader:
        assert batch['HMI']['stokes'].shape == (16, 4, 6)
    assert all_ds.data['stokes'].shape[0] > filtered.data['stokes'].shape[0]


def test_public_hmi_definition_defaults_to_uniform_sampling():
    from spinnme.instruments import load_instrument_definition
    assert load_instrument_definition('hmi')['polarization_oversampling'] is False
