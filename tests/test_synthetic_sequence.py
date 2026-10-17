import copy
import csv
import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from astropy.io import fits

from spinnme.cli import main
from spinnme.data.synthetic_sequence import _canonical_hash as canonical_hash
from spinnme.data.synthetic_sequence import _checksum as checksum
from spinnme.data.synthetic_sequence import expand_acquisition_grid, generate_dataset


def _spec():
    return {"generation": {"shape": [8, 10], "pattern_resolution": [16, 16],
                           "base_step_seconds": 5, "duration_seconds": 20,
                           "evolution_seconds_per_step": 90,
                           "oscillation_period_seconds": 180,
                           "oscillation_amplitude_steps": 30},
            "acquisition": {"variants": [{"name": "ten-second", "cadence_seconds": 10,
                                          "exposure_seconds": 10, "noise_std": [0]*4,
                                          "realization_seeds": [13]}]}}


@pytest.fixture
def fake_generator(monkeypatch):
    """Exercise acquisition statistics with cheap deterministic ME stand-in arrays."""
    pytest.importorskip("spinnme")
    from spinnme.data import create_spherical_test_set

    class Generator:
        calls: ClassVar[list] = []

        def __init__(self, nx, ny, wavelength_grid, **kwargs):
            self.shape = (nx, ny)
            self.wavelengths = len(wavelength_grid)

        def load_spherical_time_step(self, time_step, observer):
            self.calls.append((float(time_step), observer.obstime.utc.isot))
            plane = np.full(self.shape, time_step, dtype=np.float32)
            parameters = {name: np.ones(self.shape, dtype=np.float32) for name in
                          ("vmac", "damping", "b0", "b1", "kl", "mu", "cos_inc", "sin_inc2", "inc", "azi")}
            parameters.update(b_field=plane, vdop=plane,
                              b_rtp=np.stack([plane, plane*2, plane*3], -1),
                              v_rtp=np.stack([plane, plane*2, plane*3], -1))
            profiles = np.broadcast_to(plane[..., None, None], (*self.shape, 4, self.wavelengths)).copy()
            return profiles, parameters, None

    monkeypatch.setattr(create_spherical_test_set, "SphericalTestSetGenerator", Generator)
    return Generator


def test_real_circular_truth_and_exposures_are_generic_fits_readable(tmp_path):
    pytest.importorskip("spinnme")
    from spinnme.data.fits import read_fits_observation

    source = _spec()
    manifest_path = generate_dataset(source, tmp_path/"dataset")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["spec"] == source
    assert manifest["specification_sha256"] == canonical_hash(source)
    dense = manifest["dense_truth"]["frames"]
    assert [row["time_seconds"] for row in dense] == [2.5, 5, 7.5, 12.5, 15, 17.5]
    assert sum(row["base_grid"] for row in dense) == 4
    frames = manifest["acquisitions"][0]["realizations"][0]["frames"]
    frame = frames[0]
    path = manifest_path.parent/frame["path"]
    observation = read_fits_observation(path, axis_order=["stokes", "wavelength", "y", "x"],
                                        line_id="fe-i-6173")
    assert observation["stokes"].shape == (8, 10, 4, 6)
    assert np.isfinite(observation["stokes"]).any()
    assert observation["model_time"].isoformat() == "2025-01-01T00:00:05"
    header = fits.getheader(path)
    assert header["EXPTIME"] == 10
    assert header["DATE-BEG"] == "2025-01-01T00:00:00.000"
    assert header["DATE-END"] == "2025-01-01T00:00:10.000"
    assert header["DSUN_OBS"] > 1e11
    assert header["HGLN_OBS"] == pytest.approx(0)
    assert "CRLN_OBS" in header
    with fits.open(path) as hdus:
        assert hdus["VALID_MASK"].data.shape == (8, 10)
        assert hdus["VALID_MASK"].data.any()
        assert hdus["WAVELENGTH"].data.shape == (6,)
    latent = [fits.getdata(manifest_path.parent/dense[i]["reference_fits"])
              for i in frame["latent_indices"]]
    expected = np.mean(np.stack(latent).astype(np.float64), axis=0).astype(np.float32)
    np.testing.assert_array_equal(fits.getdata(path), expected)
    np.testing.assert_array_equal(fits.getdata(path), fits.getdata(manifest_path.parent/frame["truth_stokes_path"]))
    truth_row = next(row for row in dense if row["time_seconds"] == 5)
    assert checksum(manifest_path.parent/truth_row["truth_npz"]) == truth_row["truth_sha256"]
    with np.load(manifest_path.parent/truth_row["truth_npz"], allow_pickle=False) as truth:
        metadata = json.loads(truth["metadata"].item())
        assert metadata["units"]["B_los"] == "G"
        assert metadata["units"]["v_los"] == "m / s"
        assert truth["I"].shape == (8, 10, 6)
        assert truth["B"].shape == (8, 10)
        assert metadata["wavelength_center_m"] == pytest.approx(6173.3433e-10)
        assert truth_row["evolution_time_step"] == pytest.approx(5/90+30*np.sin(2*np.pi*5/180))
    with pytest.raises(FileExistsError, match="overwrite"):
        generate_dataset(source, manifest_path.parent)


def test_seeded_noise_is_paired_order_independent_and_has_correct_sigma(tmp_path, fake_generator):
    spec = _spec()
    spec["generation"].update(shape=[64, 80], duration_seconds=10)
    spec["acquisition"]["continuum_reference"] = 2
    first = spec["acquisition"]["variants"][0]
    first.update(name="low", noise_std=[0.001, 0.002, 0.003, 0.004], realization_seeds=[13, 14])
    second = {**copy.deepcopy(first), "name": "high", "noise_std": [0.002, 0.004, 0.006, 0.008]}
    spec["acquisition"]["variants"].append(second)
    original = copy.deepcopy(spec)
    path = generate_dataset(spec, tmp_path/"a")
    assert spec == original
    manifest = json.loads(path.read_text())
    assert len(fake_generator.calls) == 3  # two integration bins plus one exact midpoint
    products = []
    for acq in manifest["acquisitions"]:
        records = []
        for realization in acq["realizations"]:
            frame = realization["frames"][0]
            clean = fits.getdata(path.parent/frame["truth_stokes_path"])
            records.append(fits.getdata(path.parent/frame["path"]) - clean)
        products.append(records)
    low, high = products[0][0], products[1][0]
    np.testing.assert_allclose(high, low*2, atol=2e-6)
    for i, std in enumerate(first["noise_std"]):
        assert low[i].std() == pytest.approx(std*2, rel=0.03)
        assert abs(low[i].mean()) < std*2*0.03
    assert not np.array_equal(products[0][0], products[0][1])
    spec["acquisition"]["variants"].reverse()
    reordered = generate_dataset(spec, tmp_path/"b")
    other = json.loads(reordered.read_text())
    first_a = manifest["acquisitions"][0]["realizations"][0]["frames"][0]["path"]
    first_b = other["acquisitions"][1]["realizations"][0]["frames"][0]["path"]
    np.testing.assert_array_equal(fits.getdata(path.parent/first_a), fits.getdata(reordered.parent/first_b))


def test_cadence_changes_sample_times_without_implied_snr_and_exposure_scaling_is_explicit(tmp_path, fake_generator):
    spec = _spec()
    first = spec["acquisition"]["variants"][0]
    first.update(exposure_seconds=5, noise_std=[0.001]*4)
    spec["acquisition"]["variants"] += [
        {**first, "name": "slow", "cadence_seconds": 20},
        {**first, "name": "long-exposure", "exposure_seconds": 10,
         "noise_reference_exposure_seconds": 5},
    ]
    path = generate_dataset(spec, tmp_path/"data")
    manifest = json.loads(path.read_text())
    a, b, c = manifest["acquisitions"]
    assert a["effective_noise_std"] == b["effective_noise_std"] == [0.001]*4
    assert c["effective_noise_std"] == pytest.approx([0.001/np.sqrt(2)]*4)
    assert len(a["realizations"][0]["frames"]) == 2
    assert len(b["realizations"][0]["frames"]) == 1


def test_physical_evolution_uses_seconds_independently_of_number_of_frames(tmp_path, fake_generator):
    spec = _spec()
    path = generate_dataset(spec, tmp_path/"five")
    spec["generation"]["base_step_seconds"] = 2.5
    second = generate_dataset(spec, tmp_path/"two")
    records = []
    for manifest_path in (path, second):
        manifest = json.loads(manifest_path.read_text())
        records.append(next(row for row in manifest["dense_truth"]["frames"] if row["time_seconds"] == 5))
    assert records[0]["evolution_time_step"] == records[1]["evolution_time_step"]
    with np.load(path.parent/records[0]["truth_npz"]) as a, np.load(second.parent/records[1]["truth_npz"]) as b:
        np.testing.assert_array_equal(a["B"], b["B"])


@pytest.mark.parametrize("section,key,value,match", [
    ("generation", "base_step_seconds", 0, "base_step"),
    ("generation", "duration_seconds", 11, "integer multiple"),
    ("generation", "evolution_seconds_per_step", 0, "evolution"),
    ("generation", "oscillation_amplitude_steps", 300, "radius"),
    ("generation", "oscillation_period_seconds", 10, "four samples"),
    ("generation", "shape", [4, 2], "dimensions"),
    ("variant", "cadence_seconds", 7, "integer multiple"),
    ("variant", "exposure_seconds", 20, "exceed cadence"),
    ("variant", "phase_seconds", 20, "no complete"),
    ("variant", "name", "../bad", "path-safe"),
    ("variant", "noise_std", [0, -1, 0, 0], "noise_std"),
    ("variant", "noise_model", "poisson", "noise_model"),
    ("variant", "continuum_reference", 2, "global acquisition"),
    ("variant", "realization_seeds", [1, 1], "unique"),
])
def test_invalid_acquisition_contract_rejected_before_output_creation(tmp_path, section, key, value, match):
    spec = _spec()
    target = spec["generation"] if section == "generation" else spec["acquisition"]["variants"][0]
    target[key] = value
    with pytest.raises(ValueError, match=match):
        generate_dataset(spec, tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()


def test_grid_expansion_and_shell_table_cover_realizations(tmp_path, fake_generator):
    specification = _spec()
    specification['acquisition'] = {
        'cadences_seconds': [5, 10], 'exposure_seconds': 5, 'phases_seconds': [0],
        'noise_levels': [{'name': 'clean', 'std': 0}, {'name': 'noisy', 'std': [0, .001, .002, .003]}],
        'realization_seeds': [13, 14]}
    unchanged = copy.deepcopy(specification)
    expanded = expand_acquisition_grid(specification)
    assert len(expanded['acquisition']['variants']) == 4
    assert sum(len(row['realization_seeds']) for row in expanded['acquisition']['variants']) == 6
    assert specification == unchanged
    path = generate_dataset(specification, tmp_path/'data with spaces [one] "quote"')
    manifest = json.loads(path.read_text())
    with (path.parent/'acquisitions.tsv').open(newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        assert reader.fieldnames == ['RUN_ID', 'CADENCE_SECONDS', 'NOISE_LABEL', 'NOISE_SEED', 'INPUT_GLOB']
        rows = list(reader)
    assert len(rows) == 6
    assert len({row['RUN_ID'] for row in rows}) == 6
    expected = []
    for acquisition in manifest['acquisitions']:
        for realization in acquisition['realizations']:
            expected.append(f"{acquisition['name']}-noise{realization['seed']}")
    assert [row['RUN_ID'] for row in rows] == expected
    import glob
    for row in rows:
        assert Path(row['INPUT_GLOB']).is_absolute()
        assert len(glob.glob(row['INPUT_GLOB'])) == (4 if row['CADENCE_SECONDS'] == '5' else 2)
    assert manifest['spec'] == specification


@pytest.mark.parametrize('acquisition', [None, {'cadences_seconds': 5},
    {'cadences_seconds': [True], 'noise_levels': [{'name': 'n', 'std': .001}]},
    {'cadences_seconds': [5, 5], 'noise_levels': [{'name': 'n', 'std': .001}]},
    {'cadences_seconds': [5], 'noise_levels': [{}]}, {'variants': 'invalid'}])
def test_malformed_grid_is_rejected_before_output_creation(tmp_path, acquisition):
    specification = _spec()
    specification['acquisition'] = acquisition
    with pytest.raises(ValueError):
        generate_dataset(specification, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_training_planner_settings_do_not_belong_in_generation_config(tmp_path):
    specification = _spec()
    specification['common'] = {'training': {'max_steps': 100}}
    with pytest.raises(ValueError, match='separate inversion YAML'):
        generate_dataset(specification, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_cli_generation_yaml_and_original_simple_mode(tmp_path, fake_generator, monkeypatch):
    import yaml
    specification = tmp_path/'dataset.yaml'
    specification.write_text(yaml.safe_dump(_spec()))
    assert main(['synthetic', '--config', str(specification), '--output', str(tmp_path/'data')]) == 0
    assert (tmp_path/'data/acquisitions.tsv').is_file()
    from spinnme.data import circular
    calls = []
    monkeypatch.setattr(circular, 'create_circular_dataset', lambda output, **kwargs: calls.append(kwargs))
    assert main(['synthetic', '--output', str(tmp_path/'simple')]) == 0
    assert calls.pop() == {'size': 32, 'frames': 2}
    assert main(['synthetic', '--output', str(tmp_path/'simple'), '--size', '8', '--frames', '1']) == 0
    assert calls.pop() == {'size': 8, 'frames': 1}
    with pytest.raises(SystemExit) as result:
        main(['synthetic', '--config', str(specification), '--output', str(tmp_path/'bad'), '--size', '8'])
    assert result.value.code == 2
    assert not (tmp_path/'bad').exists()
