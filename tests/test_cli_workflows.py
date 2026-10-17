from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from spinnme.cli import main
from spinnme.configuration import resolve_config, write_config


def test_transmission_download_is_independent_of_run_configuration(tmp_path, monkeypatch):
    calls = []
    def download(inputs, output, **kwargs):
        calls.append((inputs, output, kwargs))
        return tmp_path/'transmission/manifest.json'
    def unexpected(*args, **kwargs):
        pytest.fail('Transmission download must not prepare training data or start training.')
    monkeypatch.setattr('spinnme.data.hmi_transmission.prepare_hmi_transmission_directory', download)
    monkeypatch.setattr('spinnme.data.cache.load_or_prepare_data_module', unexpected)
    monkeypatch.setattr('spinnme.training.runner.run', unexpected)
    assert main(['download', 'hmi-transmission', '--input', 'data90', 'data720',
                 '--output', str(tmp_path/'transmission'), '--email', 'user@example.org']) == 0
    assert calls == [(['data90', 'data720'], str(tmp_path/'transmission'),
                      {'email': 'user@example.org', 'overwrite': False})]
    assert not (tmp_path/'transmission').exists()


@pytest.mark.parametrize('arguments', [[], ['--input', 'stokes']])
def test_transmission_download_requires_explicit_inputs_and_email(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        main(['download', 'hmi-transmission', '--output', str(tmp_path/'profiles'), *arguments])
    assert error.value.code == 2
    assert not (tmp_path/'profiles').exists()


def test_optional_prep_only_caches_the_supplied_yaml(tmp_path, monkeypatch):
    config = resolve_config('hmi/full_disk', output=str(tmp_path/'run'),
                             bindings={'data_path': str(tmp_path/'data')})
    source = write_config(config, tmp_path/'inversion.yaml')
    before = source.read_bytes()
    calls = []
    module = SimpleNamespace(get_observations=lambda: [], train_datasets=[object()],
                             cache_reused=False, cache_manifest=tmp_path/'cache.json')
    def prepare(data, work, **kwargs):
        calls.append((data, work, kwargs))
        return module
    def unexpected(*args, **kwargs):
        pytest.fail('Cache-only prep must not download HMI calibration.')
    monkeypatch.setattr('spinnme.data.cache.load_or_prepare_data_module', prepare)
    monkeypatch.setattr('spinnme.data.hmi_transmission.prepare_hmi_transmission_directory', unexpected)
    assert main(['prep', '--config', str(source)]) == 0
    assert calls == [(config['data'], config['work_path'], {'reload': False})]
    assert source.read_bytes() == before
    assert (tmp_path/'run/preparation.json').is_file()
    assert not list((tmp_path/'run').glob('*.yaml'))


def test_hmi_quickstart_run_uses_the_bundled_yaml_without_runtime_training_edits(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr('spinnme.training.runner.run', lambda config, **kwargs: captured.append(config))
    assert main(['run', '--config', 'hmi/full_disk', '--data_path', str(tmp_path/'data'),
                 '--run_path', str(tmp_path/'run')]) == 0
    config, = captured
    for dataset in [*config['data']['train_configs'], config['data']['valid_config']]:
        assert dataset['data_path'] == str(tmp_path/'data/stokes')
        assert dataset['transmission_profile_directory'] == str(tmp_path/'data/transmission')
        assert dataset['resolution'] == [256,256]
    assert config['training']['max_steps'] == 1000
    assert config['training']['lr_params']['iterations'] == 1000
    assert 'accelerator' not in config['training']
    assert 'devices' not in config['training']
    assert not (tmp_path/'run').exists()



def test_resumed_run_refreshes_exports_on_requested_device(tmp_path, monkeypatch):
    import spinnme
    output = tmp_path / "run"
    output.mkdir()
    (output / "inversion.pme").write_bytes(b"snapshot")
    training_calls, export_calls, load_calls = [], [], []
    monkeypatch.setattr("spinnme.training.runner.run", lambda *args, **kwargs: training_calls.append(kwargs))
    result = SimpleNamespace(write=lambda path, **kwargs: export_calls.append((path, kwargs)))
    snapshot = SimpleNamespace(observations=[{"id": "train_00_000"}], evaluate=lambda *args: result)
    def load(path, **kwargs):
        load_calls.append(kwargs)
        return nullcontext(snapshot)
    monkeypatch.setattr(spinnme, "load", load)
    assert main(["run", "--config", "generic/stokes_cube", "--input", "test.fits", "--output", str(output),
                 "--device", "cpu", "--resume", "--export", "fits"]) == 0
    assert training_calls[0]["resume"]
    assert load_calls == [{"device": "cpu"}]
    assert export_calls == [(output / "exports/train_00_000", {"format": "fits", "overwrite": True})]


@pytest.mark.parametrize(('command', 'flags'), [
    ('prep', ['--hmi-transmission', '--email', 'user@example.org']),
    ('prep', ['--hinode-raster']),
    ('run', ['--set', 'training.max_steps=10']),
])
def test_run_and_prep_reject_instrument_actions_and_training_mutations(tmp_path, command, flags):
    with pytest.raises(SystemExit) as error:
        main([command, '--config', 'synthetic/circular', '--input', 'test.fits',
              '--run_path', str(tmp_path/'run'), *flags])
    assert error.value.code == 2
    assert not (tmp_path/'run').exists()



def test_config_copy_requires_explicit_overwrite(tmp_path):
    path = tmp_path / "custom.yaml"
    assert main(["config", "copy", "generic/stokes_cube", "--output", str(path)]) == 0
    path.write_text("user: edits\n")
    with pytest.raises(SystemExit) as error:
        main(["config", "copy", "generic/stokes_cube", "--output", str(path)])
    assert error.value.code == 2
    assert path.read_text() == "user: edits\n"
    assert main(["config", "copy", "generic/stokes_cube", "--output", str(path), "--overwrite"]) == 0
    assert "train_configs" in path.read_text()


def test_cli_hmi_download_forwards_requested_segments(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("spinnme.data.download.download", lambda *args, **kwargs: calls.append(kwargs))
    assert main(["download", "hmi", "--product", "B_90s", "--segments", "field", "inclination",
                 "--start", "2024-03-23T01:00:00", "--email", "user@example.org", "--output", str(tmp_path)]) == 0
    assert calls[0]["segments"] == ["field", "inclination"]
    assert calls[0]["cadence"] == "90s"


def test_cli_reports_runtime_failures_without_traceback(tmp_path, monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise RuntimeError("Device is unavailable")
    monkeypatch.setattr("spinnme.training.runner.run", unavailable)
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", "generic/stokes_cube", "--input", "test.fits", "--output", str(tmp_path)])
    assert error.value.code == 2
    assert "Device is unavailable" in capsys.readouterr().err


@pytest.mark.parametrize('failed', [False, True])
def test_phi_download_rejects_empty_and_partial_transfers(tmp_path, monkeypatch, failed):
    import sys
    import types
    from sunpy.net import Fido, attrs
    from spinnme.data.download_phi import download_phi

    class Fetched(list):
        errors = ['failed transfer'] if failed else []
    monkeypatch.setitem(sys.modules, 'sunpy_soar', types.ModuleType('sunpy_soar'))
    monkeypatch.setattr(attrs, 'soar', types.SimpleNamespace(Product=lambda name: name), raising=False)
    monkeypatch.setattr(Fido, 'search', lambda *args: [object()])
    monkeypatch.setattr(Fido, 'fetch', lambda *args, **kwargs: Fetched())
    with pytest.raises(RuntimeError, match='Incomplete' if failed else 'No files were downloaded'):
        download_phi(tmp_path, '2024-03-23', '2024-03-24')
