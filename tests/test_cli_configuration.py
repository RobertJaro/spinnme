import pytest

from spinnme.cli import main
from spinnme.configuration import list_configs, resolve_config


def test_packaged_templates_and_overrides():
    assert {"hmi/full_disk", "hinode/sp", "phi/hrt", "phi/fdt", "generic/stokes_cube"} <= set(list_configs())
    configuration = resolve_config("generic/stokes_cube", input_path="observation.fits", output="runs/test",
                                   overrides=["training.max_steps=2", "data.num_workers=0", "model.dim=16"])
    assert configuration["training"]["max_steps"] == 2
    assert configuration["data"]["train_configs"][0]["data_path"] == "observation.fits"
    assert configuration["work_path"] == "runs/test/cache"
    with pytest.raises(ValueError, match="Unknown"):
        resolve_config(configuration, overrides=["training.max_stpes=4"])
    with pytest.raises(ValueError, match='Unknown dataset'):
        resolve_config(configuration, overrides=['data.train_configs.0.polarizaton_filter={}'])
    with pytest.raises(ValueError, match="finite"):
        resolve_config(configuration, overrides=["training.max_steps=-1", "training.epochs=-1"])


def test_cli_help_and_version(capsys):
    with pytest.raises(SystemExit) as status:
        main(["--help"])
    assert status.value.code == 0
    assert "export" in capsys.readouterr().out
    assert main(["config", "list"]) == 0
    assert "hinode/sp" in capsys.readouterr().out


def test_replay_is_rejected():
    configuration = resolve_config("generic/stokes_cube", input_path="test.fits", output="runs/test")
    configuration["training"]["replay_config"] = {}
    with pytest.raises(ValueError, match="Replay"):
        resolve_config(configuration)


def test_native_yaml_timestamp_and_bundled_instrument(tmp_path):
    from datetime import datetime
    configuration = resolve_config('generic/stokes_cube', input_path='test.fits', output='runs/test', instrument='hinode')
    assert configuration['instrument'][0]['line_id'] == 'fe-i-6302'
    assert configuration['data']['train_configs'][0]['instrument_id'] == 'hinode'
    configuration['data']['ref_time'] = datetime(2024, 3, 23)
    assert resolve_config(configuration)['data']['ref_time'].year == 2024
