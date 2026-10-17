"""Named YAML overwrites resolve before any training or filesystem mutation."""
from pathlib import Path

import pytest

from spinnme.cli import main
from spinnme.configuration import list_configs, read_config, resolve_config, substitute_placeholders, write_config


def test_bundled_runs_start_with_named_paths():
    for name in list_configs():
        config = read_config(name)
        assert list(config)[:2] == ['path', 'work_path']
        assert config['path'] == '<<run_path>>'
        assert config['work_path'] == '<<work_path>>'


def test_missing_named_paths_fail_before_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as error:
        main(['run', '--config', 'synthetic/circular', '--input', 'test.fits'])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert '--run_path' in message and '--work_path' in message
    assert not list(tmp_path.iterdir())


def test_recursive_embedded_and_typed_overwrites_are_literal():
    config = {'path': '<<root>>/run', 'nested': ['<<count>>', {'path': '<<root>>/data'}]}
    result = substitute_placeholders(config, {'root': '/tmp/with spaces/$(no_shell)', 'count': 3})
    assert result == {'path': '/tmp/with spaces/$(no_shell)/run',
                      'nested': [3, {'path': '/tmp/with spaces/$(no_shell)/data'}]}
    assert config['path'] == '<<root>>/run'


def test_set_overwrites_fill_named_paths_and_typed_model_settings():
    config = read_config('synthetic/circular')
    config['model']['dim'] = '<<width>>'
    resolved = resolve_config(config, input_path='test.fits', overrides=[
        'run_path=/tmp/named run', 'work_path=/tmp/named work', 'width=16'])
    assert resolved['path'] == '/tmp/named run'
    assert resolved['work_path'] == '/tmp/named work'
    assert resolved['model']['dim'] == 16


def test_cli_named_overwrites_and_generic_declared_flags(tmp_path, monkeypatch):
    from spinnme.training import runner
    config = read_config('synthetic/circular')
    config['model']['dim'] = '<<width>>'
    config['data']['train_configs'][0]['data_path'] = '<<stokes_files>>'
    config['data']['valid_config']['data_path'] = '<<stokes_files>>'
    path = write_config(config, tmp_path/'run.yaml')
    captured = []
    monkeypatch.setattr(runner, 'run', lambda config, **kwargs: captured.append(config))
    assert main(['run', '--config', str(path), '--run_path', str(tmp_path/'run'),
                 '--work_path', str(tmp_path/'cache'), '--width', '16',
                 '--stokes_files', '/tmp/with spaces/*.fits']) == 0
    resolved, = captured
    assert resolved['path'] == str(tmp_path/'run')
    assert resolved['work_path'] == str(tmp_path/'cache')
    assert resolved['model']['dim'] == 16
    assert resolved['data']['valid_config']['data_path'] == '/tmp/with spaces/*.fits'
    assert not (tmp_path/'run').exists()


def test_unknown_cli_flag_is_not_a_silent_override(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(['run', '--config', 'synthetic/circular', '--input', 'test.fits',
              '--run_path', str(tmp_path/'run'), '--widht', '16'])
    assert error.value.code == 2
    assert not (tmp_path/'run').exists()


def test_config_copy_keeps_overwrite_headers(tmp_path):
    output = tmp_path/'copied.yaml'
    assert main(['config', 'copy', 'synthetic/circular', '--output', str(output)]) == 0
    assert output.read_text().splitlines()[:2] == ['path: "<<run_path>>"', 'work_path: "<<work_path>>"']


def test_output_alias_supplies_cache_unless_work_path_overwritten():
    resolved = resolve_config('synthetic/circular', input_path='test.fits', output='/tmp/run',
                              overrides=['work_path=/tmp/custom-cache'])
    assert resolved['path'] == '/tmp/run'
    assert resolved['work_path'] == '/tmp/custom-cache'
    assert Path(resolve_config('synthetic/circular', input_path='test.fits', output='/tmp/run')['work_path']) == Path('/tmp/run/cache')


def test_multiline_preset_keeps_distinct_line_inputs():
    with pytest.raises(ValueError, match='line6302_path'):
        resolve_config('generic/multiple_lines', input_path='6173.fits', output='/tmp/run')
    config = resolve_config('generic/multiple_lines', output='/tmp/run',
                            bindings={'line6173_path': '6173.fits', 'line6302_path': '6302.fits'})
    assert [row['data_path'] for row in config['data']['train_configs']] == ['6173.fits', '6302.fits']
    assert config['data']['valid_config']['data_path'] == '6173.fits'
