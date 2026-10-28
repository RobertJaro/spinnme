"""Execute every notebook cell with offline inputs and a two-step CPU budget.

Colab installation/upload/download boundaries, trial sizes and compilation are replaced.
Preparation, both Hinode lines, training, validation, snapshots, exports, plots,
residual checks and ZIP assembly execute the notebook's own code.
"""
import json
import shutil
import sys
import types
from pathlib import Path
from zipfile import ZipFile

import matplotlib
import numpy as np
import pytest
from astropy.io import fits
from test_fits_cropping import acquisition
from test_fits_observations import header

from spinnme.cli import main
from spinnme.data.hmi_transmission import MANIFEST_FORMAT, hmi_acquisition_key

matplotlib.use('Agg')
NOTEBOOKS = Path(__file__).resolve().parents[1] / 'examples' / 'colab'


def offline_input(kind, directory):
    directory.mkdir()
    if kind == 'hmi':
        return acquisition(directory)
    if kind == 'hinode_sp':
        waves = np.linspace(6300.9, 6303.1, 33)
        for column, index in enumerate((0, 2, 3, 5)):
            slit_header = fits.Header({'CRVAL1': waves[0], 'CRPIX1': 1.,
                                      'CDELT1': waves[1] - waves[0], 'CUNIT1': 'Angstrom',
                                      'DATE_OBS': f'2024-03-23T02:00:{35+column*4:02d}',
                                      'SLITINDX': index, 'XCEN': -100+index*.3, 'YCEN': 20.,
                                      'XSCALE': .3, 'YSCALE': .32, 'CROTA2': .1,
                                      'FGSPROT': .2636, 'SPCCDIX0': 510, 'SPCCDIX1': 513, 'CAMSSUM': 1})
            stokes = np.zeros((4, 4, len(waves)), dtype='float32')
            stokes[0] = 1 - .3 * np.exp(-((waves - 6302.4931) / .1) ** 2)
            fits.writeto(directory/f'slit_{index:03d}.fits', stokes, slit_header)
        return directory
    h = header((4, 4))
    offsets = [-.3, -.14, -.07, 0., .07, .3]
    for i, offset in enumerate(offsets, 1):
        h[f'WAVELN{i:02d}'] = h['WAVELNTH'] + offset
    h['CONTPOS'] = 6
    stokes = np.zeros((6, 4, 4, 4), dtype='float32')
    stokes[:, 0] = (1 - .3 * np.exp(-(np.array(offsets) / .1) ** 2))[:, None, None]
    path = directory / 'stokes.fits'
    fits.writeto(path, stokes, h)
    return path


@pytest.mark.integration
@pytest.mark.parametrize('name', ['circular', 'custom_fits', 'hinode_sp', 'hinode_sp_upload', 'phi', 'phi_fdt', 'hmi'])
def test_colab_end_to_end(name, tmp_path, monkeypatch):
    upload_hinode = name == 'hinode_sp_upload'
    name = 'hinode_sp' if upload_hinode else name
    telescope = 'fdt' if name == 'phi_fdt' else 'hrt'
    name = 'phi' if name == 'phi_fdt' else name
    import matplotlib.pyplot as plt
    import torch

    # Small CPU tensors are much faster without a large BLAS thread pool.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(plt, 'show', lambda: plt.close('all'))
    source = None if name == 'circular' else offline_input(name, tmp_path / 'inputs')
    downloads = []
    files = types.ModuleType('google.colab.files')
    files.upload = lambda: ({p.name: p.read_bytes() for p in source.glob('*.fits')}
                            if source.is_dir() else {source.name: source.read_bytes()})
    files.download = downloads.append
    colab = types.ModuleType('google.colab')
    colab.files = files
    monkeypatch.setitem(sys.modules, 'google.colab', colab)
    monkeypatch.setitem(sys.modules, 'google.colab.files', files)

    def command(*arguments):
        arguments = list(map(str, arguments))
        if arguments[:2] == ['download', 'hmi']:
            shutil.copytree(source, namespace['RAW'])
        elif arguments[:2] == ['download', 'hinode']:
            shutil.copytree(source, namespace['RAW'])
        elif arguments[:2] == ['download', 'phi']:
            namespace['RAW'].mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, namespace['RAW'] / source.name)
        elif arguments[:2] == ['download', 'hmi-transmission']:
            directory = namespace['DATA'] / 'transmission'
            directory.mkdir()
            np.savez(directory / 'profile.npz', offsets=np.zeros((6, 1)),
                     weights=np.ones((1, 1, 6, 1)), continuum_weights=np.zeros((1, 1, 6)))
            (directory / 'manifest.json').write_text(json.dumps({
                'format': MANIFEST_FORMAT, 'profiles': {'profile': {'file': 'profile.npz'}},
                'acquisitions': {hmi_acquisition_key('2024.03.23_00:00:37_TAI', 2): {'profile': 'profile'}},
            }))
        else:
            assert main(arguments) == 0

    notebook = json.loads((NOTEBOOKS / f'{name}.ipynb').read_text())
    namespace = {'__name__': '__main__'}
    try:
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            code = ''.join(cell['source'])
            if '%pip' in code:
                continue  # Use the existing checkout/environment, never PyPI.
            if 'ROOT = Path(' in code:
                code = code.replace(f'/content/spinnme/{"custom" if name == "custom_fits" else "hinode" if name == "hinode_sp" else name}', str(tmp_path / 'workspace'))
            if upload_hinode:
                code = code.replace('DOWNLOAD_FROM_ARCHIVE = True', 'DOWNLOAD_FROM_ARCHIVE = False')
            code = code.replace('TELESCOPE = "hrt"', f'TELESCOPE = "{telescope}"')
            for steps in (200000, 1000, 200):
                code = code.replace(f'MAX_STEPS = {steps}\n', 'MAX_STEPS = 2\n')
            code = code.replace('shape=(32, 32)', 'shape=(8, 8)')
            code = code.replace('JSOC_EMAIL = ""', 'JSOC_EMAIL = "offline@example.org"')
            code = code.replace('PIXELS = [1984, 1984, 2112, 2112]', 'PIXELS = [2, 3, 9, 8]')
            if 'from spinnme import run' in code:
                namespace['config']['training'].update(accelerator='cpu', devices=1,
                    enable_progress_bar=False, enable_model_summary=False)
                if name == 'hinode_sp':
                    # Exercise the full workflow with a small CPU model and eager synthesis.
                    namespace['config']['model'].update(dim=16, n_layers=2)
                    namespace['config']['data'].update(batch_size=32, dataset_batch_size=16,
                                                       num_dataset_workers=0)
                    namespace['config']['training']['compile_config']['me_solver'] = False
            # Execute only repository-owned notebook cells to test the published workflow.
            exec(compile(code, f'{name}.ipynb:cell {index}', 'exec'), namespace)  # noqa: S102
            if 'ROOT = Path(' in code:
                namespace['spinnme'] = command
        if name == 'hinode_sp':
            assert len(list(namespace['RAW'].iterdir())) == 4
            assert all(p.suffix == '.fits' for p in namespace['RAW'].iterdir())
            assert namespace['data_module'].image_shape == (4, 4)
        run = namespace['RUN']
        assert namespace['trainer'].global_step == 2
        manifest = json.loads((run / 'run_manifest.json').read_text())
        assert manifest['status'] == 'complete'
        residuals = json.loads((run / 'products/stokes-rms.json').read_text())
        assert len(residuals) == (2 if name in ('circular', 'hinode_sp') else 1)
        assert all(np.isfinite([row[key] for key in 'IQUV']).all() for row in residuals)
        assert len(downloads) == 1
        with ZipFile(downloads[0]) as archive:
            names = archive.namelist()
            assert {'inversion.pme', 'last.ckpt', 'resolved.yaml', 'run_manifest.json'} <= set(names)
            assert any(n.endswith('.fits') for n in names)
            assert any(n.endswith('.jpg') for n in names)
            assert any(n.endswith('parameters.npz') for n in names)
            assert any(n.endswith('stokes-fit.png') for n in names)
            assert 'run.yaml' not in names
    finally:
        torch.set_num_threads(threads)
        plt.close('all')
