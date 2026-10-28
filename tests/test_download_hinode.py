"""Generic time-based and explicit-URL Hinode downloads."""
import numpy as np
from astropy.io import fits



def slit(tmp_path):
    raw = np.arange(4*8*6, dtype=np.float32).reshape(4, 8, 6)
    header = fits.Header({'CRVAL1': 6302., 'CRPIX1': 2., 'CDELT1': -.02,
                          'CUNIT1': 'Angstrom', 'XSCALE': .3, 'SLITINDX': 17,
                          'DATE_OBS': '2024-03-23T02:00:35.666'})
    source = tmp_path/'slit.fits'
    fits.writeto(source, raw, header)
    return source


def test_cli_downloads_stokes_by_time_without_reference(tmp_path, monkeypatch):
    import io
    import json
    from spinnme.cli import main
    from spinnme.data.download_hinode import ARCHIVE

    source = slit(tmp_path)
    names = [f'SP3D20240323_0200{second}.6C.fits' for second in (35, 36, 37)]
    day = ARCHIVE+'2024/03/23/SP3D/'
    raster = day+'20240323_020035/'
    pages = {day: b'<a href="20240323_020035/">scan</a>',
             raster: ''.join(f'<a href="{name}">slit</a>' for name in names).encode()}
    requests = []

    def urlopen(url, **kwargs):
        requests.append(url)
        if url.endswith('.fits'):
            return io.BytesIO(source.read_bytes())
        return io.BytesIO(pages.get(url, b''))

    monkeypatch.setattr('spinnme.data.download_hinode.urlopen', urlopen)
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    output = tmp_path/'download'
    command = ['download', 'hinode', '--output', str(output),
               '--start', '2024-03-23T02:01:12', '--end', '2024-03-23T02:01:14', '--time-scale', 'tai']
    assert main(command) == 0
    report = json.loads((output/'download-manifest.json').read_text())
    assert [row['path'] for row in report['files']] == names[:2]
    assert set(output.glob('*.fits')) == {output/name for name in names[:2]}
    assert all(row['sha256'] for row in report['files'])
    assert raster+names[2] not in requests
    count = len(requests)
    assert main(command) == 0
    assert len(requests) == count  # Existing verified files are reused.


def test_cli_preserves_explicit_url_downloads(tmp_path, monkeypatch):
    from spinnme.cli import main
    calls = []
    monkeypatch.setattr('spinnme.data.download.download', lambda *a, **k: calls.append((a, k)))
    urls = ['https://example.org/a.fits', 'https://example.org/b.fits']
    assert main(['download', 'hinode', '--output', str(tmp_path), '--url', urls[0], '--url', urls[1]]) == 0
    assert calls[0][0] == ('hinode',)
    assert calls[0][1]['urls'] == urls
