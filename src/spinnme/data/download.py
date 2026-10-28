"""Unified explicit download dispatch used by the public CLI."""
from datetime import UTC


def _dispatch(source, *, output, start=None, end=None, email=None, product=None,
             telescope=None, cadence="720s", series=None, segments=None,
             time_scale="utc", level="L2", urls=None):
    """Download HMI, PHI or calibrated Hinode data; importing this module is offline."""
    source = source.lower()
    if urls:
        import shutil
        from pathlib import Path
        from urllib.parse import unquote, urlparse
        from urllib.request import urlopen
        from tqdm.auto import tqdm
        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=True)
        destinations = []
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                raise ValueError('Download URLs must use http or https.')
            name = Path(unquote(parsed.path)).name
            if not name or name in ('.', '..'):
                raise ValueError('Download URLs must identify named files.')
            destinations.append((url, directory/name))
        if len({target.name for _,target in destinations}) != len(destinations):
            raise ValueError('Download URLs must have distinct filenames to avoid overwriting observations.')
        paths = []
        for url, target in tqdm(destinations, desc="Downloading", unit="file"):
            temporary = target.with_suffix(target.suffix + '.partial')
            try:
                with urlopen(url, timeout=60) as incoming, temporary.open('wb') as outgoing:
                    size = getattr(incoming, 'headers', {}).get('Content-Length')
                    with tqdm.wrapattr(outgoing, 'write', total=int(size) if size else None,
                                       desc=target.name, leave=False) as progress:
                        shutil.copyfileobj(incoming, progress)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            paths.append(str(target))
        return paths
    if source == 'hmi':
        if not email or start is None:
            raise ValueError('HMI downloads require email and start.')
        from .download_hmi import download_series
        return download_series(output, email, start, end, cadence=cadence,
                               series=series or product or 'S_720s',
                               segments=segments, time_scale=time_scale)
    if source in ('phi', 'phi-hrt', 'phi-fdt'):
        if start is None or end is None:
            raise ValueError('PHI downloads require start and end.')
        from .download_phi import download_phi
        from .fits import parse_time
        start = parse_time(start, time_scale).isoformat()
        end = parse_time(end, time_scale).isoformat()
        telescope = telescope or (source.split('-')[-1] if '-' in source else 'hrt')
        return download_phi(output, start, end, telescope, products=product or ('stokes',), level=level)
    if source == 'hinode':
        if start is None or end is None:
            raise ValueError('Hinode downloads require start and end, or explicit URLs.')
        from .download_hinode import download_hinode
        return download_hinode(output, start, end, time_scale=time_scale)
    raise ValueError('Supported download sources are hmi, phi, phi-hrt, phi-fdt and hinode.')


def download(source, *, output, start=None, end=None, email=None, product=None,
             telescope=None, cadence="720s", series=None, segments=None,
             time_scale="utc", level="L2", urls=None):
    """Download into a checksum-verified collection and return its manifest.

    Identical requests reuse a complete, verified collection. Scientific FITS
    headers retain archive product/release and time metadata in the manifest.
    Archive clients handle their own retry/export lifecycle; URL transfers use
    atomic partial files.
    """
    import hashlib
    import json
    from datetime import datetime
    from pathlib import Path

    from astropy.io import fits
    directory = Path(output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / 'download-manifest.json'
    request = {"source": source, "start": str(start) if start is not None else None,
                   "end": str(end) if end is not None else None, "product": product,
                   "telescope": telescope, "cadence": cadence, "series": series, "segments": segments,
                   "time_scale": time_scale, "level": level, "urls": list(urls or [])}
    request = json.loads(json.dumps(request, default=str))
    # The email address is operational JSOC authentication metadata, not a
    # scientific input and does not need to be distributed with the collection.
    def digest(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                value.update(block)
        return value.hexdigest()
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        records = previous.get('files', [])
        if previous.get('request') == request and records and all(
                (directory/record['path']).is_file() and digest(directory/record['path']) == record['sha256']
                for record in records):
            return manifest_path
        if previous.get('request') != request:
            raise ValueError('This output directory contains a different download collection. '
                             'Use a new directory to avoid mixing scientific inputs.')
        raise ValueError('The existing download collection failed checksum verification. '
                         'Use a clean directory to download a verified replacement.')
    if any(directory.iterdir()):
        raise ValueError('Download output must be empty or contain a matching verified manifest. '
                         'Choose a new directory instead of mixing existing scientific inputs.')
    result = _dispatch(source, output=directory, start=start, end=end, email=email,
                       product=product, telescope=telescope, cadence=cadence, series=series,
                       segments=segments, time_scale=time_scale, level=level, urls=urls)
    if isinstance(result, (list, tuple)):
        paths = [Path(path) for path in result]
    elif hasattr(result, 'columns') and 'download' in result.columns:
        paths = [Path(path) for path in result['download'] if isinstance(path, str)]
    else:
        paths = sorted(directory.glob('*.fits*'))
    if not paths:
        raise RuntimeError('The archive returned no files; no successful download manifest was written.')
    records = []
    keys = ('INSTRUME','TELESCOP','LEVEL','VERSION','VERS_SW','VERS_CAL','PHIDATID',
            'PHIDTYPE','T_REC','T_OBS','HCAMID','QUALITY','DATE-OBS','DATE-AVG','DATE-END',
            'DATE_EAR','DATE_SUN','WAVELNTH','WAVEUNIT','ACCRAWIT')
    for path in paths:
        if not path.is_absolute():
            # Archive clients return either basenames relative to the requested
            # directory or paths relative to the caller's working directory.
            # Prefer the latter only when it already names a downloaded file
            # inside this collection; do not prepend output twice.
            from_cwd = path.resolve()
            path = from_cwd if from_cwd.is_relative_to(directory) and from_cwd.is_file() else directory/path
        path = path.resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f'Archive client reported a file outside the download collection: {path}')
        if not path.is_file():
            raise FileNotFoundError(f'Archive client reported a missing download: {path}')
        metadata = {}
        if '.fits' in path.name.lower():
            with fits.open(path, memmap=False) as hdus:
                for hdu in hdus:
                    metadata.update({key:hdu.header[key] for key in keys if key in hdu.header})
        records.append({"path": str(path.relative_to(directory)), "bytes": path.stat().st_size,
                            "sha256": digest(path), "metadata": metadata})
    if not records:
        raise ValueError('The archive returned no downloaded files.')
    manifest = {"schema_version": 1, "created": datetime.now(UTC).isoformat(),
                    "request": request, "files": records}
    temporary = manifest_path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(manifest, indent=2, default=str)+'\n')
    temporary.replace(manifest_path)
    return manifest_path
