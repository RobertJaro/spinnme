"""Explicit Solar Orbiter Archive downloads; importing performs no network I/O."""
from pathlib import Path


def download_phi(output_directory, start, end, telescope="hrt", products=("stokes",),
                 level="L2"):
    """Fetch PHI products through SunPy-SOAR and return downloaded paths.

    Archive availability and versions are determined by the returned records;
    the caller should retain a manifest rather than guessing filenames.
    """
    if telescope not in ("hrt", "fdt"):
        raise ValueError("telescope must be hrt or fdt.")
    allowed = {"stokes", "bmag", "binc", "bazi", "blos", "vlos", "icnt"}
    products = [products] if isinstance(products, str) else list(products)
    if not products or not set(products) <= allowed:
        raise ValueError(f"Choose PHI products from {sorted(allowed)}.")
    try:
        import sunpy_soar  # noqa: F401 - registers the archive client
        from sunpy.net import Fido, attrs as a
    except ImportError as error:
        raise ImportError("PHI downloads require the spinnme[download] extra.") from error
    from astropy.time import Time
    begin, finish = Time(start), Time(end)
    if finish <= begin:
        raise ValueError("end must be later than start.")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for product in products:
        result = Fido.search(a.Instrument("PHI"), a.Time(begin, finish),
                             a.soar.Product(f"phi-{telescope}-{product}"), a.Level(level))
        if len(result) == 0:
            raise ValueError(f"No phi-{telescope}-{product} observations matched the requested interval.")
        fetched = Fido.fetch(result, path=str(output))
        if getattr(fetched, 'errors', []):
            raise RuntimeError(f"Incomplete phi-{telescope}-{product} download: {len(fetched.errors)} archive transfer(s) failed.")
        if not fetched:
            raise RuntimeError(f"No files were downloaded for phi-{telescope}-{product}.")
        downloaded.extend(str(path) for path in fetched)
    return downloaded
