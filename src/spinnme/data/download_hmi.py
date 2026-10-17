"""Import-safe JSOC HMI downloads with explicit input time scales."""
from pathlib import Path

from astropy.time import Time


def download_series(download_dir, email, t_start, t_end=None,
                    cadence="720s", series="S_720s", segments=None,
                    time_scale="utc"):
    """Export an HMI series; plain ISO input is UTC unless configured otherwise."""
    try:
        import drms
    except ImportError as error:
        raise ImportError("HMI downloads require the spinnme[download] extra.") from error
    begin = Time(t_start, scale=time_scale).tai
    stamp = begin.strftime("%Y.%m.%d_%H:%M:%S_TAI")
    series = series if series.startswith("hmi.") else "hmi." + series
    if t_end is None:
        query = f"{series}[{stamp}]"
    else:
        duration = (Time(t_end, scale=time_scale).tai - begin).to_value("s")
        if duration <= 0:
            raise ValueError("t_end must be later than t_start.")
        query = f"{series}[{stamp}/{duration:g}s@{cadence}]"
    if segments:
        query += "{" + (segments if isinstance(segments, str) else ",".join(segments)) + "}"
    output = Path(download_dir)
    output.mkdir(parents=True, exist_ok=True)
    export = drms.Client(email=email).export(query, protocol="fits")
    export.wait()
    if export.status != 0:
        raise RuntimeError(f"JSOC export failed for {query}: status={export.status}")
    return export.download(str(output))
