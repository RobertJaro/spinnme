"""Download calibrated Hinode/SP Stokes slit files from DARTS."""
from datetime import datetime, timedelta
from html.parser import HTMLParser
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import urlopen

from .fits import parse_time

ARCHIVE = 'https://data.darts.isas.jaxa.jp/pub/hinode/sot/level1hao/'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.append(dict(attrs).get('href', ''))


def _listing(url):
    parser = Links()
    try:
        with urlopen(url, timeout=60) as stream:
            parser.feed(stream.read().decode('utf-8'))
    except HTTPError as error:
        if error.code != 404:
            raise
    return parser.links


def download_hinode(output, start, end, *, time_scale='utc'):
    """Download Level-1 slit files whose archive timestamps lie in [start, end)."""
    from .download import _dispatch

    begin, finish = parse_time(start, time_scale), parse_time(end, time_scale)
    if finish <= begin:
        raise ValueError('Hinode end must be later than start.')
    urls = []
    # Include scans starting on the preceding day and continuing past midnight.
    day = begin.date()-timedelta(days=1)
    while day <= finish.date():
        base = urljoin(ARCHIVE, day.strftime('%Y/%m/%d/SP3D/'))
        for raster in _listing(base):
            try:
                epoch = datetime.strptime(raster, '%Y%m%d_%H%M%S/')
            except ValueError:
                continue
            if epoch >= finish:
                continue
            directory = urljoin(base, raster)
            for name in _listing(directory):
                if not name.startswith('SP3D') or not name.endswith('C.fits') or '/' in name:
                    continue
                epoch = datetime.strptime(name[4:-6], '%Y%m%d_%H%M%S.%f')
                if begin <= epoch < finish:
                    urls.append(urljoin(directory, name))
        day += timedelta(days=1)
    if not urls:
        raise RuntimeError('No calibrated Hinode Stokes files in the requested interval.')
    return _dispatch('hinode', output=output, urls=sorted(set(urls)))
