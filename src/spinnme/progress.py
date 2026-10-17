"""Progress bars for training and validation data loading."""
from multiprocessing import current_process
import os

from tqdm import tqdm


def progress(iterable, *, desc, total=None, position=0):
    """Throttle terminal updates; preparation children leave the parent bar alone."""
    rank = os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0'))
    return tqdm(iterable, desc=desc, total=total, unit='item',
                mininterval=2, dynamic_ncols=True, leave=True, position=position,
                disable=current_process().name != 'MainProcess' or rank != '0')
