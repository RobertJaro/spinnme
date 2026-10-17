"""Export physical maps on a custom WCS grid and optional fixed-observer cadence.

Run from the checkout: python examples/export_python.py --help
Grid geometry, calibration, support masks and timing are handled by the loader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from spinnme import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observation', help='Saved observation ID; defaults to the first saved grid.')
    parser.add_argument('--shape', nargs=2, type=int, metavar=('NY', 'NX'))
    parser.add_argument('--bounds', nargs=4, type=float, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'))
    parser.add_argument('--cadence', type=float, help='Output interval in seconds at fixed observing geometry.')
    parser.add_argument('--device', default=None, help='Optional device override; default selects CUDA when available, otherwise CPU.')
    args = parser.parse_args()
    with load(args.snapshot, device=args.device) as snapshot:
        observation = args.observation or snapshot.observations[0]['id']
        grid = snapshot.grid(observation, shape=args.shape, bounds=args.bounds)
        times = [None] if args.cadence is None else snapshot.cadence_times(args.cadence, instrument_id=grid['instrument_id'])
        for index, epoch in enumerate(times):
            result = snapshot.evaluate(grid, time=epoch, parameters=['Brtp', 'Vrtp', 'Blos', 'Vlos'])
            directory = args.output if epoch is None else args.output/f'frame_{index:04d}'
            result.write_fits(directory/'fits')
            result.write_npz(directory/'parameters.npz')
            result.write_jpg(directory/'preview')
            print(directory, result.metadata['model_time'])


if __name__ == '__main__':
    main()
