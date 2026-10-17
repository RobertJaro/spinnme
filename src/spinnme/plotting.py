"""Reusable map, residual and density figures, independent of instruments/studies.

Inputs are EvaluationResult objects or dictionaries of 2D arrays. Functions
return Matplotlib figures; callers choose styles, scientific labels and outputs.
"""
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from matplotlib.colors import LogNorm, Normalize

from .evaluation.metrics import comparison, residual


def save_figure(figure, stem, formats=('pdf', 'png'), dpi=400):
    """Write publication formats from one figure, returning their paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = stem.with_suffix('.'+extension)
        metadata = {'Date': None} if extension == 'svg' else {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {}
        figure.savefig(path, dpi=dpi, metadata=metadata)
        written.append(path)
    return written


def _arrays(source):
    return source.arrays if hasattr(source, 'arrays') else source


def _mask(source, name, masks):
    values = np.asarray(_arrays(source)[name])
    if values.ndim != 2:
        raise ValueError(f'Plotting requires a scalar 2D map: {name} has shape {values.shape}.')
    valid = np.isfinite(values)
    if hasattr(source, 'mask'):
        valid &= source.mask
    supplied = masks.get(name) if isinstance(masks, dict) else masks
    if supplied is not None:
        valid &= supplied
    return valid


def _layout(source, reference, parameters, header, units):
    arrays = _arrays(source)
    names = ([name for name in arrays if reference is None or name in _arrays(reference)]
             if parameters is None else list(parameters))
    if not names:
        raise ValueError('Select at least one parameter to plot.')
    if header is None:
        header = getattr(reference, 'header', None) if reference is not None else None
        if header is None:
            header = getattr(source, 'header', None)
    inferred_units = dict(getattr(source, 'units', {}))
    inferred_units.update(units or {})
    projection = {} if header is None else {'projection': WCS(header).celestial}
    return names, inferred_units, projection


def magnetic_norm(name, values):
    """Standard SPINNME magnetic maps in gauss; residuals use their own scale."""
    if name in ('Br', 'Btheta', 'Bphi', 'Bt', 'Bp'):
        return Normalize(-300, 300)
    if name in ('B', 'b_field'):
        values = np.asarray(values)
        finite = values[np.isfinite(values)]
        maximum = float(finite.max()) if finite.size else 1.
        return LogNorm(vmin=1, vmax=max(maximum, 1.01), clip=True)
    return None


def _norm(values, supplied=None, name=None):
    if supplied is not None:
        return supplied
    magnetic = magnetic_norm(name, values)
    if magnetic is not None:
        return magnetic
    if not values.size:
        raise ValueError('No finite pixels remain for plotting.')
    low, high = np.percentile(values, [1, 99])
    if low == high:
        low, high = low-1, high+1
    return Normalize(low, high)


def map_axis(ax, values, mask, norm, cmap, *, bottom=True, left=True):
    """Draw one masked map on a regular or celestial-WCS axes."""
    image = ax.imshow(np.where(mask, values, np.nan), origin='lower', norm=norm, cmap=cmap,
                      interpolation='nearest', rasterized=True)
    if not hasattr(ax, 'coords'):
        ax.set(xlabel='X [pixel]' if bottom else '', ylabel='Y [pixel]' if left else '')
        ax.tick_params(labelbottom=bottom, labelleft=left)
        return image
    hpc = tuple(ax.wcs.wcs.ctype) == ('HPLN-TAN', 'HPLT-TAN')
    for index, coord in enumerate(ax.coords):
        if hpc:
            coord.set_format_unit(u.arcsec)
            coord.set_major_formatter('s.s')
        coord.set_ticks(number=4)
        coord.set_ticks_position('b' if index == 0 else 'l')
        visible = bottom if index == 0 else left
        coord.set_ticklabel_visible(visible)
        if not visible:
            coord.set_axislabel('')
        elif hpc:
            coord.set_axislabel('Solar X [arcsec]' if index == 0 else 'Solar Y [arcsec]')
    return image


def _row_label(ax, label):
    ax.annotate(label, xy=(1, .5), xycoords='axes fraction',
                xytext=(8, 0), textcoords='offset points',
                rotation=270, rotation_mode='anchor', ha='center', va='bottom',
                annotation_clip=False)


def plot_comparison(prediction, reference, parameters=None, *, header=None, masks=None,
                    labels=None, units=None, norms=None, cmaps=None, periods=None,
                    source_labels=('Reference', 'Model'), title='', difference_floor=1e-12):
    """Reference/model/residual panels with common scales and wrapped residuals.

    Norms, colormaps, labels and angular periods are dictionaries keyed by product.
    Values must already share a physical grid and units; no alignment is inferred.
    """
    import matplotlib.pyplot as plt
    names, units, projection = _layout(prediction, reference, parameters, header, units)
    labels, norms, cmaps, periods = labels or {}, norms or {}, cmaps or {}, periods or {}
    single = len(names) == 1
    fig, axes = plt.subplots(1 if single else 3, 3 if single else len(names),
        figsize=(8.1, 3.2) if single else (2.7*len(names), 7.5), squeeze=False,
        layout='constrained', subplot_kw=projection)
    if single:
        axes = axes.reshape(3, 1)
    for column, name in enumerate(names):
        a, b = np.asarray(_arrays(prediction)[name]), np.asarray(_arrays(reference)[name])
        valid = _mask(prediction, name, masks) & _mask(reference, name, masks)
        _, difference = comparison(a, b, valid, period=periods.get(name))
        norm = _norm(np.r_[a[valid], b[valid]], norms.get(name), name)
        bound = max(float(np.percentile(np.abs(difference[valid]), 99)), difference_floor)
        quantity = labels.get(name, name)+(f' [{units[name]}]' if units.get(name) else '')
        for row, (values, label) in enumerate(((b, source_labels[0]), (a, source_labels[1]),
                (difference, source_labels[1]+' − '+source_labels[0]))):
            ax = axes[row, column]
            image = map_axis(ax, values, valid, Normalize(-bound, bound) if row == 2 else norm,
                'RdBu_r' if row == 2 else cmaps.get(name, 'viridis'), bottom=single or row == 2,
                left=(row == 0 if single else column == 0))
            if single or column == len(names)-1:
                _row_label(ax, label)
            if row in (1, 2):
                bar = fig.colorbar(image, ax=list(axes[:2, column]) if row == 1 else ax,
                                   location='top', fraction=.045, pad=.02)
                bar.set_label(('Δ ' if row == 2 else '')+quantity)
    fig.suptitle(title)
    return fig


def plot_maps(source, parameters=None, *, header=None, masks=None, labels=None, units=None,
              norms=None, cmaps=None, source_label='Model', title=''):
    """One row of arbitrary scalar maps; source may be an EvaluationResult."""
    import matplotlib.pyplot as plt
    names, units, projection = _layout(source, None, parameters, header, units)
    labels, norms, cmaps = labels or {}, norms or {}, cmaps or {}
    fig, axes = plt.subplots(1, len(names), figsize=(2.7*len(names), 3.2), squeeze=False,
                             layout='constrained', subplot_kw=projection)
    for index, (ax, name) in enumerate(zip(axes.flat, names)):
        values = np.asarray(_arrays(source)[name])
        valid = _mask(source, name, masks)
        image = map_axis(ax, values, valid, _norm(values[valid], norms.get(name), name),
                         cmaps.get(name, 'viridis'), left=index == 0)
        if index == len(names)-1:
            _row_label(ax, source_label)
        quantity = labels.get(name, name)+(f' [{units[name]}]' if units.get(name) else '')
        fig.colorbar(image, ax=ax, location='top', fraction=.045, pad=.02, label=quantity)
    fig.suptitle(title)
    return fig


def plot_scatter(prediction, reference, parameters=None, *, masks=None, labels=None, units=None,
                 periods=None, source_labels=('Reference', 'Model'), title='', columns=4, gridsize=90):
    """Density scatter panels with identity lines and paired-pixel error metrics."""
    import matplotlib.pyplot as plt
    names, units, _ = _layout(prediction, reference, parameters, None, units)
    labels, periods = labels or {}, periods or {}
    columns = min(len(names), max(1, columns))
    rows = int(np.ceil(len(names)/columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.7*columns, 2.8*rows), squeeze=False, layout='constrained')
    for ax, name in zip(axes.flat, names):
        valid = _mask(prediction, name, masks) & _mask(reference, name, masks)
        x, y = np.asarray(_arrays(reference)[name])[valid], np.asarray(_arrays(prediction)[name])[valid]
        metric, _ = comparison(y, x, period=periods.get(name))
        if name in periods:
            y = x+residual(y, x, periods[name])
        image = ax.hexbin(x, y, gridsize=gridsize, bins='log', mincnt=1, cmap='magma', rasterized=True)
        low, high = np.percentile(np.r_[x, y], [.5, 99.5])
        if low == high:
            low, high = low-1, high+1
        ax.plot([low, high], [low, high], color='.45', lw=.8, ls='--')
        unit = units.get(name, '')
        suffix = f' [{unit}]' if unit else ''
        ax.set(xlim=(low, high), ylim=(low, high), aspect='equal',
               xlabel=source_labels[0]+suffix, ylabel=source_labels[1]+suffix)
        ax.text(.04, .96, f'RMSE {metric["rmse"]:.2f}\nMAE {metric["mae"]:.2f}',
                transform=ax.transAxes, va='top')
        bar = fig.colorbar(image, ax=ax, location='top', fraction=.045, pad=.02,
                           label=labels.get(name, name)+' · Pixels')
        bar.minorticks_off()
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    fig.suptitle(title)
    return fig
