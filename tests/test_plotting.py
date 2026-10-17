"""Instrument-independent comparisons and plots with explicit conventions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits

from spinnme.evaluation import comparison
from spinnme.exports import EvaluationResult
from spinnme.plotting import plot_comparison, plot_maps, plot_scatter, save_figure


def test_masked_metrics_unsigned_values_and_arbitrary_period():
    metric, difference = comparison(np.array([1, 255], dtype=np.uint8), np.array([2, 254], dtype=np.uint8))
    np.testing.assert_array_equal(difference, [-1, 1])
    assert metric['mae'] == 1
    metric, difference = comparison(np.array([359., np.nan, 1.]), np.array([1., 0., 359.]), period=360)
    np.testing.assert_allclose(difference[[0, 2]], [-2, 2])
    assert metric['n'] == 2 and metric['pearson_r'] is None
    values = np.ones((2, 3, 4))
    metric, _ = comparison(values, values+2, np.array([[True, False, False], [False, False, True]]))
    assert metric['n'] == 8
    assert metric['rmse'] == 2


def test_generic_map_panels_share_scales_and_honor_masks(tmp_path):
    values = np.arange(48., dtype=float).reshape(6, 8)
    target = {'temperature': values.copy()}
    prediction = {'temperature': values+2}
    prediction['temperature'][0, 0] = 1e8
    mask = np.ones((6, 8), dtype=bool)
    mask[0, 0] = False
    figure = plot_comparison(prediction, target, masks=mask, units={'temperature': 'K'},
                             source_labels=('Observation', 'Fit'), labels={'temperature': 'Temperature'})
    panels = [ax for ax in figure.axes if ax.images]
    assert len(panels) == 3
    assert panels[0].images[0].get_clim() == panels[1].images[0].get_clim()
    assert panels[0].images[0].get_clim()[1] < 100
    assert panels[1].images[0].get_array().mask[0, 0]
    assert panels[0].texts[0].get_text() == 'Observation'
    assert panels[0].get_title() == ''
    for path in save_figure(figure, tmp_path/'nested'/'comparison', formats=('png', 'pdf', 'svg'), dpi=90):
        assert path.stat().st_size > 1000
    plt.close(figure)


def test_plotting_results_non_solar_wcs_and_wrapped_scatter(tmp_path):
    values = np.arange(48., dtype=float).reshape(6, 8)
    header = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN', 'CUNIT1': 'deg', 'CUNIT2': 'deg',
                          'CRPIX1': 4., 'CRPIX2': 3., 'CRVAL1': 20., 'CRVAL2': 30., 'CDELT1': .01, 'CDELT2': .01})
    result = EvaluationResult({'phase': values+358}, {'phase': 'deg'}, header, np.ones((6, 8), bool), {})
    target = EvaluationResult({'phase': values}, {'phase': 'deg'}, header, result.mask, {})
    figure = plot_maps(result, source_label='Fit')
    assert figure.axes[0].images[0].get_array().shape == (6, 8)
    save_figure(figure, tmp_path/'sky', formats=('png',), dpi=90)
    plt.close(figure)
    figure = plot_scatter(result, target, periods={'phase': 360}, columns=1)
    assert 'RMSE 2.00' in figure.axes[0].texts[0].get_text()
    assert figure.axes[0].get_xlabel() == 'Reference [deg]'
    save_figure(figure, tmp_path/'scatter', formats=('png',), dpi=90)
    plt.close(figure)


def test_comparison_rejects_empty_overlap():
    with pytest.raises(ValueError, match='No finite paired'):
        comparison(np.array([np.nan]), np.array([0.]))
