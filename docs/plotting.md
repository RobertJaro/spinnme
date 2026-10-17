# Reusable comparison and plotting

The framework provides array metrics in `spinnme.evaluation` and map figures in
`spinnme.plotting`. Dataset acquisition, pairing, unit assumptions, physical
conventions and study-specific selections remain in the evaluation project.

```python
import matplotlib.pyplot as plt
from spinnme.exports import read_npz
from spinnme.evaluation import comparison
from spinnme.plotting import plot_comparison, plot_maps, plot_scatter, save_figure

prediction = read_npz("products/prediction.npz")
reference = read_npz("products/reference.npz")

# The caller aligns grids, units and conventions before comparison.
metrics, difference = comparison(
    prediction.arrays["Br"], reference.arrays["Br"],
    prediction.mask & reference.mask,
)
print(metrics["bias"], metrics["mae"], metrics["rmse"])

figure = plot_comparison(
    prediction, reference, parameters=["Br", "Btheta", "Bphi"],
    source_labels=("Pipeline", "SPINNME"), title="Magnetic field comparison",
)
save_figure(figure, "figures/magnetic_comparison", formats=("pdf", "png"), dpi=400)
plt.close(figure)

figure = plot_scatter(
    prediction, reference, parameters=["azimuth"], periods={"azimuth": 180},
    source_labels=("Pipeline", "SPINNME"),
)
save_figure(figure, "figures/azimuth_scatter")
plt.close(figure)

figure = plot_maps(prediction, parameters=["vr", "vtheta", "vphi"], source_label="SPINNME")
save_figure(figure, "figures/velocity")
plt.close(figure)
```

`plot_comparison` shows reference, model and difference with shared map scales.
`plot_scatter` shows density, an identity line and the paired RMSE/MAE.
`plot_maps` draws any selection of scalar maps. Functions return Matplotlib
figures and do not change the backend or global style.

Inputs can also be dictionaries of 2D arrays. Pass `header` for celestial WCS;
without it, maps use pixel axes. Result objects provide WCS, masks and units.
`labels`, `units`, `norms`, `cmaps` and `periods` are optional dictionaries keyed
by product. `norms` accepts Matplotlib normalization objects. `masks` accepts a
shared mask or a dictionary of per-product masks. Colormaps, angular conventions
and scientific ranges are explicit caller choices.

Standard magnetic maps in gauss default to `LogNorm` for `B`/`b_field`, from
1 G to the data maximum, and linear −300 to +300 G for `Br`, `Btheta`, `Bphi`.
`magnetic_norm(name, values)` exposes these defaults for study-specific layouts.
Explicit `norms` override them. Difference maps retain a signed linear scale.

`comparison(predicted, reference, mask=None, period=None)` returns metrics and
a residual array with invalid pixels set to NaN. Metrics include paired count,
bias, MAE, RMSE, median/p95 absolute error and Pearson correlation. For angular
residuals, provide a positive period in the arrays' units; residuals wrap around
zero and Pearson correlation is omitted. The function supports scalar maps,
spectra and coordinate samples with leading support masks.

The plotting layer assumes inputs already share a physical grid and units. It
does not infer instrument conventions, apply reference offsets or disambiguate
vectors. HMI, PHI and study adapters supply those choices in `spinnme-evaluation`.
