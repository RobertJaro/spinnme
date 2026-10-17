# Crop FITS images and cubes

`spinnme crop` operates on FITS images from any instrument. It retains native pixels and spectral samples, updates spatial WCS and writes new files without modifying the inputs. It does not resample or reproject the data.

## Select input pixels

```bash
spinnme crop --input './data/*.fits' --output ./data/cropped \
  --pixels 100 100 356 356
```

The four values are **X0 Y0 X1 Y1**, zero-based with upper bounds excluded. This example retains 256 × 256 pixels. Pixel cropping also works without WCS or instrument metadata. Bounds must lie inside every selected image.

`--input` accepts files, directories or quoted globs, including multiple inputs. Each matched file is processed independently. Choose a new or empty output directory; filenames are retained, so inputs must have distinct filenames.

## Select world coordinates

For solar images with helioprojective WCS:

```bash
spinnme crop --input './data/*.fits' --output ./data/region \
  --bounds -250 -200 -100 -50
```

Bounds are **XMIN YMIN XMAX YMAX** in the file's own WCS, in arcseconds by default. Use `--unit deg` for bounds in degrees, for example with RA/Dec images. World cropping requires valid, separable spatial WCS, but no instrument-specific observer metadata.

The command converts the four world corners to pixels and takes their enclosing rectangle, clipped to the input image. Rotated edges can therefore extend beyond the requested world rectangle. A region with no intersection is rejected. For strongly nonlinear projections or precise curved boundaries, reproject or construct a mask separately. Bounds do not track solar rotation or moving features.

## Select a cube's spatial axes and science extension

By default, the first image HDU is selected and **FITS axes 1 and 2** are x and y. These are the last two dimensions of a NumPy array. For a cube stored as NumPy `(y, x, wavelength, Stokes)` in extension `SCI`:

```bash
spinnme crop --input cube.fits --output ./data/cropped-cube \
  --hdu SCI --spatial-axes 3 4 --pixels 10 20 110 120
```

`--hdu` accepts an extension name or index. `--spatial-axes` uses one-based FITS numbering in x/y order. All other dimensions are retained.

Image extensions with matching spatial shapes are assumed co-aligned and cropped together, including two-dimensional masks or raster acquisition times. Tables, wavelength arrays and unrelated extensions are retained. HDU order and names are preserved; cropped compressed images are written as ordinary image HDUs with decoded pixel values. Spatial WCS using auxiliary distortion lookup tables is rejected.

Existing `CCD_X0`/`CCD_Y0` detector origins are shifted with the crop. Native full-disk HMI images also establish missing origins automatically. See [HMI subframes](hmi.md#invert-an-hmi-subframe) and [Hinode rasters](hinode.md#crop-a-prepared-raster) for inversion examples. To select a region from an existing inversion instead, use [Python exports](python_exports.md).
