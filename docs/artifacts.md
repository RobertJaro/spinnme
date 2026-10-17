# Instrument corrections

Corrections are off in the bundled templates. Enable them in a new run and compare the resulting Stokes residuals and field maps with an uncorrected run.

## Correct intensity leakage into polarization

Set `artifact_correction: true` in the relevant `instrument` entry of your YAML, then run that config:

```bash
spinnme run --config phi-corrected.yaml --run_path ./runs/phi-corrected
```

The model learns spatial coefficients for intensity leaking into Q/U/V, multiplied by a learned gate. It leaves I unchanged and starts with zero leakage. This corrects a fitted leakage pattern; it does not replace flat-field, fringe or cavity calibration. A lower residual after adding fitted corrections does not independently establish an instrumental origin; compare with calibration evidence and a controlled run without corrections.

Export the fitted coefficients and gate:

```bash
spinnme export ./runs/phi-corrected/inversion.pme \
  --all-observations --parameters Artifact --format fits \
  --output ./products/phi-artifacts
```

`Artifact` writes `artifact_alpha_Q`, `artifact_alpha_U`, `artifact_alpha_V` and `artifact_gate`. Interpret each coefficient together with the gate: their product sets the leakage amplitude. The gate is separate from a magnetic filling factor.

For stable detector-coordinate corrections across cutouts, generic inputs can provide `detector_shape`, `detector_origin` and `detector_step`; HMI/PHI readers retain detector mappings. Without that mapping, the fitted pattern follows normalized image coordinates.

## Other optional corrections

| Instrument setting | Effect | Diagnostic export |
| --- | --- | --- |
| `limb_correction: true` | Fits viewing-angle-dependent source scaling, Doppler correction and magnetic filling | `limb_filling_factor` |
| `velocity_correction: true` | Fits an instrument-specific time-dependent Doppler offset | `forward_vdop` |

Default presets use one magnetic component. Enabling limb correction adds a magnetic/field-free mixture using the same atmosphere; its filling factor differs from the artifact leakage gate. A learned velocity offset can trade against the fitted solar Doppler zero-point. Keep correction settings fixed when comparing training options. Select intrinsic parameters individually, such as `Vdop`, `vmac` or `b0`. Use explicit corrected keys such as `forward_vdop` or `forward_b0` for values after configured instrument corrections. Use `IQUV` to export the synthesized spectra through the same correction path.
