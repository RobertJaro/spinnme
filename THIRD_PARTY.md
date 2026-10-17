# Third-party scientific code

The core is derived from Robert Jarolim's PINN-ME research code and remains under
the repository's GPL-3.0 license. Source headers and attributions are retained.

`src/spinnme/train/soap.py` retains the SOAP implementation and
attribution to PyTorch AdamW and GaLore. Source files retain the applicable
scientific-code attributions.

Copies of the upstream notices are distributed in `licenses/`: SOAP's MIT
notice (Nikhil Vyas, 2024), GaLore's Apache-2.0 license, and PyTorch's BSD notice.
The SOAP implementation has been adapted for SPINNME's scientific
training and numerical diagnostics. Upstream sources:

- https://github.com/nikhilvyas/SOAP
- https://github.com/jiaweizzhao/GaLore
- https://github.com/pytorch/pytorch

Data downloaded from mission archives are not included in the distribution.
Follow the relevant mission's acknowledgement, citation and data-use guidance.
