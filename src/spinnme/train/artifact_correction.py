"""Optional detector-coordinate I-to-polarization leakage correction."""
import torch
from torch import nn

from spinnme.model import MLPModel


class ArtifactCorrectionModule(nn.Module):
    """Learn small I-to-Q/U/V leakages, initialized to the identity.

    ``correction_gate`` controls the artifact correction amplitude. It is not
    the magnetic filling factor, which mixes magnetic and field-free spectra.
    Coordinates must use the adapter's documented, stable detector convention.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = MLPModel(in_dim=2, out_dim=4, dim=64, n_layers=4,
                              encoding_config={'type': 'fourier', 'num_frequencies': 32,
                                               'max_frequencies': 128}, **kwargs)
        with torch.no_grad():
            self.model.out_layer.weight.zero_()
            self.model.out_layer.bias.zero_()

    def forward(self, pix, stokes):
        if pix.shape[-1] != 2 or stokes.shape[-2] != 4 or pix.shape[:-1] != stokes.shape[:-2]:
            raise ValueError('Expected matching detector coordinates [...,2] and Stokes [...,4,wavelength].')
        output = self.model(pix)
        leakage = output[..., :3] * .01
        gate = torch.sigmoid(output[..., 3:4])
        polarization = stokes[..., 1:, :] + gate[..., None] * leakage[..., None] * stokes[..., 0:1, :]
        corrected = torch.cat((stokes[..., 0:1, :], polarization), dim=-2)
        return {'stokes_corr': corrected, 'correction_params': torch.cat((leakage, gate), dim=-1),
                'correction_gate': gate}
