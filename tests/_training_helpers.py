import torch
from astropy import units as u
from spinnme.instrument import hmi_wavelength_config
from spinnme.train.me_spherical_module import MESphericalModule

def _module(_unused=None, **kwargs):
    return MESphericalModule(
        image_shape=(4, 4), wavelength_config={'HMI': hmi_wavelength_config()},
        instrument_config=[{'type': 'hmi', 'instrument_id': 'HMI'}],
        gauss_per_dB=1000, Rs_per_ds=1, seconds_per_dt=36000,
        model_config={'type': 'mlp', 'dim': 8, 'n_layers': 2,
                      'encoding_config': {'type': 'identity'}},
        stokes_loss_config={'continuum_normalization': True},
        normalization_config={'stokes_scales': [1, .01, .01, .01]},
        optimizer_config={'type': 'adamw'}, **kwargs,
    )


def _physical_batch():
    inputs = {'coords': torch.zeros(16, 4), 'stokes': torch.zeros(16, 4, 6),
              'spectral_weights': torch.zeros(16, 6, 3)}
    inputs['coords'][:, 1] = 1
    inputs['coords'][:, 3] = torch.arange(16) % 2
    inputs['stokes'].zero_()
    inputs['stokes'][:, 0] = 1
    inputs['cartesian_to_spherical_transform'] = torch.eye(3).expand(16, 3, 3)
    inputs['rtp_to_img_transform'] = torch.tensor([[0., 1., 0.], [0., 0., 1.],
                                                  [1., 0., 0.]]).expand(16, 3, 3)
    inputs['v_obs_los'] = torch.zeros(16, 1)
    inputs['wavelength_grid'] = torch.tensor(
        hmi_wavelength_config()['wavelength_grid'].to_value(u.m), dtype=torch.float32,
    ).expand(16, -1)
    inputs['spectral_weights'].fill_(1 / 3)
    inputs['spectral_offsets'] = torch.zeros(16, 6, 3)
    inputs['continuum_weights'] = torch.zeros(16, 6)
    return {'HMI': inputs}



def physical_inputs(batch_size, wavelengths, device, dtype=torch.float32):
    # Deterministic, varied atmospheres including disk-center/limb and weak fields.
    column = torch.linspace(0, 1, batch_size, device=device, dtype=dtype)[:, None]
    parameters = dict(b_field=10 + 1990 * column, cos_inc=.1 + .8 * column,
                      vmac=1500 + 1000 * column, damping=.01 + .99 * column,
                      b0=.8 - .3 * column, b1=.2 + .3 * column,
                      mu=.05 + .95 * column, vdop=-3000 + 6000 * column,
                      kl=5 + 20 * column)
    parameters['sin_inc2'] = 1 - parameters['cos_inc'].square()
    parameters['sin_inc2_cos2azi'] = parameters['sin_inc2'] * torch.cos(column * 4)
    parameters['sin_inc2_sin2azi'] = parameters['sin_inc2'] * torch.sin(column * 4)
    parameters = {key: value.detach().requires_grad_() for key, value in parameters.items()}
    grid = torch.linspace(-.2, .2, wavelengths, device=device, dtype=dtype)[None] * 1e-10
    # Match the per-profile grid layout in training.
    return parameters, grid.expand(batch_size, -1).contiguous()
