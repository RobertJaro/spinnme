import pytest
import torch

from spinnme.model import LimbCorrectionModel
from spinnme.train.spherical_synthesis import (
    field_free_forward_parameters,
    mix_magnetic_filling_factor,
)


def test_limb_correction_predicts_velocity_and_filling_factor_from_mu():
    module = LimbCorrectionModel()
    mu = torch.tensor([[0.2], [0.7], [1.0]])

    correction = module(mu)

    assert set(correction) == {'c_b0', 'c_b1', 'c_vdop', 'filling_factor'}
    assert all(value.shape == mu.shape for value in correction.values())
    torch.testing.assert_close(
        correction['filling_factor'],
        torch.full_like(mu, 0.99),
    )


def test_field_free_atmosphere_preserves_parameters_except_magnetic_strength():
    parameters = {
        'b_field': torch.tensor([[1200.0], [300.0]]),
        'vdop': torch.tensor([[100.0], [-50.0]]),
        'b0': torch.tensor([[0.4], [0.6]]),
    }

    field_free = field_free_forward_parameters(parameters)

    torch.testing.assert_close(field_free['b_field'], torch.zeros_like(parameters['b_field']))
    assert field_free['vdop'] is parameters['vdop']
    assert field_free['b0'] is parameters['b0']
    torch.testing.assert_close(parameters['b_field'], torch.tensor([[1200.0], [300.0]]))


@pytest.mark.parametrize('factor', [0.0, 0.25, 1.0])
def test_filling_factor_linearly_mixes_magnetic_and_field_free_stokes(factor):
    magnetic = torch.full((2, 4, 6), 3.0)
    field_free = torch.full((2, 4, 6), 1.0)
    filling_factor = torch.full((2, 1), factor, requires_grad=True)

    mixed = mix_magnetic_filling_factor(magnetic, field_free, filling_factor)

    torch.testing.assert_close(mixed, torch.full_like(mixed, 1.0 + 2.0 * factor))
    mixed.sum().backward()
    torch.testing.assert_close(
        filling_factor.grad,
        torch.full_like(filling_factor, 4 * 6 * 2.0),
    )


def test_filling_factor_rejects_sample_shape_mismatch():
    stokes = torch.zeros(2, 4, 6)
    with pytest.raises(ValueError, match='filling_factor must have shape'):
        mix_magnetic_filling_factor(stokes, stokes, torch.zeros(2))
