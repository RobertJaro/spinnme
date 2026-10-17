import pytest
import torch

from spinnme.model import GaussianFourierEncoding, MLPModel


def test_gaussian_mapping_and_coordinate_derivatives():
    encoding = GaussianFourierEncoding(2, 3, seed=42).double()
    coords = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)
    phase = (2 * torch.pi * coords) @ encoding.B.T
    torch.testing.assert_close(encoding(coords), torch.cat((phase.sin(), phase.cos()), -1))
    assert torch.autograd.gradcheck(encoding, (coords,))
    assert torch.autograd.gradgradcheck(encoding, (coords,))
    assert list(encoding.parameters()) == []


def test_seed_scaling_and_checkpoint_restore():
    rng = torch.random.get_rng_state()
    base = GaussianFourierEncoding(2, 8, scale=1, seed=7)
    scaled = GaussianFourierEncoding(2, 8, scale=[2, 3], seed=7)
    torch.testing.assert_close(scaled.B, base.B * torch.tensor([2, 3]))
    assert torch.equal(rng, torch.random.get_rng_state())
    restored = GaussianFourierEncoding(2, 8, seed=9)
    restored.load_state_dict(scaled.state_dict())
    torch.testing.assert_close(restored.B, scaled.B)


def test_mlp_integration_and_optional_raw_coordinates():
    config = dict(type='gaussian', num_frequencies=8, scale=[1, 2, 3], seed=0,
                  include_input=True)
    model = MLPModel(3, 2, dim=16, n_layers=2, encoding_config=config, activation='relu')
    coords = torch.randn(5, 3)
    assert model(coords).shape == (5, 2)
    assert model.in_layer.in_features == 19
    torch.testing.assert_close(model.encoding(coords)[..., :3], coords)
    assert config['type'] == 'gaussian'


@pytest.mark.parametrize('options', [dict(num_frequencies=0), dict(num_frequencies=1.5),
                                    dict(scale=0), dict(scale=float('nan')),
                                    dict(scale=[1, 2, 3])])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        GaussianFourierEncoding(2, **options)


def test_reference_feature_order_without_raw_coordinates():
    encoding = GaussianFourierEncoding(2, 2, seed=0).double()
    with torch.no_grad():
        encoding.B.copy_(torch.eye(2, dtype=torch.float64))
    coords = torch.tensor([[0.0, 0.25]], dtype=torch.float64)
    torch.testing.assert_close(encoding(coords),
                               torch.tensor([[0., 1., 1., 0.]], dtype=torch.float64),
                               atol=1e-15, rtol=0)
    assert encoding.out_dim == 4
