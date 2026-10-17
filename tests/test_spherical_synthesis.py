import torch

from spinnme.train.spherical_synthesis import transform_spherical_parameters


def _transform_field(field):
    field = field.reshape(1, 3)
    output = {
        'b_x': field[:, 0:1],
        'b_y': field[:, 1:2],
        'b_z': field[:, 2:3],
        'v_x': torch.zeros_like(field[:, 0:1]),
        'v_y': torch.zeros_like(field[:, 1:2]),
        'v_z': torch.zeros_like(field[:, 2:3]),
    }
    identity = torch.eye(3).unsqueeze(0)
    coords = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return transform_spherical_parameters(
        output, coords, identity, identity,
        meters_per_ds=1.0, seconds_per_dt=1.0,
    )


def test_azimuth_products_are_computed_without_transverse_normalization():
    transformed = _transform_field(torch.tensor([3.0, 4.0, 12.0]))

    torch.testing.assert_close(
        transformed['sin_inc2_sin2azi'], torch.tensor([[-24.0 / 169.0]])
    )
    torch.testing.assert_close(
        transformed['sin_inc2_cos2azi'], torch.tensor([[7.0 / 169.0]])
    )
    assert 'sin2azi' not in transformed
    assert 'cos2azi' not in transformed


def test_azimuth_product_gradients_are_finite_for_longitudinal_field():
    field = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
    transformed = _transform_field(field)
    loss = (
        transformed['sin_inc2_sin2azi']
        + transformed['sin_inc2_cos2azi']
        + transformed['sin_inc2']
        + transformed['cos_inc']
    ).sum()
    loss.backward()

    assert torch.isfinite(field.grad).all()
