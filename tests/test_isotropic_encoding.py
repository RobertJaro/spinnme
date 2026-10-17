import torch

from spinnme.model import IsotropicPositionalEncoding


def test_sunerf_mapping_and_derivatives():
    encoding = IsotropicPositionalEncoding(
        4, [8, 3], [-1, 0], [4, 2], seed=7
    ).double()
    coords = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    spatial = coords[..., :3] @ encoding.B.T
    temporal = coords[..., 3:] * encoding.frequencies[0]
    expected = torch.cat((spatial.sin(), spatial.cos(), temporal.sin(), temporal.cos()), -1)
    torch.testing.assert_close(encoding(coords), expected)
    torch.testing.assert_close(encoding.B.norm(dim=-1),
                               (2 ** torch.linspace(-1, 4, 8) * torch.pi).double())
    assert torch.autograd.gradcheck(encoding, (coords,))
    assert torch.autograd.gradgradcheck(encoding, (coords,))
    assert all(not param.requires_grad for param in encoding.parameters())


def test_independent_space_time_and_checkpoint():
    encoding = IsotropicPositionalEncoding(4, [8, 3], seed=7)
    coords = torch.randn(5, 4)
    changed = coords.clone()
    changed[:, 3] += 1
    torch.testing.assert_close(encoding(coords)[..., :16], encoding(changed)[..., :16])
    restored = IsotropicPositionalEncoding(4, [8, 3], seed=9)
    restored.load_state_dict(encoding.state_dict())
    torch.testing.assert_close(restored(coords), encoding(coords))
    spatial_only = IsotropicPositionalEncoding(3, 8, include_input=True)
    assert spatial_only(coords[..., :3]).shape == (5, 19)
    torch.testing.assert_close(spatial_only(coords[..., :3])[..., :3], coords[..., :3])
