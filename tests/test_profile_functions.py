import numpy as np
import torch
from scipy.special import wofz

from spinnme.train.profile_functions import Faddeeva, FaradayVoigt, Voigt


def test_faddeeva_matches_scipy_in_profile_domain():
    x = torch.linspace(-20, 20, 401, dtype=torch.float64)
    damping = torch.tensor([0.0, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    z = x[:, None] + 1j * damping

    actual = Faddeeva()(z).detach().numpy()
    expected = wofz(z.numpy())

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual[x.numel() // 2, 0], 1.0, rtol=0, atol=1e-12)


def test_faddeeva_gradient_matches_analytic_derivative():
    x = torch.linspace(-8, 8, 65, dtype=torch.float64, requires_grad=True)
    damping = torch.full_like(x, 0.2)
    z = torch.complex(x, damping)
    value = Faddeeva()(z)

    real_gradient = torch.autograd.grad(value.real.sum(), x, retain_graph=True)[0]
    imaginary_gradient = torch.autograd.grad(value.imag.sum(), x)[0]
    expected = -2 * z * value + 2j / np.sqrt(np.pi)

    torch.testing.assert_close(real_gradient, expected.real, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(imaginary_gradient, expected.imag, rtol=1e-9, atol=1e-10)


def test_voigt_and_faraday_voigt_symmetries():
    x = torch.linspace(0, 10, 101, dtype=torch.float64)
    sigma = torch.ones_like(x)
    damping = torch.full_like(x, 0.2)

    voigt = Voigt()
    faraday = FaradayVoigt()
    torch.testing.assert_close(voigt(-x, sigma, damping), voigt(x, sigma, damping))
    torch.testing.assert_close(faraday(-x, sigma, damping), -faraday(x, sigma, damping))


def test_shared_profiles_match_separate_outputs_and_all_input_gradients():
    for dtype in (torch.float32, torch.float64):
        x = torch.linspace(-20, 20, 117, dtype=dtype).reshape(13, 9).requires_grad_()
        sigma = torch.linspace(.4, 2., 13, dtype=dtype)[:, None].requires_grad_()
        gamma = torch.linspace(0., 2., 13, dtype=dtype)[:, None].requires_grad_()
        inputs = (x, sigma, gamma)
        pair = Voigt().forward_pair(*inputs)
        separate = (Voigt()(*inputs), FaradayVoigt()(*inputs))
        for actual, expected in zip(pair, separate):
            torch.testing.assert_close(actual, expected)
        weights = torch.linspace(-1, 1, x.numel(), dtype=dtype).reshape_as(x)
        actual_grad = torch.autograd.grad((pair[0] * weights + pair[1].square()).sum(), inputs)
        expected_grad = torch.autograd.grad((separate[0] * weights + separate[1].square()).sum(), inputs)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=2e-5 if dtype == torch.float32 else 1e-10,
                                       atol=2e-6 if dtype == torch.float32 else 1e-12)


def test_shared_profiles_gradcheck():
    inputs = (torch.tensor([-.8, .2, 2.], dtype=torch.float64, requires_grad=True),
              torch.tensor([.7, 1., 1.3], dtype=torch.float64, requires_grad=True),
              torch.tensor([.01, .2, .7], dtype=torch.float64, requires_grad=True))
    assert torch.autograd.gradcheck(Voigt().forward_pair, inputs)


def test_unit_upper_profiles_match_generic_outputs_and_gradients():
    for dtype in (torch.float32, torch.float64):
        x = torch.linspace(-20, 20, 117, dtype=dtype).reshape(13, 9).requires_grad_()
        gamma = torch.linspace(0, 2, 13, dtype=dtype)[:, None].requires_grad_()
        profile = Voigt()
        actual = profile.forward_pair_unit(x, gamma)
        expected = profile.forward_pair(x, torch.ones_like(x), gamma)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        actual_grad = torch.autograd.grad(sum(v.square().sum() for v in actual), (x, gamma))
        expected_grad = torch.autograd.grad(sum(v.square().sum() for v in expected), (x, gamma))
        for a, b in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, b)


def test_generic_faddeeva_retains_lower_half_plane_reflection():
    z = torch.tensor([-.8-.2j, .5-.1j, 2.-.7j], dtype=torch.complex128)
    np.testing.assert_allclose(Faddeeva()(z).numpy(), wofz(z.numpy()), rtol=1e-10, atol=1e-12)
