import math

import pytest
import torch

from spinnme.train import physics
from spinnme.train.physics import (
    PhysicsCharacteristicScales,
    PhysicsConstraintModule,
    PhysicsWeightSchedule,
)

DTYPE = torch.float64


@pytest.mark.parametrize("vector_potential", [False, True])
def test_characteristic_losses_are_invariant_to_model_units(vector_potential):
    """The same physical fields must have identical losses under dataset rescaling."""
    results = []
    gradients = []
    for scales in (
        PhysicsCharacteristicScales(0.1, 0.2, 2.0, 0.5),
        PhysicsCharacteristicScales(0.005, 0.125, 0.25, 0.04),
    ):
        coords = (_coords().detach() * torch.tensor(
            [scales.length] * 3 + [scales.time], dtype=DTYPE
        )).requires_grad_()
        x, y, z, t = (coords / coords.new_tensor(
            [scales.length] * 3 + [scales.time]
        )).unbind(dim=-1)
        amplitude = torch.tensor(1.0, dtype=DTYPE, requires_grad=True)
        b = amplitude * torch.stack((x + t, 2 * y + x, z), dim=-1) * scales.magnetic_field
        v = torch.stack((torch.ones_like(x), 0 * x, 0 * x), dim=-1) * scales.velocity
        selected = {"divergence", "potential", "force_free", "dB_dt", "induction"}
        a_jacobian = None
        if vector_potential:
            # Independent analytic A for checking induction/gauge unit conversion.
            a = amplitude * torch.stack((-z * (y + x / 2), z * (x + t), 0 * x), dim=-1)
            a *= scales.magnetic_field * scales.length
            a_jacobian = physics.jacobian(a, coords)
            selected.add("gauge")
        result = PhysicsConstraintModule(
            normalization="characteristic", characteristic_scales=scales
        )(b, v, coords, selected, a_jac_matrix=a_jacobian)
        sum(loss.sum() for loss in result.losses.values()).backward()
        results.append(result)
        gradients.append(amplitude.grad)
    for name in results[0].losses:
        torch.testing.assert_close(results[0].losses[name], results[1].losses[name])
        torch.testing.assert_close(results[0].residual_norms[name], results[1].residual_norms[name])
    torch.testing.assert_close(results[0].losses["divergence"], torch.full((2,), 16.0, dtype=DTYPE))
    torch.testing.assert_close(results[0].losses["potential"], torch.ones(2, dtype=DTYPE))
    torch.testing.assert_close(results[0].losses["dB_dt"], torch.ones(2, dtype=DTYPE))
    torch.testing.assert_close(gradients[0], gradients[1])
    assert torch.isfinite(gradients[0])


@pytest.mark.parametrize("vector_potential", [False, True])
def test_characteristic_induction_supports_independent_time_and_velocity(vector_potential):
    coords = _coords()
    x, _, _, t = coords.unbind(dim=-1)
    zero = 0 * t
    b = torch.stack((zero, x - t, zero), dim=-1)
    v = torch.stack((torch.ones_like(t), zero, zero), dim=-1)
    a_jacobian = None
    if vector_potential:
        a = torch.stack((zero, zero, -0.5 * (x - t).square()), dim=-1)
        a_jacobian = physics.jacobian(a, coords)
    result = PhysicsConstraintModule(
        normalization="characteristic",
        characteristic_scales=PhysicsCharacteristicScales(2.0, 3.0, 4.0, 5.0),
    )(b, v, coords, {"induction"}, a_jac_matrix=a_jacobian)
    torch.testing.assert_close(result.losses["induction"], torch.zeros(2, dtype=DTYPE), atol=1e-28, rtol=0)


def test_characteristic_normalization_requires_valid_scales():
    with pytest.raises(ValueError, match="requires characteristic scales"):
        PhysicsConstraintModule(normalization="characteristic")
    for value in (0, -1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="finite and positive"):
            PhysicsCharacteristicScales(value, 1, 1, 1)


def _coords():
    return torch.tensor(
        [
            [0.2, 1.0, 0.4, 0.8],
            [0.7, 0.6, 1.2, 0.5],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )


def _zero_velocity(coords):
    t = coords[:, 3]
    return torch.stack((0 * t, 0 * t, 0 * t), dim=-1)


def test_raw_constraints_match_analytic_linear_field():
    coords = _coords()
    x, y, z, _ = coords.unbind(dim=-1)
    b = torch.stack((x, y, z), dim=-1)

    result = PhysicsConstraintModule()(
        b,
        _zero_velocity(coords),
        coords,
        selected={"divergence", "force_free", "potential", "dB_dt"},
    )

    torch.testing.assert_close(
        result.losses["divergence"], torch.full((2,), 9.0, dtype=DTYPE)
    )
    torch.testing.assert_close(result.losses["force_free"], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(result.losses["potential"], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(result.losses["dB_dt"], torch.zeros(2, dtype=DTYPE))
    assert set(result.losses) == {"divergence", "force_free", "potential", "dB_dt"}
    assert set(result.residual_norms) == set(result.losses)
    assert result.diagnostics == {}


def test_force_free_beltrami_field_has_unit_current_and_zero_force():
    coords = _coords()
    z = coords[:, 2]
    b = torch.stack((torch.sin(z), torch.cos(z), 0 * z), dim=-1)

    result = PhysicsConstraintModule()(
        b,
        _zero_velocity(coords),
        coords,
        selected={"divergence", "force_free", "potential"},
    )

    torch.testing.assert_close(result.losses["divergence"], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(result.losses["force_free"], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(result.losses["potential"], torch.ones(2, dtype=DTYPE))


def test_force_free_is_lorentz_force_normalized_by_field_strength():
    coords = _coords()
    x, y, _, _ = coords.unbind(dim=-1)
    one = torch.ones_like(x)
    b = torch.stack((-y, x, one), dim=-1)
    epsilon = 1e-6

    result = PhysicsConstraintModule(epsilon=epsilon)(
        b,
        _zero_velocity(coords),
        coords,
        selected={"force_free"},
    )

    # curl(B) = (0, 0, 2), so |curl(B) x B|^2 = 4 * (x^2 + y^2).
    expected = 4 * (x.square() + y.square()) / (
        x.square() + y.square() + 1 + epsilon**2
    )
    torch.testing.assert_close(result.losses["force_free"], expected)
    torch.testing.assert_close(
        result.residual_norms["force_free"], torch.sqrt(expected)
    )


def test_force_free_normalization_is_finite_at_a_magnetic_null():
    coords = _coords()
    amplitude = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    x, y, _, _ = coords.unbind(dim=-1)
    b = amplitude * torch.stack((-y, x, torch.ones_like(x)), dim=-1)

    loss = PhysicsConstraintModule()(b, _zero_velocity(coords), coords, {"force_free"}).losses[
        "force_free"
    ]
    loss.sum().backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert torch.isfinite(amplitude.grad)


def test_direct_induction_matches_exact_advected_field():
    coords = _coords()
    x, _, _, t = coords.unbind(dim=-1)
    zero = 0 * t
    b = torch.stack((zero, x - t, zero), dim=-1)
    v = torch.stack((torch.ones_like(t), zero, zero), dim=-1)

    result = PhysicsConstraintModule()(b, v, coords, selected={"induction"})

    torch.testing.assert_close(result.losses["induction"], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(
        result.residual_norms["induction"], torch.zeros(2, dtype=DTYPE)
    )


def test_vector_potential_induction_and_gauge_match_exact_solution():
    coords = _coords()
    t = coords[:, 3]
    zero = 0 * t
    b = torch.stack((zero, torch.ones_like(t), zero), dim=-1)
    v = torch.stack((torch.ones_like(t), zero, zero), dim=-1)
    a_jacobian = torch.zeros((coords.shape[0], 3, 4), dtype=DTYPE)
    a_jacobian[:, 2, 3] = 1

    result = PhysicsConstraintModule()(
        b,
        v,
        coords,
        selected={"induction", "gauge"},
        a_jac_matrix=a_jacobian,
    )

    torch.testing.assert_close(result.losses["induction"], torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(result.losses["gauge"], torch.zeros(2, dtype=DTYPE))


def test_dB_dt_is_squared_and_has_finite_gradient_at_zero_amplitude():
    coords = _coords()
    t = coords[:, 3]
    amplitude = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    b = amplitude * torch.stack((t, 2 * t, 0 * t), dim=-1)

    loss = PhysicsConstraintModule()(
        b,
        _zero_velocity(coords),
        coords,
        selected={"dB_dt"},
    ).losses["dB_dt"]
    loss.sum().backward()

    torch.testing.assert_close(loss, torch.zeros(2, dtype=DTYPE))
    torch.testing.assert_close(amplitude.grad, torch.tensor(0.0, dtype=DTYPE))
    assert torch.isfinite(amplitude.grad)

    nonzero_amplitude = torch.tensor(2.0, dtype=DTYPE, requires_grad=True)
    b = nonzero_amplitude * torch.stack((t, 2 * t, 0 * t), dim=-1)
    loss = PhysicsConstraintModule()(
        b,
        _zero_velocity(coords),
        coords,
        selected={"dB_dt"},
    ).losses["dB_dt"]
    torch.testing.assert_close(loss, torch.full((2,), 20.0, dtype=DTYPE))


def test_relative_divergence_is_invariant_to_field_amplitude():
    coords = _coords()
    x, y, z, _ = coords.unbind(dim=-1)
    base = torch.stack((2 * x, y, z), dim=-1)
    module = PhysicsConstraintModule(normalization="relative")

    original = module(base, _zero_velocity(coords), coords, {"divergence"}).losses[
        "divergence"
    ]
    scaled = module(7 * base, _zero_velocity(coords), coords, {"divergence"}).losses[
        "divergence"
    ]

    torch.testing.assert_close(original, torch.full((2,), 16 / 6, dtype=DTYPE))
    torch.testing.assert_close(scaled, original)


@pytest.mark.parametrize(
    ("selected", "use_vector_potential", "expected_calls"),
    [
        (set(), False, 0),
        ({"gauge"}, True, 0),
        ({"divergence"}, False, 1),
        ({"force_free", "potential", "dB_dt"}, False, 1),
        ({"induction"}, False, 2),
        ({"induction"}, True, 0),
    ],
)
def test_only_requested_derivatives_are_evaluated(
    monkeypatch,
    selected,
    use_vector_potential,
    expected_calls,
):
    calls = []
    original_jacobian = physics.jacobian

    def counting_jacobian(*args, **kwargs):
        calls.append(args[0])
        return original_jacobian(*args, **kwargs)

    monkeypatch.setattr(physics, "jacobian", counting_jacobian)
    coords = _coords()
    x, y, z, t = coords.unbind(dim=-1)
    b = torch.stack((t + x, y, z), dim=-1)
    v = torch.stack((torch.ones_like(t), 0 * t, 0 * t), dim=-1)
    a_jacobian = (
        torch.zeros((coords.shape[0], 3, 4), dtype=DTYPE)
        if use_vector_potential
        else None
    )

    PhysicsConstraintModule()(
        b,
        v,
        coords,
        selected=selected,
        a_jac_matrix=a_jacobian,
    )

    assert len(calls) == expected_calls


def test_selection_validation_rejects_unknown_and_missing_vector_potential():
    coords = _coords()
    b = coords[:, :3]
    v = _zero_velocity(coords)

    with pytest.raises(ValueError, match="Unknown physics constraints"):
        PhysicsConstraintModule()(b, v, coords, {"not_a_constraint"})
    with pytest.raises(ValueError, match="gauge constraint requires"):
        PhysicsConstraintModule()(b, v, coords, {"gauge"})




def test_fixed_schedule_configuration():
    fixed = PhysicsWeightSchedule.from_config(0.25)
    configured = PhysicsWeightSchedule.from_config({"type": "fixed", "value": 0.4})

    assert fixed.value_at(0) == pytest.approx(0.25)
    assert fixed.value_at(1_000_000) == pytest.approx(0.25)
    assert configured.schedule_type == "fixed"
    assert configured.value_at(10) == pytest.approx(0.4)
    assert configured.configuration() == {"type": "fixed", "start": 0.4, "end": 0.4}


def test_scheduled_weights_are_stateless_and_obey_warmup_boundaries():
    linear = PhysicsWeightSchedule.from_config(
        {
            "type": "linear",
            "start": 1.0,
            "end": 3.0,
            "iterations": 4,
            "warmup_iterations": 2,
        }
    )
    exponential = PhysicsWeightSchedule.from_config(
        {
            "type": "exponential",
            "start": 1.0,
            "end": 16.0,
            "iterations": 4,
            "warmup_iterations": 1,
        }
    )
    step = PhysicsWeightSchedule.from_config(
        {
            "type": "step",
            "start": 0.1,
            "end": 1.0,
            "iterations": 3,
            "warmup_iterations": 2,
        }
    )
    smooth = PhysicsWeightSchedule.from_config(
        {
            "type": "smoothstep",
            "start": 0.0,
            "end": 1.0,
            "iterations": 4,
            "warmup_iterations": 2,
        }
    )

    assert [linear.value_at(step_) for step_ in (6, 0, 2, 4, 6)] == pytest.approx(
        [3.0, 1.0, 1.0, 2.0, 3.0]
    )
    assert exponential.value_at(1) == pytest.approx(1.0)
    assert exponential.value_at(3) == pytest.approx(4.0)
    assert exponential.value_at(5) == pytest.approx(16.0)
    assert step.value_at(4) == pytest.approx(0.1)
    assert step.value_at(5) == pytest.approx(1.0)
    assert smooth.value_at(2) == pytest.approx(0.0)
    assert smooth.value_at(4) == pytest.approx(0.5)
    assert smooth.value_at(6) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (-1.0, "nonnegative"),
        (
            {"type": "linear", "start": 0, "end": 1, "iterations": 0},
            "positive iterations",
        ),
        ({"type": "linear", "start": -1, "end": 1, "iterations": 2}, "nonnegative"),
        (
            {"type": "linear", "start": 0, "end": 1, "iterations": 2.5},
            "nonnegative integer",
        ),
        (
            {"type": "linear", "start": 0, "end": 1, "iterations": 2, "warmup_iterations": -0.5},
            "nonnegative integer",
        ),
        (
            {"type": "exponential", "start": 0, "end": 1, "iterations": 2},
            "strictly positive",
        ),
        ({"type": "fixed", "start": 1, "end": 2}, "must match"),
        ({"type": "mystery", "start": 0, "end": 1, "iterations": 2}, "Unknown"),
    ],
)
def test_invalid_weight_schedules_are_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        PhysicsWeightSchedule.from_config(config)


@pytest.mark.parametrize("global_step", [-1, 0.5, math.inf, True])
def test_schedule_rejects_invalid_global_steps(global_step):
    with pytest.raises(ValueError, match="nonnegative integer"):
        PhysicsWeightSchedule.from_config(1.0).value_at(global_step)


@pytest.mark.parametrize('rs_per_ds,gauss_per_db', [(1.0, 1000.0), (0.2, 250.0)])
def test_spherical_dipole_from_vector_potential_is_solenoidal_and_force_free(rs_per_ds, gauss_per_db):
    """A physical dipole in the requested shell checks second derivatives and units."""
    coords = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.05, 0.0, 0.0],
        [0.0, 0.0, 1.1, 0.0],
        [0.6, 0.6, 0.6, 0.0],
    ], dtype=DTYPE) / rs_per_ds
    coords.requires_grad_()
    x, y, z = (coords[:, :3] * rs_per_ds).unbind(-1)
    radius = torch.sqrt(x.square() + y.square() + z.square())
    # A = m x r / r^3; moment chosen so the equatorial surface field is 1000 G.
    a = torch.stack((-y, x, 0 * z), dim=-1) / radius[:, None].pow(3)
    a = a * (1000 / gauss_per_db / rs_per_ds)
    a_jac = physics.jacobian(a, coords)
    b = physics._curl_from_jacobian(a_jac)
    expected = torch.stack((3*x*z, 3*y*z, 3*z.square() - radius.square()), dim=-1)
    expected = expected / radius[:, None].pow(5) * (1000 / gauss_per_db)
    torch.testing.assert_close(b, expected)
    result = PhysicsConstraintModule(
        normalization='characteristic',
        characteristic_scales=PhysicsCharacteristicScales(
            10e6 / (695700000 * rs_per_ds), 1000/3600, 1000/gauss_per_db,
            1000*3600/(695700000 * rs_per_ds),
        ),
    )(b, _zero_velocity(coords), coords, {'divergence', 'force_free', 'potential'})
    for loss in result.losses.values():
        torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-28, rtol=0)
