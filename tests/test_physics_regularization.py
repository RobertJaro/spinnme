import copy
import math

import pytest
import torch

from spinnme.train.collocation import LongitudeInterval, SphericalCollocationDomain
from spinnme.train.physics_regularization import (
    PhysicsRegularization,
    build_spherical_collocation_domain,
    split_stokes_and_physics_config,
)
from spinnme.train.me_spherical_module import MESphericalModule


class _Dataset:
    def __init__(self, *, time, center_deg, half_width_deg=0.5):
        center = math.radians(center_deg)
        half_width = math.radians(half_width_deg)
        n_points = 100
        self.physics_bounds = {
            "radius_Rs": (1.0, 1.0),
            "latitude_rad": (-0.2, 0.3),
            "longitude_center_rad": center,
            "longitude_offset_rad": (-half_width, half_width),
            "longitude_vector_sum": (
                math.sin(center) * n_points,
                math.cos(center) * n_points,
            ),
            "n_points": n_points,
            "normalized_time": time,
        }


class _DivergentModel(torch.nn.Module):
    vector_potential = False

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, coords):
        self.calls += 1
        x, y, z, _ = coords.unbind(dim=-1)
        zero = 0 * x
        return {
            "b_x": (self.scale * x)[:, None],
            "b_y": (self.scale * y)[:, None],
            "b_z": (self.scale * z)[:, None],
            "v_x": zero[:, None],
            "v_y": zero[:, None],
            "v_z": zero[:, None],
            "a_jac_matrix": None,
        }


def _domain():
    return SphericalCollocationDomain(
        normalized_time_range=(0.0, 1.0),
        latitude_range=(-0.2, 0.3),
        longitude=LongitudeInterval(center=0.0, width=0.5),
        normalized_radius_range=(1.0, 1.0),
        surface_radius=1.0,
    )


def _inversion_module(physics_config):
    return MESphericalModule(
        image_shape=(1, 1),
        wavelength_config={},
        gauss_per_dB=1000,
        Rs_per_ds=1,
        seconds_per_dt=36000,
        instrument_config=[],
        model_config={
            "type": "mlp",
            "dim": 4,
            "n_layers": 1,
            "encoding_config": {"type": "identity"},
        },
        physics_config=physics_config,
        physics_domain=_domain() if physics_config is not None else None,
    )


def test_global_domain_combines_all_training_times_and_handles_longitude_seam():
    domain = build_spherical_collocation_domain(
        [_Dataset(time=-0.1, center_deg=179), _Dataset(time=0.2, center_deg=-179)],
        Rs_per_ds=0.5,
        radius_range_Rs=(1.0, 1.0005),
    )

    assert domain.normalized_time_range == pytest.approx((-0.1, 0.2))
    assert domain.latitude_range == pytest.approx((-0.2, 0.3))
    assert math.degrees(domain.longitude.width) == pytest.approx(3.0)
    assert abs(abs(math.degrees(domain.longitude.wrapped_center)) - 180) < 1e-10
    assert domain.surface_radius == pytest.approx(2.0)
    assert domain.normalized_radius_range == pytest.approx((2.0, 2.001))


def test_global_domain_uses_minimum_circular_cover_for_separated_footprints():
    datasets = [
        _Dataset(time=index, center_deg=center)
        for index, center in enumerate((-126.13, -81.27, 127.02, 33.56))
    ]

    domain = build_spherical_collocation_domain(datasets, Rs_per_ds=1)

    # The largest uncovered gap is 113.83 degrees after accounting for each
    # footprint's one-degree width; its complement is the minimum cover.
    assert math.degrees(domain.longitude.width) == pytest.approx(246.17)


def test_physics_domain_rejects_stale_cached_datasets_without_bounds():
    stale_dataset = type("StaleDataset", (), {"ds_id": "old-cache"})()

    with pytest.raises(RuntimeError, match="--reload"):
        build_spherical_collocation_domain([stale_dataset], Rs_per_ds=1)


def test_stokes_weights_and_physics_constraints_are_separate():
    with pytest.raises(ValueError, match="physics.constraints"):
        split_stokes_and_physics_config(
            {"I": 2.0, "divergence": 1e-4}, {"num_points": 32},
        )
    config = {"num_points": 32, "constraints": {"divergence": 1e-4}}
    stokes, physics = split_stokes_and_physics_config({"I": 2.0}, config)
    assert stokes == {"I": 2.0}
    assert physics == config
    assert physics is not config


def test_regularization_skips_zero_weight_and_adds_weighted_divergence():
    regularization = PhysicsRegularization(
        {
            "num_points": 64,
            "constraints": {
                "divergence": {
                    "type": "linear",
                    "start": 0.0,
                    "end": 2.0,
                    "iterations": 10,
                },
            },
        },
        _domain(),
        vector_potential=False,
    )
    model = _DivergentModel()
    reference = torch.tensor(4.0, requires_grad=True)

    initial = regularization.evaluate(model, global_step=0, reference=reference)
    assert model.calls == 0
    assert initial.total.item() == 0
    assert initial.metrics == {"physics.loss": initial.total}

    midpoint = regularization.evaluate(model, global_step=5, reference=reference)
    assert model.calls == 1
    # div(x,y,z)=3, so raw divergence loss is 9 and the midpoint weight is 1.
    assert midpoint.metrics["physics.divergence"].item() == pytest.approx(9.0)
    assert midpoint.metrics["physics.loss"].item() == pytest.approx(9.0)
    assert set(midpoint.metrics) == {"physics.divergence", "physics.loss"}
    assert midpoint.total.item() == pytest.approx(9.0)
    midpoint.total.backward()
    assert reference.grad is None
    assert model.scale.grad.item() == pytest.approx(18.0)


def test_physics_configuration_is_canonical_and_includes_domain_fingerprint():
    config = {
        "num_points": 64,
        "seed": 7,
        "radius_range_Rs": [1.0, 1.0],
        "constraints": {"divergence": 1e-6},
    }
    first = PhysicsRegularization(config, _domain(), vector_potential=False)
    second = PhysicsRegularization(
        copy.deepcopy(config), _domain(), vector_potential=False
    )

    assert first.configuration() == second.configuration()
    assert len(first.configuration()["fingerprint"]) == 64


def test_characteristic_units_are_converted_and_checkpointed():
    config = {
        "normalization": "characteristic",
        "constraints": {"divergence": 1e-6, "potential": 1e-6},
    }
    module = _inversion_module(config)
    scales = module.physics_regularization.constraint_module.characteristic_scales
    assert scales.length == pytest.approx(1e6 / module.meters_per_ds)
    assert scales.time == pytest.approx(1000 / 36000)
    assert scales.magnetic_field == pytest.approx(0.01)
    assert scales.velocity == pytest.approx(1000 * 36000 / module.meters_per_ds)
    assert scales.induction_factor == pytest.approx(1.0)
    assert module.physics_config["characteristic_units"] == {
        "length_Mm": 1.0, "time_s": 1000.0,
        "magnetic_field_G": 10.0, "velocity_km_s": 1.0,
    }
    for schedule in module.physics_regularization.schedules.values():
        assert schedule.value_at(0) == schedule.value_at(100000) == 1e-6
    checkpoint = {"state_dict": copy.deepcopy(module.state_dict())}
    module.on_save_checkpoint(checkpoint)
    _inversion_module(config).on_load_checkpoint(copy.deepcopy(checkpoint))


@pytest.mark.parametrize("units,match", [
    ({"length_Mm": 0}, "finite and positive"),
    ({"time_s": float("nan")}, "finite and positive"),
    ({"velocity_km_s": True}, "finite and positive"),
    ({"unknown": 1}, "Unknown characteristic units"),
])
def test_invalid_characteristic_units_are_rejected(units, match):
    with pytest.raises(ValueError, match=match):
        _inversion_module({
            "normalization": "characteristic", "characteristic_units": units,
            "constraints": {"divergence": 1e-6},
        })


def test_characteristic_normalization_requires_dataset_units():
    with pytest.raises(ValueError, match="positive gauss_per_dB"):
        PhysicsRegularization({
            "normalization": "characteristic", "constraints": {"divergence": 1e-6},
        }, _domain(), vector_potential=False)


def test_characteristic_units_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="requires normalization"):
        _inversion_module({
            "normalization": "relative", "characteristic_units": {"length_Mm": 1},
            "constraints": {"divergence": 1e-6},
        })


def test_checkpoint_keeps_current_physics_or_domain_switches():
    config = {
        "num_points": 64,
        "seed": 7,
        "radius_range_Rs": [1.0, 1.0],
        "constraints": {"divergence": 1e-6},
    }
    original = _inversion_module(config)
    checkpoint = {"state_dict": copy.deepcopy(original.state_dict())}
    original.on_save_checkpoint(checkpoint)

    _inversion_module(copy.deepcopy(config)).on_load_checkpoint(
        copy.deepcopy(checkpoint)
    )
    changed = copy.deepcopy(config)
    changed["seed"] = 8
    changed["constraints"]["divergence"] = 1e-4
    changed_module = _inversion_module(changed)
    changed_checkpoint = copy.deepcopy(checkpoint)
    with pytest.warns(UserWarning, match="current configuration will replace"):
        changed_module.on_load_checkpoint(changed_checkpoint)
    assert changed_checkpoint["spinnme_physics_config"] == changed_module.physics_config
    assert changed_checkpoint["spinnme_physics_config"]["seed"] == 8
    assert changed_checkpoint["spinnme_physics_config"]["constraints"]["divergence"] == {
        "type": "fixed",
        "start": 1e-4,
        "end": 1e-4,
    }

    disabled_module = _inversion_module(None)
    disabled_checkpoint = copy.deepcopy(checkpoint)
    with pytest.warns(UserWarning, match="current configuration will replace"):
        disabled_module.on_load_checkpoint(disabled_checkpoint)
    assert disabled_checkpoint["spinnme_physics_config"] is None


@pytest.mark.parametrize(
    ("physics_config", "match"),
    [
        ({}, "constraints"),
        ({"constraints": {"unknown": 1}}, "Unknown physics constraints"),
        ({"constraints": {"gauge": 1}}, "gauge constraint requires"),
        ({"constraints": {"divergence": -1}}, "nonnegative"),
    ],
)
def test_invalid_physics_configuration_is_rejected(physics_config, match):
    with pytest.raises(ValueError, match=match):
        PhysicsRegularization(physics_config, _domain(), vector_potential=False)


def test_scanned_raster_physics_domain_includes_pixel_time_range():
    dataset = _Dataset(time=0., center_deg=0.)
    dataset.physics_bounds['normalized_time_range'] = [-.25, .75]
    domain = build_spherical_collocation_domain([dataset], 1.)
    assert domain.normalized_time_range == (-.25, .75)


def test_inversion_and_shell_force_free_loss_backpropagate_through_vector_potential():
    from _training_helpers import _physical_batch
    from spinnme.instrument import hmi_wavelength_config
    from spinnme.train.physics import PhysicsConstraintModule

    domain = build_spherical_collocation_domain(
        [_Dataset(time=0., center_deg=0.)], Rs_per_ds=1., radius_range_Rs=(1., 1.1),
    )
    module = MESphericalModule(
        image_shape=(4, 4), wavelength_config={'HMI': hmi_wavelength_config()},
        instrument_config=[{'type': 'hmi', 'instrument_id': 'HMI'}],
        gauss_per_dB=1000, Rs_per_ds=1, seconds_per_dt=3600,
        model_config={'type': 'mlp', 'dim': 8, 'n_layers': 2, 'vector_potential': True,
                      'encoding_config': {'type': 'identity'}},
        stokes_loss_config={'continuum_normalization': True},
        normalization_config={'asinh_alphas': {'Q': .01, 'U': .01, 'V': .01}},
        physics_domain=domain,
        physics_config={'num_points': 32, 'radius_range_Rs': [1., 1.1],
                        'normalization': 'characteristic',
                        'characteristic_units': {'length_Mm': 10., 'magnetic_field_G': 1000.},
                        'constraints': {'force_free': 1.}},
    )
    physics_result = module.physics_regularization.evaluate(
        module.parameter_model, global_step=0, reference=torch.tensor(0.),
    )
    assert physics_result.total > 0
    physics_grads = torch.autograd.grad(physics_result.total, module.parameter_model.out_layer.weight)[0]
    assert torch.isfinite(physics_grads).all() and physics_grads.abs().sum() > 0
    loss = module.training_step(_physical_batch(), 0)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    points = module.physics_regularization.sampler.sample(0)
    assert (points.normalized_radius >= 1.).all() and (points.normalized_radius <= 1.1).all()
    output = module.parameter_model(points.coords)
    b = torch.cat([output[f'b_{axis}'] for axis in 'xyz'], dim=-1)
    divergence = PhysicsConstraintModule()(b, b*0, points.coords, {'divergence'}).losses['divergence']
    assert divergence.max() < 1e-12
