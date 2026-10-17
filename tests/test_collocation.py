import math

import pytest
import torch

from spinnme.train.collocation import (
    LongitudeInterval,
    SmoothSphericalCollocationSampler,
    SphericalCollocationConfig,
    SphericalCollocationDomain,
)


def _domain(**kwargs):
    defaults = {
        "normalized_time_range": (-0.25, 0.75),
        "latitude_range": (-math.pi / 3, math.pi / 4),
        "longitude": LongitudeInterval(center=0.4, width=1.2),
        "normalized_radius_range": (1.0, 1.01),
        "surface_radius": 1.0,
    }
    defaults.update(kwargs)
    return SphericalCollocationDomain(**defaults)


def test_longitude_shortest_arc_is_seam_safe():
    longitude = LongitudeInterval.shortest_arc(math.radians(179), math.radians(-179))
    lower, upper = longitude.bounds

    assert math.degrees(longitude.width) == pytest.approx(2.0)
    assert math.degrees(lower) == pytest.approx(179.0)
    assert math.degrees(upper) == pytest.approx(181.0)

    sampler = SmoothSphericalCollocationSampler(
        _domain(longitude=longitude),
        SphericalCollocationConfig(global_size=4096, seed=3),
    )
    batch = sampler.sample(0, requires_grad=False)
    wrapped = torch.atan2(
        torch.sin(batch.longitude_unwrapped), torch.cos(batch.longitude_unwrapped)
    )

    assert torch.all(torch.abs(torch.rad2deg(wrapped)) >= 179.0)


def test_sampling_is_stateless_and_deterministic_by_global_step():
    sampler = SmoothSphericalCollocationSampler(
        _domain(),
        SphericalCollocationConfig(global_size=127, seed=23),
    )

    first = sampler.sample(12, requires_grad=False)
    repeated = sampler.sample(12, requires_grad=False)
    next_step = sampler.sample(13, requires_grad=False)

    torch.testing.assert_close(first.coords, repeated.coords, rtol=0, atol=0)
    assert not torch.equal(first.coords, next_step.coords)
    assert first.coords.is_leaf


def test_fixed_global_set_partitions_exactly_across_ranks():
    sampler = SmoothSphericalCollocationSampler(
        _domain(),
        SphericalCollocationConfig(global_size=17, seed=5),
    )
    single = sampler.sample(7, rank=0, world_size=1, requires_grad=False)
    distributed = [
        sampler.sample(7, rank=rank, world_size=3, requires_grad=False)
        for rank in range(3)
    ]

    assert [batch.local_size for batch in distributed] == [5, 6, 6]
    torch.testing.assert_close(
        torch.cat([batch.coords for batch in distributed]),
        single.coords,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.cat([batch.global_indices for batch in distributed]),
        torch.arange(17),
        rtol=0,
        atol=0,
    )


def test_ddp_scaled_local_losses_average_to_global_mean():
    sampler = SmoothSphericalCollocationSampler(
        _domain(),
        SphericalCollocationConfig(global_size=17, seed=11),
    )
    single = sampler.sample(4, rank=0, world_size=1, requires_grad=False)
    expected = single.coords.square().sum(dim=-1).mean()

    local_losses = []
    for rank in range(3):
        batch = sampler.sample(4, rank=rank, world_size=3, requires_grad=False)
        per_point = batch.coords.square().sum(dim=-1)
        local_losses.append(batch.ddp_scaled_mean(per_point))

    torch.testing.assert_close(torch.stack(local_losses).mean(), expected)


def test_default_coordinates_are_leaf_tensors_with_finite_gradients():
    sampler = SmoothSphericalCollocationSampler(
        _domain(),
        SphericalCollocationConfig(global_size=32),
    )
    batch = sampler.sample(0)

    assert batch.coords.is_leaf
    assert batch.coords.requires_grad
    batch.ddp_scaled_mean(batch.coords.square().sum(dim=-1)).backward()
    assert batch.coords.grad is not None
    assert torch.isfinite(batch.coords.grad).all()


def test_latitude_is_equal_area_and_radius_is_uniform_in_volume():
    domain = _domain(
        latitude_range=(-math.pi / 2, math.pi / 2),
        normalized_radius_range=(1.0, 2.0),
    )
    sampler = SmoothSphericalCollocationSampler(
        domain,
        SphericalCollocationConfig(
            global_size=65536, seed=101, radial_measure="volume"
        ),
    )
    batch = sampler.sample(2, dtype=torch.float64, requires_grad=False)
    sin_latitude = torch.sin(batch.latitude)
    radius_cubed = batch.normalized_radius.pow(3)

    assert sin_latitude.mean().abs() < 2e-5
    assert abs(sin_latitude.square().mean().item() - 1 / 3) < 2e-5
    assert abs(radius_cubed.mean().item() - 4.5) < 1e-4


def test_surface_shell_supports_nonunit_model_surface_radius():
    domain = SphericalCollocationDomain.surface_shell(
        normalized_time_range=(0, 0),
        latitude_range=(-0.2, 0.2),
        longitude=LongitudeInterval.from_unwrapped_bounds(3.0, 3.2),
        surface_radius=2.5,
        shell_height=0.025,
    )
    sampler = SmoothSphericalCollocationSampler(
        domain,
        SphericalCollocationConfig(global_size=1024, seed=0),
    )
    batch = sampler.sample(0, requires_grad=False)

    assert domain.normalized_radius_range == pytest.approx((2.5, 2.525))
    assert torch.all(batch.normalized_radius >= 2.5)
    assert torch.all(batch.normalized_radius <= 2.525)
    torch.testing.assert_close(
        torch.linalg.vector_norm(batch.coords[:, :3], dim=-1),
        batch.normalized_radius,
    )
    torch.testing.assert_close(batch.coords[:, 3], torch.zeros(1024))


def test_shell_can_straddle_the_model_surface():
    domain = _domain(normalized_radius_range=(0.99, 1.01))
    sampler = SmoothSphericalCollocationSampler(
        domain,
        SphericalCollocationConfig(global_size=1024, seed=0),
    )
    batch = sampler.sample(0, requires_grad=False)

    assert torch.all(batch.normalized_radius >= 0.99)
    assert torch.all(batch.normalized_radius <= 1.01)
    assert torch.any(batch.normalized_radius < domain.surface_radius)
    assert torch.any(batch.normalized_radius > domain.surface_radius)


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: LongitudeInterval(center=0, width=0), "longitude width"),
        (
            lambda: LongitudeInterval(center=0, width=2 * math.pi + 0.1),
            "longitude width",
        ),
        (lambda: _domain(latitude_range=(-2, 0)), "latitude_range"),
        (lambda: _domain(normalized_radius_range=(0.8, 0.9)), "surface_radius"),
        (lambda: SphericalCollocationConfig(global_size=0), "global_size"),
        (lambda: SphericalCollocationConfig(seed=-1), "seed"),
        (
            lambda: SphericalCollocationConfig(radial_measure="unknown"),
            "radial_measure",
        ),
    ],
)
def test_invalid_configuration_is_rejected(factory, match):
    with pytest.raises((TypeError, ValueError), match=match):
        factory()


def test_partition_validation_is_actionable():
    sampler = SmoothSphericalCollocationSampler(
        _domain(),
        SphericalCollocationConfig(global_size=2),
    )
    with pytest.raises(ValueError, match="at least world_size"):
        sampler.sample(0, rank=0, world_size=3)
    with pytest.raises(ValueError, match="rank must be in"):
        sampler.sample(0, rank=3, world_size=3)
