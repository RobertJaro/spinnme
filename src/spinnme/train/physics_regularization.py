"""Composition of continuous physics constraints with the Stokes objective.

This module owns the small amount of orchestration between an explicit global
collocation domain, the collocation sampler, constraint residuals, and their
weight schedules.  It intentionally knows nothing about the Stokes forward
model or data-loader batches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from spinnme.train.collocation import (
    LongitudeInterval,
    SmoothSphericalCollocationSampler,
    SphericalCollocationConfig,
    SphericalCollocationDomain,
)
from spinnme.train.physics import (
    CONSTRAINT_NAMES,
    PhysicsCharacteristicScales,
    PhysicsConstraintModule,
    PhysicsWeightSchedule,
)

STOKES_COMPONENTS = ("I", "Q", "U", "V")
PHYSICS_CONSTRAINTS = frozenset(CONSTRAINT_NAMES)


@dataclass(frozen=True)
class PhysicsRegularizationResult:
    """The rank-local DDP-scaled physics objective and logging metrics."""

    total: torch.Tensor
    metrics: dict[str, torch.Tensor]


def split_stokes_and_physics_config(
    weights_config: Mapping[str, Any] | None,
    physics_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate separate Stokes weights and continuous physics configuration."""
    weights_config = copy.deepcopy(dict(weights_config or {}))
    unknown = set(weights_config).difference(STOKES_COMPONENTS)
    if unknown:
        raise ValueError(
            f"Unknown weights keys {sorted(unknown)}; expected Stokes components "
            f"{STOKES_COMPONENTS}. Put physics weights in physics.constraints."
        )
    return weights_config, (
        None if physics_config is None else copy.deepcopy(dict(physics_config))
    )


def _minimum_covering_longitude(
    bounds: Sequence[Mapping[str, Any]],
) -> LongitudeInterval:
    """Return the smallest circular arc containing every dataset interval."""
    two_pi = 2 * math.pi
    segments = []
    for item in bounds:
        offset_min, offset_max = map(float, item["longitude_offset_rad"])
        width = offset_max - offset_min
        if width < 0 or width > two_pi + 1e-12:
            raise ValueError(f"Invalid dataset longitude width: {width} rad.")
        if width >= two_pi - 1e-12:
            return LongitudeInterval(center=0.0, width=two_pi)

        start = (float(item["longitude_center_rad"]) + offset_min) % two_pi
        stop = start + width
        if stop <= two_pi:
            segments.append((start, stop))
        else:
            segments.extend(((start, two_pi), (0.0, stop - two_pi)))

    segments.sort()
    merged = []
    for start, stop in segments:
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    if not merged:
        raise ValueError("The training datasets contain no longitude coverage.")

    gaps = [
        (merged[index][1], merged[index + 1][0]) for index in range(len(merged) - 1)
    ]
    gaps.append((merged[-1][1], merged[0][0] + two_pi))
    gap_start, gap_stop = max(gaps, key=lambda gap: gap[1] - gap[0])
    uncovered_width = gap_stop - gap_start
    if uncovered_width <= 1e-12:
        return LongitudeInterval(center=0.0, width=two_pi)

    covering_width = two_pi - uncovered_width
    if covering_width <= 1e-12:
        raise ValueError(
            "The combined training longitude range must have nonzero width."
        )
    covering_start = gap_stop % two_pi
    return LongitudeInterval(
        center=covering_start + covering_width / 2,
        width=covering_width,
    )


def build_spherical_collocation_domain(
    train_datasets: Sequence,
    Rs_per_ds: float,
    radius_range_Rs: Sequence[float] | None = None,
) -> SphericalCollocationDomain:
    """Build one seam-safe global domain from all TRAIN acquisitions."""
    if not train_datasets:
        raise ValueError(
            "Physics regularization requires at least one training dataset."
        )
    Rs_per_ds = float(Rs_per_ds)
    if not math.isfinite(Rs_per_ds) or Rs_per_ds <= 0:
        raise ValueError("Rs_per_ds must be finite and positive.")

    missing_bounds = [
        getattr(dataset, "ds_id", str(index))
        for index, dataset in enumerate(train_datasets)
        if not getattr(dataset, "physics_bounds", None)
    ]
    if missing_bounds:
        raise RuntimeError(
            "The cached training data predates global physics-domain metadata for "
            f"datasets {missing_bounds[:5]}. Rebuild it with --reload or use a fresh "
            "work_directory before enabling physics regularization."
        )
    bounds = [copy.deepcopy(dataset.physics_bounds) for dataset in train_datasets]
    longitude = _minimum_covering_longitude(bounds)

    latitude_min = min(item["latitude_rad"][0] for item in bounds)
    latitude_max = max(item["latitude_rad"][1] for item in bounds)
    if latitude_max <= latitude_min:
        raise ValueError(
            "The combined training latitude range must have nonzero width."
        )
    time_min = min(item.get("normalized_time_range", [item["normalized_time"]] * 2)[0] for item in bounds)
    time_max = max(item.get("normalized_time_range", [item["normalized_time"]] * 2)[1] for item in bounds)

    if radius_range_Rs is None:
        radius_min_Rs = min(item["radius_Rs"][0] for item in bounds)
        radius_max_Rs = max(item["radius_Rs"][1] for item in bounds)
    else:
        if len(radius_range_Rs) != 2:
            raise ValueError("physics.radius_range_Rs must contain exactly two values.")
        radius_min_Rs, radius_max_Rs = map(float, radius_range_Rs)
    if not (math.isfinite(radius_min_Rs) and math.isfinite(radius_max_Rs)):
        raise ValueError("physics.radius_range_Rs values must be finite.")
    if radius_min_Rs <= 0 or radius_max_Rs < radius_min_Rs:
        raise ValueError("physics.radius_range_Rs must be positive and ordered.")

    return SphericalCollocationDomain(
        normalized_time_range=(time_min, time_max),
        latitude_range=(latitude_min, latitude_max),
        longitude=longitude,
        normalized_radius_range=(radius_min_Rs / Rs_per_ds, radius_max_Rs / Rs_per_ds),
        surface_radius=1.0 / Rs_per_ds,
    )


class PhysicsRegularization:
    """Evaluate scheduled constraints at independent continuous coordinates."""

    _CONFIG_KEYS: ClassVar[set[str]] = {
        "constraints",
        "num_points",
        "seed",
        "radial_measure",
        "normalization",
        "epsilon",
        "domain",
        "radius_range_Rs",
        "characteristic_units",
    }

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        domain: SphericalCollocationDomain | None,
        *,
        vector_potential: bool,
        gauss_per_dB: float | None = None,
        meters_per_ds: float | None = None,
        seconds_per_dt: float | None = None,
    ):
        self._enabled = config is not None
        if not self._enabled:
            self.config = None
            self.domain = None
            self.sampler = None
            self.constraint_module = None
            self.schedules = {}
            return

        config = copy.deepcopy(dict(config))
        unknown = set(config).difference(self._CONFIG_KEYS)
        if unknown:
            raise ValueError(f"Unknown physics configuration keys: {sorted(unknown)}.")
        if config.get("domain", "full_train") != "full_train":
            raise ValueError("physics.domain currently supports only 'full_train'.")
        if domain is None:
            raise ValueError(
                "Physics regularization requires an explicit global training domain."
            )

        constraints = config.get("constraints")
        if not isinstance(constraints, Mapping) or not constraints:
            raise ValueError(
                "physics.constraints must contain at least one weighted constraint."
            )
        unknown_constraints = set(constraints).difference(PHYSICS_CONSTRAINTS)
        if unknown_constraints:
            raise ValueError(
                f"Unknown physics constraints: {sorted(unknown_constraints)}."
            )
        if "gauge" in constraints and not vector_potential:
            raise ValueError(
                "The gauge constraint requires model.vector_potential: true."
            )
        if vector_potential and "divergence" in constraints:
            warnings.warn(
                "divergence is analytically redundant when B=curl(A); the sampled "
                "penalty only measures automatic-differentiation roundoff.",
                UserWarning,
                stacklevel=2,
            )
        if domain.normalized_time_range[0] == domain.normalized_time_range[1] and {
            "induction",
            "dB_dt",
        }.intersection(constraints):
            raise ValueError(
                "Temporal physics constraints require more than one training time."
            )

        self.domain = domain
        self.schedules = {
            name: PhysicsWeightSchedule.from_config(weight)
            for name, weight in constraints.items()
        }
        sampler_config = SphericalCollocationConfig(
            global_size=config.get("num_points", 4096),
            seed=config.get("seed", 0),
            radial_measure=config.get("radial_measure", "volume"),
        )
        self.sampler = SmoothSphericalCollocationSampler(domain, sampler_config)
        normalization = config.get("normalization", "raw").lower()
        characteristic_units = None
        characteristic_scales = None
        if normalization == "characteristic":
            characteristic_units = {
                "length_Mm": 1.0,
                "magnetic_field_G": 10.0,
                "velocity_km_s": 1.0,
                "time_s": 1000.0,
            }
            supplied_units = config.get("characteristic_units", {})
            if not isinstance(supplied_units, Mapping):
                raise ValueError("physics.characteristic_units must be a mapping.")
            unknown_units = set(supplied_units).difference(characteristic_units)
            if unknown_units:
                raise ValueError(f"Unknown characteristic units: {sorted(unknown_units)}.")
            characteristic_units.update(supplied_units)
            for name, value in characteristic_units.items():
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Characteristic unit {name} must be finite and positive.")
                characteristic_units[name] = float(value)
            for name, value in (("gauss_per_dB", gauss_per_dB),
                                ("meters_per_ds", meters_per_ds),
                                ("seconds_per_dt", seconds_per_dt)):
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Characteristic normalization requires a finite positive {name}.")
            characteristic_scales = PhysicsCharacteristicScales(
                length=characteristic_units["length_Mm"] * 1e6 / meters_per_ds,
                time=characteristic_units["time_s"] / seconds_per_dt,
                magnetic_field=characteristic_units["magnetic_field_G"] / gauss_per_dB,
                velocity=characteristic_units["velocity_km_s"] * 1e3 * seconds_per_dt / meters_per_ds,
            )
        elif "characteristic_units" in config:
            raise ValueError("physics.characteristic_units requires normalization: characteristic.")
        self.constraint_module = PhysicsConstraintModule(
            normalization=normalization,
            epsilon=float(config.get("epsilon", 1e-6)),
            characteristic_scales=characteristic_scales,
        )
        radius_range_Rs = config.get("radius_range_Rs")
        self.config = {
            "domain": "full_train",
            "num_points": sampler_config.global_size,
            "seed": sampler_config.seed,
            "radial_measure": sampler_config.radial_measure,
            "normalization": self.constraint_module.normalization,
            "epsilon": self.constraint_module.epsilon,
            "radius_range_Rs": None
            if radius_range_Rs is None
            else list(radius_range_Rs),
            "constraints": {
                name: schedule.configuration()
                for name, schedule in self.schedules.items()
            },
        }
        if characteristic_units is not None:
            self.config["characteristic_units"] = characteristic_units

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configuration(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        metadata = {
            **copy.deepcopy(self.config),
            "collocation_domain": self.domain.canonical_metadata(),
        }
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        metadata["fingerprint"] = hashlib.sha256(payload).hexdigest()
        return metadata

    def evaluate(
        self,
        parameter_model,
        *,
        global_step: int,
        reference: torch.Tensor,
    ) -> PhysicsRegularizationResult:
        zero = reference.new_zeros(())
        if not self.enabled:
            return PhysicsRegularizationResult(zero, {})

        weights = {
            name: schedule.value_at(global_step)
            for name, schedule in self.schedules.items()
        }
        active = tuple(name for name, weight in weights.items() if weight > 0)
        if not active:
            return PhysicsRegularizationResult(zero, {"physics.loss": zero})

        collocation = self.sampler.sample(
            int(global_step),
            device=reference.device,
            dtype=reference.dtype,
        )
        output = parameter_model(collocation.coords)
        b = torch.cat([output["b_x"], output["b_y"], output["b_z"]], dim=-1)
        v = torch.cat([output["v_x"], output["v_y"], output["v_z"]], dim=-1)
        result = self.constraint_module(
            b=b,
            v=v,
            coords=collocation.coords,
            selected=active,
            a_jac_matrix=output.get("a_jac_matrix"),
        )

        total = zero
        metrics = {}
        for name in active:
            loss = collocation.ddp_scaled_mean(result.losses[name])
            weighted = loss * weights[name]
            total = total + weighted
            metrics[f"physics.{name}"] = loss
        if not torch.isfinite(total):
            raise RuntimeError("Encountered a non-finite physics regularization loss.")
        metrics["physics.loss"] = total
        return PhysicsRegularizationResult(total, metrics)
