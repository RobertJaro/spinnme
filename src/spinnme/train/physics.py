"""Physics constraints used by spherical Milne--Eddington inversions.

The public :class:`PhysicsConstraintModule` evaluates only the derivatives that
the requested constraints need.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, ClassVar

import torch
from torch import nn

from spinnme.data.util import cartesian_to_spherical
from spinnme.model import jacobian

CONSTRAINT_NAMES = (
    "induction",
    "divergence",
    "force_free",
    "potential",
    "dB_dt",
    "gauge",
)


@dataclass(frozen=True)
class PhysicsConstraintResult:
    """Unreduced constraint losses and descriptive residual diagnostics.

    Every tensor has one value per collocation point.  ``losses`` remain in the
    autograd graph; diagnostics are intended for logging but are not detached so
    callers may choose how to use them.
    """

    losses: dict[str, torch.Tensor]
    residual_norms: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PhysicsWeightSchedule:
    """Stateless, checkpoint-friendly scalar constraint-weight schedule."""

    schedule_type: str
    start: float
    end: float
    iterations: int = 0
    warmup_iterations: int = 0

    SUPPORTED_TYPES: ClassVar[tuple[str, ...]] = (
        "fixed",
        "linear",
        "exponential",
        "step",
        "smoothstep",
    )

    def __post_init__(self):
        schedule_type = str(self.schedule_type).lower()
        object.__setattr__(self, "schedule_type", schedule_type)
        if schedule_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown physics-weight schedule {schedule_type!r}; "
                f"expected one of {self.SUPPORTED_TYPES}."
            )
        for name in ("start", "end"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    f"Physics weight {name} must be finite and nonnegative."
                )
            object.__setattr__(self, name, float(value))
        for name in ("iterations", "warmup_iterations"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
                or int(value) != value
            ):
                raise ValueError(
                    f"Physics-weight schedule {name} must be a nonnegative integer."
                )
            object.__setattr__(self, name, int(value))
        if schedule_type != "fixed" and self.iterations <= 0:
            raise ValueError(f"{schedule_type} schedules require positive iterations.")
        if schedule_type == "fixed" and self.start != self.end:
            raise ValueError("Fixed schedules require matching start and end weights.")
        if schedule_type == "exponential" and (self.start <= 0 or self.end <= 0):
            raise ValueError(
                "Exponential schedules require strictly positive start and end weights."
            )

    @classmethod
    def from_config(cls, config: Real | Mapping[str, Any]) -> PhysicsWeightSchedule:
        """Parse a fixed number or the existing scheduled-weight mapping."""
        if isinstance(config, Real) and not isinstance(config, bool):
            value = float(config)
            return cls("fixed", value, value)
        if not isinstance(config, Mapping):
            raise TypeError("Physics weights must be numbers or schedule mappings.")

        schedule_type = str(config.get("type", "exponential")).lower()
        if schedule_type not in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown physics-weight schedule {schedule_type!r}; "
                f"expected one of {cls.SUPPORTED_TYPES}."
            )
        warmup_iterations = config.get("warmup_iterations", 0)
        if schedule_type == "fixed":
            supplied_values = [
                float(config[key]) for key in ("value", "start", "end") if key in config
            ]
            if supplied_values and any(
                value != supplied_values[0] for value in supplied_values[1:]
            ):
                raise ValueError(
                    "Fixed schedule value, start, and end must match when supplied."
                )
            value = supplied_values[0] if supplied_values else 0.0
            return cls("fixed", value, value, warmup_iterations=warmup_iterations)

        try:
            start = float(config["start"])
            end = float(config["end"])
            iterations = config["iterations"]
        except KeyError as error:
            raise ValueError(
                f"{schedule_type} schedules require start, end, and iterations."
            ) from error
        return cls(schedule_type, start, end, iterations, warmup_iterations)

    def value_at(self, global_step: int) -> float:
        """Return the weight for ``global_step`` without mutating schedule state."""
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, (Integral, Real))
            or not math.isfinite(global_step)
            or global_step < 0
            or int(global_step) != global_step
        ):
            raise ValueError("global_step must be a nonnegative integer.")
        global_step = int(global_step)
        if self.schedule_type == "fixed":
            return self.start

        elapsed = global_step - self.warmup_iterations
        if elapsed <= 0:
            return self.start
        if self.schedule_type == "step":
            return self.end if elapsed >= self.iterations else self.start

        progress = min(elapsed / self.iterations, 1.0)
        if self.schedule_type == "linear":
            fraction = progress
        elif self.schedule_type == "smoothstep":
            fraction = progress * progress * (3.0 - 2.0 * progress)
        elif self.schedule_type == "exponential":
            return self.start * (self.end / self.start) ** progress
        else:  # pragma: no cover - construction validates the type
            raise RuntimeError(f"Unhandled schedule type: {self.schedule_type}")
        return self.start + (self.end - self.start) * fraction

    def configuration(self) -> dict[str, float | int | str]:
        config: dict[str, float | int | str] = {
            "type": self.schedule_type,
            "start": self.start,
            "end": self.end,
        }
        if self.schedule_type != "fixed":
            config["iterations"] = self.iterations
        if self.warmup_iterations:
            config["warmup_iterations"] = self.warmup_iterations
        return config


def _validate_vector_field(
    name: str, value: torch.Tensor, coords: torch.Tensor
) -> None:
    if value.ndim != 2 or value.shape != (coords.shape[0], 3):
        raise ValueError(
            f"{name} must have shape [sample,3] matching coords; "
            f"got {tuple(value.shape)} for coords {tuple(coords.shape)}."
        )


def _curl_from_jacobian(field_jacobian: torch.Tensor) -> torch.Tensor:
    """Curl of a vector field whose Jacobian axes are [component,x,y,z,t]."""
    return torch.stack(
        [
            field_jacobian[:, 2, 1] - field_jacobian[:, 1, 2],
            field_jacobian[:, 0, 2] - field_jacobian[:, 2, 0],
            field_jacobian[:, 1, 0] - field_jacobian[:, 0, 1],
        ],
        dim=-1,
    )


def _spatial_divergence(field_jacobian: torch.Tensor) -> torch.Tensor:
    return field_jacobian[:, 0, 0] + field_jacobian[:, 1, 1] + field_jacobian[:, 2, 2]


def _squared_norm(value: torch.Tensor) -> torch.Tensor:
    return value.square().sum(dim=-1)


@dataclass(frozen=True)
class PhysicsCharacteristicScales:
    """Characteristic units expressed in the model's normalized units.

    Thus x_hat=x/length, t_hat=t/time, B_hat=B/magnetic_field and
    v_hat=v/velocity. The vector-potential unit is magnetic_field * length.
    """

    length: float
    time: float
    magnetic_field: float
    velocity: float

    def __post_init__(self):
        for name in ("length", "time", "magnetic_field", "velocity"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Characteristic {name} must be finite and positive.")

    @property
    def induction_factor(self) -> float:
        return self.time * self.velocity / self.length


class PhysicsConstraintModule(nn.Module):
    """Selectively evaluate ideal-MHD and magnetic-field constraints.

    ``raw`` uses quadratic residuals with a squared temporal derivative norm;
    the force-free residual is normalized by ``|B|``. ``relative`` provides
    amplitude-insensitive dimensionless residuals for the remaining constraints.
    ``characteristic`` nondimensionalizes fields and coordinate derivatives
    using fixed characteristic units, independently of dataset normalization.
    """

    def __init__(self, normalization: str = "raw", epsilon: float = 1e-6,
                 characteristic_scales: PhysicsCharacteristicScales | None = None):
        super().__init__()
        normalization = normalization.lower()
        if normalization not in ("raw", "relative", "characteristic"):
            raise ValueError("Physics normalization must be 'raw', 'relative' or 'characteristic'.")
        if normalization == "characteristic" and characteristic_scales is None:
            raise ValueError("Characteristic physics normalization requires characteristic scales.")
        if normalization != "characteristic" and characteristic_scales is not None:
            raise ValueError("Characteristic scales require normalization='characteristic'.")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Physics epsilon must be finite and strictly positive.")
        self.normalization = normalization
        self.epsilon = float(epsilon)
        self.characteristic_scales = characteristic_scales

    def _loss(
        self, residual_squared: torch.Tensor, scale_squared: torch.Tensor | None
    ) -> torch.Tensor:
        if self.normalization in ("raw", "characteristic"):
            return residual_squared
        if scale_squared is None:
            raise RuntimeError(
                "Relative physics constraints require a normalization scale."
            )
        return residual_squared / scale_squared.clamp_min(self.epsilon**2)

    def forward(
        self,
        b: torch.Tensor,
        v: torch.Tensor,
        coords: torch.Tensor,
        selected: Collection[str],
        a_jac_matrix: torch.Tensor | None = None,
        include_diagnostics: bool = False,
    ) -> PhysicsConstraintResult:
        if coords.ndim != 2 or coords.shape[-1] != 4:
            raise ValueError(
                f"coords must have shape [sample,4]; got {tuple(coords.shape)}."
            )
        _validate_vector_field("b", b, coords)
        _validate_vector_field("v", v, coords)
        if isinstance(selected, str):
            selected = (selected,)
        requested = set(selected)
        unknown = requested.difference(CONSTRAINT_NAMES)
        if unknown:
            raise ValueError(
                f"Unknown physics constraints {sorted(unknown)}; expected names from {CONSTRAINT_NAMES}."
            )
        if "gauge" in requested and a_jac_matrix is None:
            raise ValueError(
                "The gauge constraint requires a vector-potential Jacobian."
            )
        if a_jac_matrix is not None and a_jac_matrix.shape != (coords.shape[0], 3, 4):
            raise ValueError(
                f"a_jac_matrix must have shape [sample,3,4]; got {tuple(a_jac_matrix.shape)}."
            )
        if not requested:
            return PhysicsConstraintResult({}, {}, {})

        ordered = [name for name in CONSTRAINT_NAMES if name in requested]
        losses: dict[str, torch.Tensor] = {}
        residual_norms: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, torch.Tensor] = {}

        derivative_scale = coords.new_ones(4)
        induction_factor = 1.0
        if self.characteristic_scales is not None:
            scales = self.characteristic_scales
            b = b / scales.magnetic_field
            v = v / scales.velocity
            # Chain rule: d/dx_hat = (L0/L_model) d/dx_model,
            # and similarly for time. Model input coordinates stay unchanged.
            derivative_scale = coords.new_tensor(
                [scales.length, scales.length, scales.length, scales.time]
            )
            induction_factor = scales.induction_factor
            if a_jac_matrix is not None:
                a_jac_matrix = (
                    a_jac_matrix * derivative_scale
                    / (scales.magnetic_field * scales.length)
                )

        needs_b_jacobian = bool(
            requested.intersection({"divergence", "force_free", "potential", "dB_dt"})
            or ("induction" in requested and a_jac_matrix is None)
            or include_diagnostics
        )
        b_jacobian = jacobian(b, coords) * derivative_scale if needs_b_jacobian else None
        spatial_b_squared = None
        if b_jacobian is not None:
            spatial_b_squared = b_jacobian[..., :3].square().sum(dim=(-2, -1))

        current_density = None
        if requested.intersection({"force_free", "potential"}) or include_diagnostics:
            if (
                b_jacobian is None
            ):  # defensive; dependencies above should guarantee this
                raise RuntimeError(
                    "A magnetic-field Jacobian is required to compute curl(B)."
                )
            current_density = _curl_from_jacobian(b_jacobian)

        induction_lhs = induction_rhs = induction_residual = None
        v_cross_b = None
        if "induction" in requested or include_diagnostics:
            v_cross_b = torch.cross(v, b, dim=-1)
        if "induction" in requested:
            if a_jac_matrix is None:
                if b_jacobian is None:
                    raise RuntimeError(
                        "Direct-field induction requires a magnetic-field Jacobian."
                    )
                v_cross_b_jacobian = jacobian(v_cross_b, coords) * derivative_scale
                induction_lhs = b_jacobian[..., 3]
                induction_rhs = induction_factor * _curl_from_jacobian(v_cross_b_jacobian)
            else:
                induction_lhs = a_jac_matrix[..., 3]
                induction_rhs = induction_factor * v_cross_b
            induction_residual = induction_lhs - induction_rhs

        for name in ordered:
            if name == "divergence":
                divergence = _spatial_divergence(b_jacobian)
                residual_squared = divergence.square()
                losses[name] = self._loss(residual_squared, spatial_b_squared)
                residual_norms[name] = divergence.abs()
            elif name == "force_free":
                force = torch.cross(current_density, b, dim=-1)
                # Penalize the Lorentz force per unit magnetic-field strength:
                #
                #     r_ff = (curl(B) x B) / |B|.
                #
                # The smooth epsilon floor keeps the residual and its gradient
                # finite at magnetic nulls.  Do not additionally divide by
                # |curl(B)|: its magnitude is the physical quantity minimized
                # by this constraint.
                b_norm = torch.sqrt(_squared_norm(b) + self.epsilon**2)
                normalized_force = force / b_norm.unsqueeze(-1)
                losses[name] = _squared_norm(normalized_force)
                residual_norms[name] = torch.linalg.vector_norm(
                    normalized_force, dim=-1
                )
            elif name == "potential":
                residual_squared = _squared_norm(current_density)
                losses[name] = self._loss(residual_squared, spatial_b_squared)
                residual_norms[name] = torch.linalg.vector_norm(current_density, dim=-1)
            elif name == "dB_dt":
                dB_dt = b_jacobian[..., 3]
                residual_squared = _squared_norm(dB_dt)
                losses[name] = self._loss(residual_squared, _squared_norm(b))
                residual_norms[name] = torch.linalg.vector_norm(dB_dt, dim=-1)
            elif name == "induction":
                residual_squared = _squared_norm(induction_residual)
                lhs_norm = torch.linalg.vector_norm(induction_lhs, dim=-1)
                rhs_norm = torch.linalg.vector_norm(induction_rhs, dim=-1)
                scale_squared = (lhs_norm + rhs_norm).square()
                losses[name] = self._loss(residual_squared, scale_squared)
                residual_norms[name] = torch.linalg.vector_norm(
                    induction_residual, dim=-1
                )
            elif name == "gauge":
                divergence_a = _spatial_divergence(a_jac_matrix)
                residual_squared = divergence_a.square()
                spatial_a_squared = a_jac_matrix[..., :3].square().sum(dim=(-2, -1))
                losses[name] = self._loss(residual_squared, spatial_a_squared)
                residual_norms[name] = divergence_a.abs()

        if include_diagnostics:
            if current_density is not None:
                diagnostics["current_density_squared"] = _squared_norm(current_density)
            if v_cross_b is not None:
                diagnostics["v_cross_b_norm"] = torch.linalg.vector_norm(
                    v_cross_b, dim=-1
                )
            if b_jacobian is not None:
                spherical_coords = cartesian_to_spherical(coords[..., :3], torch)
                radial_direction = torch.stack(
                    [
                        torch.sin(spherical_coords[..., 1])
                        * torch.cos(spherical_coords[..., 2]),
                        torch.sin(spherical_coords[..., 1])
                        * torch.sin(spherical_coords[..., 2]),
                        torch.cos(spherical_coords[..., 1]),
                    ],
                    dim=-1,
                )
                radial_derivative = torch.einsum(
                    "nci,ni->nc", b_jacobian[..., :3], radial_direction
                )
                diagnostics["radial_gradient_norm"] = torch.linalg.vector_norm(
                    radial_derivative, dim=-1
                )

        return PhysicsConstraintResult(losses, residual_norms, diagnostics)
