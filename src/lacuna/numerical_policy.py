"""Numerical defaults and implementation metadata for native scalar profiles.

Metadata alone never authorizes a graph to run. The compiler, native ABI and
deployment loader validate the chosen profile independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

from .precision import PrecisionProfile, normalize_precision


_POLICY_REVISION = 1
_PRIMITIVE_BINDING_SCOPE = (
    "CONST", "PARAM", "NEG", "ADD", "SUB", "MUL", "DIV", "MAX",
)
_REDUCED_PRECISION_REQUIREMENTS = (
    "target-toolchain arithmetic and math-library validation",
    "device execution, memory capacity, and clock-horizon validation",
)


class NumericalStatus(str, Enum):
    """Implementation baseline or outstanding certification, never a guarantee."""

    EXISTING = "existing"
    IMPLEMENTED = "implemented"
    UNCERTIFIED = "uncertified"


@dataclass(frozen=True)
class NumericalCapabilities:
    """Simulation capabilities, separate from the primitive binding's scope.

    The float64 clock-comparison entry refers only to its existing same-width
    behavior. It does not certify comparisons in the mixed profile.
    """

    analytical_propagation: NumericalStatus
    crossing: NumericalStatus
    root_finding: NumericalStatus
    adaptive_integration: NumericalStatus
    hazard: NumericalStatus
    mixed_clock_comparisons: NumericalStatus

    def to_record(self) -> dict[str, str]:
        """Return detached, JSON-compatible capability statuses."""

        return {
            item.name: getattr(self, item.name).value for item in fields(self)
        }


@dataclass(frozen=True)
class NumericalPolicy:
    """An immutable policy identity and precision-aware numerical defaults.

    Use :func:`numerical_policy` to resolve profile and clock aliases. Primitive
    scope names the bounded binding contract, not its availability in a loaded
    library. It excludes transcendental functions, simulation, learning, and
    clock arithmetic. In particular it is not a no-hidden-widening certificate
    for strict float32 simulation.
    """

    precision: PrecisionProfile
    policy_revision: int = field(default=_POLICY_REVISION, init=False)
    authorizes_execution: bool = field(default=False, init=False)
    primitive_binding_scope: tuple[str, ...] = field(
        default=_PRIMITIVE_BINDING_SCOPE, init=False,
    )
    capabilities: NumericalCapabilities = field(init=False)
    preserves_existing_float64_settings: bool = field(init=False)
    required_certifications: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.precision, PrecisionProfile):
            raise TypeError("NumericalPolicy precision must be a PrecisionProfile")
        existing = self.precision is PrecisionProfile.FLOAT64
        status = (NumericalStatus.EXISTING if existing else NumericalStatus.UNCERTIFIED
                  if self.precision is PrecisionProfile.FLOAT16 else NumericalStatus.IMPLEMENTED)
        object.__setattr__(self, "capabilities", NumericalCapabilities(
            **{item.name: status for item in fields(NumericalCapabilities)}
        ))
        object.__setattr__(self, "preserves_existing_float64_settings", existing)
        object.__setattr__(self, "required_certifications", (
            () if existing else _REDUCED_PRECISION_REQUIREMENTS
        ))

    @property
    def arithmetic_revision(self) -> int:
        """Reuse the selected profile's declared arithmetic identity."""

        return self.precision.arithmetic_revision

    @property
    def step_defaults(self):
        """Return local integration targets, not a global trajectory-error bound."""
        from dataclasses import replace
        from .ir import NumericalConfig

        config = NumericalConfig()
        if self.precision is PrecisionProfile.FLOAT16:
            return replace(
                config, relative_tolerance=8 * float.fromhex("0x1p-10"),
                absolute_tolerance=float.fromhex("0x1p-24"),
                event_tolerance=float.fromhex("0x1p-10"),
                minimum_step=float.fromhex("0x1p-24"),
            )
        if self.precision.real_bits == 32:
            config = replace(
                config, relative_tolerance=64 * float.fromhex("0x1p-23"),
                event_tolerance=8 * float.fromhex("0x1p-23"),
            )
        return config

    @property
    def crossing_relative_tolerance(self) -> float:
        """Return a crossing target subject to adjacent clock representation."""
        if self.precision is PrecisionProfile.FLOAT16:
            return 2 * float.fromhex("0x1p-10")
        return 1e-10 if self.precision.real_bits == 64 else 8 * float.fromhex("0x1p-23")

    @property
    def cache_key(self) -> tuple[str, int, int, int, int]:
        """Append policy revision to the complete precision cache identity."""

        return (*self.precision.cache_key, self.policy_revision)

    def to_record(self) -> dict[str, object]:
        """Export detached metadata without solver defaults or runtime authority."""

        return {
            "schema_version": 1,
            "policy_revision": self.policy_revision,
            "precision": self.precision.to_record(),
            "capabilities": self.capabilities.to_record(),
            "preserves_existing_float64_settings": (
                self.preserves_existing_float64_settings
            ),
            "required_certifications": list(self.required_certifications),
            "primitive_binding_scope": list(self.primitive_binding_scope),
            "authorizes_execution": self.authorizes_execution,
        }


def numerical_policy(
    precision: PrecisionProfile | str = PrecisionProfile.FLOAT64,
    *,
    time_precision: str | None = None,
) -> NumericalPolicy:
    """Describe the policy without loading a backend or approving a graph.

    Float16 numerical capabilities remain uncertified. Float64 retains existing settings.
    Float32 local error targets account for
    scalar roundoff, and event localization can stop at adjacent timestamps.
    """

    return NumericalPolicy(normalize_precision(precision, time_precision))
