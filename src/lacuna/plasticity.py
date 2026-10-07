"""Standard edge-local synaptic plasticity rules.

The classes in this module are immutable authoring values.  Execution belongs
exclusively to the C evaluator. Python only validates, serializes, and packs the
selected rule for each edge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

from .errors import ResolutionError


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{name} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ResolutionError(f"{name} must be a positive finite number")
    return result


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{name} must be a nonnegative finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{name} must be a nonnegative finite number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ResolutionError(f"{name} must be a nonnegative finite number")
    return result


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ResolutionError(f"{name} must be a finite number")
    return result


def _bounds(value: object) -> tuple[float, float]:
    try:
        lower, upper = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ResolutionError("plasticity bounds must contain (minimum, maximum)") from exc
    lower = _nonnegative(lower, "minimum plastic weight")
    upper = _positive(upper, "maximum plastic weight")
    if not lower < upper:
        raise ResolutionError("plasticity weight bounds require minimum < maximum")
    return lower, upper


@dataclass(frozen=True)
class PairSTDP:
    """All-to-all multiplicative pair STDP on normalized weight magnitude."""

    tau_pre: float = 5.0
    tau_post: float = 5.0
    a_plus: float = 0.6
    a_minus: float = 0.3
    learning_rate: float = 0.05
    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau_pre", _positive(self.tau_pre, "tau_pre"))
        object.__setattr__(self, "tau_post", _positive(self.tau_post, "tau_post"))
        object.__setattr__(self, "a_plus", _nonnegative(self.a_plus, "a_plus"))
        object.__setattr__(self, "a_minus", _nonnegative(self.a_minus, "a_minus"))
        object.__setattr__(
            self, "learning_rate", _nonnegative(self.learning_rate, "learning_rate")
        )
        object.__setattr__(self, "bounds", _bounds(self.bounds))


@dataclass(frozen=True)
class TripletSTDP:
    """Full all-to-all Pfister--Gerstner triplet STDP rule."""

    tau_plus: float = 16.8
    tau_minus: float = 33.7
    tau_x: float = 946.0
    tau_y: float = 27.0
    a2_plus: float = 6.1e-3
    a2_minus: float = 1.6e-3
    a3_plus: float = 6.7e-3
    a3_minus: float = 1.4e-3
    learning_rate: float = 1.0
    bounds: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        for name in ("tau_plus", "tau_minus", "tau_x", "tau_y"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("a2_plus", "a2_minus", "a3_plus", "a3_minus"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(
            self, "learning_rate", _nonnegative(self.learning_rate, "learning_rate")
        )
        object.__setattr__(self, "bounds", _bounds(self.bounds))

    @classmethod
    def visual_cortex(
        cls, *, learning_rate: float = 1.0, bounds: tuple[float, float] = (0.0, 1.0)
    ) -> "TripletSTDP":
        """Return the visual-cortex parameters from Pfister and Gerstner."""

        return cls(
            tau_plus=16.8,
            tau_minus=33.7,
            tau_x=101.0,
            tau_y=125.0,
            a2_plus=5.0e-10,
            a2_minus=7.0e-3,
            a3_plus=6.2e-3,
            a3_minus=2.3e-4,
            learning_rate=learning_rate,
            bounds=bounds,
        )

    @classmethod
    def hippocampus(
        cls, *, learning_rate: float = 1.0, bounds: tuple[float, float] = (0.0, 1.0)
    ) -> "TripletSTDP":
        """Return the hippocampal parameters from Pfister and Gerstner."""

        return cls(learning_rate=learning_rate, bounds=bounds)


@dataclass(frozen=True)
class ModulatedSTDP:
    """Split causal/anti-causal eligibility with a scoped third factor."""

    tau_pre: float = 5.0
    tau_post: float = 5.0
    tau_eligibility_plus: float = 7210.0
    tau_eligibility_minus: float = 3610.0
    positive_plus: float = 1.0
    positive_minus: float = -1.0
    negative_plus: float = -1.0
    negative_minus: float = 1.0
    learning_rate: float = 0.05
    bounds: tuple[float, float] = (1.0e-5, 1.0)
    consume_on_modulation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "tau_pre",
            "tau_post",
            "tau_eligibility_plus",
            "tau_eligibility_minus",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in (
            "positive_plus",
            "positive_minus",
            "negative_plus",
            "negative_minus",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self, "learning_rate", _nonnegative(self.learning_rate, "learning_rate")
        )
        object.__setattr__(self, "bounds", _bounds(self.bounds))
        if not isinstance(self.consume_on_modulation, bool):
            raise ResolutionError("consume_on_modulation must be boolean")


@dataclass(frozen=True)
class VoltageModulatedSTDP:
    """Third-factor learning with spike and voltage-sensitivity eligibility.

    ``surrogate_threshold`` and ``surrogate_slope`` define a smooth softplus
    firing surrogate for a deterministic threshold neuron. The compiled
    learning equation uses the derivative of that surrogate. It is deliberately
    identified as a surrogate rather than the derivative of the discontinuous
    hard-spike count.
    """

    tau_pre: float = 20.0
    tau_post: float = 20.0
    tau_eligibility_plus: float = 80.0
    tau_eligibility_minus: float = 80.0
    tau_voltage_eligibility: float = 80.0
    surrogate_threshold: float = -60.0
    surrogate_slope: float = 3.0
    spike_scale: float = 1.0
    voltage_scale: float = 1.0
    learning_rate: float = 0.05
    bounds: tuple[float, float] = (1.0e-5, 1.0)
    consume_on_modulation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "tau_pre",
            "tau_post",
            "tau_eligibility_plus",
            "tau_eligibility_minus",
            "tau_voltage_eligibility",
            "surrogate_slope",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in (
            "surrogate_threshold",
            "spike_scale",
            "voltage_scale",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self, "learning_rate", _nonnegative(self.learning_rate, "learning_rate")
        )
        object.__setattr__(self, "bounds", _bounds(self.bounds))
        if not isinstance(self.consume_on_modulation, bool):
            raise ResolutionError("consume_on_modulation must be boolean")


@dataclass(frozen=True)
class SoftExcursionModulated:
    """Apically modulated eligibility from graded near-threshold excursions.

    This rule deliberately has no postsynaptic-spike or pair-STDP term.  Each
    presynaptic delivery samples the postsynaptic voltage, compares it with an
    edge-local exponentially weighted voltage baseline, and marks the synapse
    in proportion to both upward movement and near-threshold proximity.

    The smooth gates model graded local biophysics. They are not identified as
    derivatives of the hard spike function.
    """

    tau_pre: float = 20.0
    tau_voltage_baseline: float = 40.0
    tau_eligibility: float = 80.0
    threshold: float = -60.0
    proximity_width: float = 5.0
    proximity_slope: float = 0.75
    excursion_smoothing: float = 0.05
    baseline_epsilon: float = 1.0e-9
    fixed_baseline: float = -65.0
    adaptive_baseline: bool = True
    use_upward_excursion: bool = True
    use_proximity: bool = True
    soft_scale: float = 1.0
    learning_rate: float = 0.05
    bounds: tuple[float, float] = (1.0e-5, 1.0)
    consume_on_modulation: bool = True

    def __post_init__(self) -> None:
        for name in (
            "tau_pre",
            "tau_voltage_baseline",
            "tau_eligibility",
            "proximity_width",
            "proximity_slope",
            "excursion_smoothing",
            "baseline_epsilon",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))
        object.__setattr__(
            self, "fixed_baseline", _finite(self.fixed_baseline, "fixed_baseline")
        )
        object.__setattr__(self, "soft_scale", _finite(self.soft_scale, "soft_scale"))
        object.__setattr__(
            self, "learning_rate", _nonnegative(self.learning_rate, "learning_rate")
        )
        object.__setattr__(self, "bounds", _bounds(self.bounds))
        for name in (
            "adaptive_baseline",
            "use_upward_excursion",
            "use_proximity",
            "consume_on_modulation",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ResolutionError(f"{name} must be boolean")


PlasticityRule = (
    PairSTDP
    | TripletSTDP
    | ModulatedSTDP
    | VoltageModulatedSTDP
    | SoftExcursionModulated
)


def plasticity_to_document(rule: PlasticityRule) -> dict[str, object]:
    """Serialize one plasticity rule into canonical graph data."""

    if isinstance(rule, PairSTDP):
        kind = "PAIR_STDP"
    elif isinstance(rule, TripletSTDP):
        kind = "TRIPLET_STDP"
    elif isinstance(rule, ModulatedSTDP):
        kind = "MODULATED_STDP"
    elif isinstance(rule, VoltageModulatedSTDP):
        kind = "VOLTAGE_MODULATED_STDP"
    elif isinstance(rule, SoftExcursionModulated):
        kind = "SOFT_EXCURSION_MODULATED"
    else:  # pragma: no cover - guarded by graph validation
        raise TypeError("unsupported plasticity rule")
    values = asdict(rule)
    values["bounds"] = list(rule.bounds)
    return {"kind": kind, "parameters": values}


def plasticity_from_document(value: object) -> PlasticityRule:
    """Validate and reconstruct one plasticity rule from graph data."""

    if not isinstance(value, Mapping) or not isinstance(value.get("parameters"), Mapping):
        raise ResolutionError("plasticity must be a kind/parameters object")
    parameters = dict(value["parameters"])
    if "bounds" in parameters:
        try:
            parameters["bounds"] = tuple(parameters["bounds"])
        except TypeError as exc:
            raise ResolutionError("plasticity bounds must be an array") from exc
    kind = value.get("kind")
    try:
        if kind == "PAIR_STDP":
            return PairSTDP(**parameters)
        if kind == "TRIPLET_STDP":
            return TripletSTDP(**parameters)
        if kind == "MODULATED_STDP":
            return ModulatedSTDP(**parameters)
        if kind == "VOLTAGE_MODULATED_STDP":
            return VoltageModulatedSTDP(**parameters)
        if kind == "SOFT_EXCURSION_MODULATED":
            return SoftExcursionModulated(**parameters)
    except TypeError as exc:
        raise ResolutionError(f"invalid {kind!r} plasticity parameters") from exc
    raise ResolutionError(f"unknown plasticity kind {kind!r}")
