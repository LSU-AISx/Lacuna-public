"""High-level Python definitions for Lacuna's standard neuron families.

These objects are authoring conveniences, not alternate evaluators.  They emit the
same intrinsic model records used by the equation DSL, and resolution still lowers
through the existing capability resolver to the C runtime.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

from .dsl import parse_neuron
from .errors import CapabilityError, ResolutionError
from .graph import GraphModel, GraphNode
from .ir import (
    NeuronPolarity,
    NeuronModel,
    NumericalConfig,
    ResolvedAdaptiveLIF,
    ResolvedReactiveIF,
    ResolvedScalarLIF,
    ResolvedSteppedNeuron,
)
from .resolver import (
    resolve_adaptive_escape_lif,
    resolve_adaptive_lif,
    resolve_escape_lif,
    resolve_reactive_if,
    resolve_scalar_lif,
    resolve_stepped_neuron,
)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ResolutionError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ResolutionError(f"{name} must be a finite real number")
    return result


def _literal(value: float) -> str:
    """Return a stable, parser-friendly binary64 literal."""

    return repr(float(value))


class StandardNeuron(ABC):
    """Common graph and resolver interface for a standard neuron definition.

    A standard model owns the default parameter set shared by a population.  Its
    :meth:`node` method stores only per-node overrides, so graph artifacts remain
    compact and retain the same model/binding separation as hand-authored models.
    """

    name: str
    refractory: float

    @property
    @abstractmethod
    def defaults(self) -> Mapping[str, float]:
        """Default model parameter values keyed by their public Python names."""

    @property
    @abstractmethod
    def positive_parameters(self) -> frozenset[str]:
        """Parameters whose accepted domain is strictly positive."""

    @property
    @abstractmethod
    def state_names(self) -> tuple[str, ...]:
        """Intrinsic state order used for initial values and inspection."""

    @property
    @abstractmethod
    def source(self) -> str:
        """Canonical intrinsic DSL generated for the existing graph schema."""

    def _validate_definition(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ResolutionError("standard neuron name must be a nonempty string")
        values = {key: _number(value, key) for key, value in self.defaults.items()}
        self._validate_values(values)
        refractory = _number(self.refractory, "refractory")
        if refractory < 0.0:
            raise ResolutionError("refractory must be nonnegative")

    def _validate_values(self, values: Mapping[str, float]) -> None:
        for name in self.positive_parameters:
            if values[name] <= 0.0:
                raise ResolutionError(f"{name} must be positive")

    def _bindings(
        self,
        bindings: Mapping[str, float] | None,
        overrides: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        supplied = dict(bindings or {})
        overlap = set(supplied).intersection(overrides)
        if overlap:
            raise ResolutionError(
                "parameter supplied twice: " + ", ".join(sorted(overlap))
            )
        supplied.update(overrides)
        unknown = set(supplied).difference(self.defaults)
        if unknown:
            raise ResolutionError(
                "unknown standard neuron parameter(s): "
                + ", ".join(sorted(unknown))
            )
        normalized = {
            name: _number(value, f"parameter '{name}'")
            for name, value in supplied.items()
        }
        values = {name: float(value) for name, value in self.defaults.items()}
        values.update(normalized)
        self._validate_values(values)
        return normalized, values

    @property
    def model(self) -> GraphModel:
        """Graph model record accepted by :class:`lacuna.Graph`."""

        return GraphModel(self.name, self.source)

    @property
    def authored_model(self) -> NeuronModel:
        """Typed equation model accepted by the low-level resolvers."""

        return parse_neuron(self.source)

    @property
    def default_initial(self) -> float | tuple[float, ...]:
        """Initial state used by :meth:`node` when none is supplied."""

        return self._default_initial(self.defaults)

    @abstractmethod
    def _default_initial(
        self, values: Mapping[str, float]
    ) -> float | tuple[float, ...]:
        pass

    def _normalize_initial(
        self, initial: float | tuple[float, ...]
    ) -> float | tuple[float, ...]:
        if isinstance(initial, (tuple, list)):
            values = tuple(
                _number(value, f"initial state component {index}")
                for index, value in enumerate(initial)
            )
            if len(values) != len(self.state_names):
                raise ResolutionError(
                    f"initial state for {type(self).__name__} must contain "
                    f"{len(self.state_names)} values in {self.state_names} order"
                )
            return values[0] if len(values) == 1 else values
        return _number(initial, "initial state")

    def node(
        self,
        node_id: int,
        *,
        initial: float | tuple[float, ...] | None = None,
        bindings: Mapping[str, float] | None = None,
        polarity: NeuronPolarity = NeuronPolarity.EXCITATORY,
        **parameter_overrides: float,
    ) -> GraphNode:
        """Create a graph node with validated, sparse parameter overrides.

        If ``initial`` is omitted, the membrane starts at the effective ``v_rest``
        (or ``v_center`` for QIF) and adaptation state starts at zero.  Consequently
        a per-node resting-potential override also updates its implicit initial state.
        """

        if not isinstance(polarity, NeuronPolarity):
            raise ResolutionError(
                "polarity must be EXCITATORY, INHIBITORY, or MIXED"
            )
        normalized, values = self._bindings(bindings, parameter_overrides)
        chosen_initial = (
            self._default_initial(values) if initial is None else initial
        )
        return GraphNode(
            id=node_id,
            model=self.name,
            initial=self._normalize_initial(chosen_initial),
            bindings=normalized,
            polarity=polarity,
        )

    def resolve(
        self,
        bindings: Mapping[str, float] | None = None,
        *,
        numerical: NumericalConfig | None = None,
        **parameter_overrides: float,
    ) -> ResolvedScalarLIF | ResolvedAdaptiveLIF | ResolvedReactiveIF | ResolvedSteppedNeuron:
        """Resolve this definition directly through Lacuna's existing backend."""

        normalized, _ = self._bindings(bindings, parameter_overrides)
        return self._resolve(normalized, numerical)

    @abstractmethod
    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedScalarLIF | ResolvedAdaptiveLIF | ResolvedReactiveIF | ResolvedSteppedNeuron:
        pass


@dataclass(frozen=True, kw_only=True)
class IntegrateAndFire(StandardNeuron):
    """Deposit-triggered integrate-and-fire with optional timestamp leak.

    With ``leak=False``, subthreshold charge persists until a later event batch.
    With ``leak=True``, the membrane is reset once immediately before all deposits
    arriving at a timestamp are summed.  The latter is RISP's timestamp-leak
    behavior, not an exponential membrane decay.
    """

    name: str = "if"
    threshold: float = 1.0
    reset: float = 0.0
    refractory: float = 0.0
    leak: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.leak, bool):
            raise ResolutionError("leak must be a boolean")
        self._validate_definition()

    @property
    def defaults(self) -> Mapping[str, float]:
        return {"threshold": self.threshold, "reset": self.reset}

    @property
    def positive_parameters(self) -> frozenset[str]:
        return frozenset()

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("v",)

    @property
    def source(self) -> str:
        mode = "RESET_BEFORE_DEPOSIT" if self.leak else "HOLD"
        return f"""
neuron {self.name} {{
  params {{
    threshold = {_literal(self.threshold)}
    reset = {_literal(self.reset)}
  }}
  state {{ v: membrane }}
  dynamics {{ dv/dt = 0.0 }}
  threshold {{ v > threshold }}
  reset {{ v <- reset }}
  refractory {{ duration = {_literal(self.refractory)} }}
  reactive {{ mode = {mode} }}
}}
""".strip()

    def _default_initial(self, values: Mapping[str, float]) -> float:
        return values["reset"]

    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedReactiveIF:
        del numerical
        return resolve_reactive_if(self.authored_model, bindings)


@dataclass(frozen=True, kw_only=True)
class LIF(StandardNeuron):
    """Current-driven leaky integrate-and-fire model.

    The generated equation is
    ``tau_m * dv/dt = -(v-v_rest) + resistance * (drive + i_syn)``.
    ``i_syn`` is present only when ``synaptic_input=True``. This optional receptor
    form is the one accepted by Lacuna's folded and per-edge filtered-current
    synapse capabilities. Without it, delta spikes deposit directly into ``v``.
    """

    name: str = "lif"
    tau_m: float = 20.0
    v_rest: float = -65.0
    resistance: float = 1.0
    drive: float = 0.0
    v_threshold: float = -50.0
    v_reset: float = -65.0
    refractory: float = 2.0
    synaptic_input: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.synaptic_input, bool):
            raise ResolutionError("synaptic_input must be a boolean")
        self._validate_definition()

    @property
    def defaults(self) -> Mapping[str, float]:
        return {
            "tau_m": self.tau_m,
            "v_rest": self.v_rest,
            "resistance": self.resistance,
            "drive": self.drive,
            "v_threshold": self.v_threshold,
            "v_reset": self.v_reset,
        }

    @property
    def positive_parameters(self) -> frozenset[str]:
        return frozenset({"tau_m", "resistance"})

    @property
    def state_names(self) -> tuple[str, ...]:
        # The receptor is an authoring/mapping declaration.  Its executable state
        # is supplied by the attached synapse block, not by the node initial tuple.
        return ("v",)

    @property
    def receptor_name(self) -> str | None:
        return "i_syn" if self.synaptic_input else None

    def _validate_values(self, values: Mapping[str, float]) -> None:
        super()._validate_values(values)
        if values["v_reset"] >= values["v_threshold"]:
            raise ResolutionError("v_reset must be strictly below v_threshold")

    def _default_initial(self, values: Mapping[str, float]) -> float:
        return values["v_rest"]

    @property
    def source(self) -> str:
        values = self.defaults
        receptor_state = "\n        i_syn : receptor" if self.synaptic_input else ""
        receptor_term = " + resistance*i_syn/tau_m" if self.synaptic_input else ""
        return f"""neuron StandardLIF {{
    params {{
        tau_m : positive = {_literal(values['tau_m'])}
        v_rest = {_literal(values['v_rest'])}
        resistance : positive = {_literal(values['resistance'])}
        drive = {_literal(values['drive'])}
        v_threshold = {_literal(values['v_threshold'])}
        v_reset = {_literal(values['v_reset'])}
    }}
    state {{
        v : membrane{receptor_state}
    }}
    dynamics {{
        dv/dt = -(v-v_rest)/tau_m + resistance*drive/tau_m{receptor_term}
    }}
    threshold {{ v > v_threshold }}
    reset {{ v <- v_reset }}
    refractory {{ duration = {_literal(self.refractory)} }}
}}
"""

    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedScalarLIF:
        if numerical is not None:
            raise ResolutionError("numerical configuration does not apply to analytical LIF")
        if self.synaptic_input:
            raise CapabilityError(
                "a receptor-enabled LIF must be resolved in a Graph with its "
                "node-scoped or per-edge synapse mapping"
            )
        return resolve_scalar_lif(self.authored_model, bindings)


@dataclass(frozen=True, kw_only=True)
class AdaptiveLIF(StandardNeuron):
    """LIF with one independent, spike-triggered adaptation current.

    The generated equations are
    ``tau_m * dv/dt = -(v-v_rest) + resistance*(drive-w)`` and
    ``dw/dt = -w/tau_adaptation``.  A spike applies
    ``w <- w + adaptation_increment``.  This family selects Lacuna's exact
    two-real-exponential analytical capability.
    """

    name: str = "adaptive_lif"
    tau_m: float = 20.0
    tau_adaptation: float = 100.0
    v_rest: float = -65.0
    resistance: float = 1.0
    drive: float = 0.0
    adaptation_increment: float = 1.0
    v_threshold: float = -50.0
    v_reset: float = -65.0
    refractory: float = 2.0

    def __post_init__(self) -> None:
        self._validate_definition()

    @property
    def defaults(self) -> Mapping[str, float]:
        return {
            "tau_m": self.tau_m,
            "tau_adaptation": self.tau_adaptation,
            "v_rest": self.v_rest,
            "resistance": self.resistance,
            "drive": self.drive,
            "adaptation_increment": self.adaptation_increment,
            "v_threshold": self.v_threshold,
            "v_reset": self.v_reset,
        }

    @property
    def positive_parameters(self) -> frozenset[str]:
        return frozenset({"tau_m", "tau_adaptation", "resistance"})

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("v", "w")

    def _validate_values(self, values: Mapping[str, float]) -> None:
        super()._validate_values(values)
        if values["tau_m"] == values["tau_adaptation"]:
            raise ResolutionError(
                "tau_adaptation must differ from tau_m for the current analytical capability"
            )
        if values["adaptation_increment"] < 0.0:
            raise ResolutionError("adaptation_increment must be nonnegative")
        if values["v_reset"] >= values["v_threshold"]:
            raise ResolutionError("v_reset must be strictly below v_threshold")

    def _default_initial(self, values: Mapping[str, float]) -> tuple[float, float]:
        return (values["v_rest"], 0.0)

    @property
    def source(self) -> str:
        values = self.defaults
        return f"""neuron StandardAdaptiveLIF {{
    params {{
        tau_m : positive = {_literal(values['tau_m'])}
        tau_adaptation : positive = {_literal(values['tau_adaptation'])}
        v_rest = {_literal(values['v_rest'])}
        resistance : positive = {_literal(values['resistance'])}
        drive = {_literal(values['drive'])}
        adaptation_increment = {_literal(values['adaptation_increment'])}
        v_threshold = {_literal(values['v_threshold'])}
        v_reset = {_literal(values['v_reset'])}
    }}
    state {{
        v : membrane
        w : adaptation
    }}
    dynamics {{
        dv/dt = -(v-v_rest)/tau_m + resistance*(drive-w)/tau_m
        dw/dt = -w/tau_adaptation
    }}
    threshold {{ v > v_threshold }}
    reset {{
        v <- v_reset
        w <- w + adaptation_increment
    }}
    refractory {{ duration = {_literal(self.refractory)} }}
}}
"""

    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedAdaptiveLIF:
        if numerical is not None:
            raise ResolutionError(
                "numerical configuration does not apply to analytical AdaptiveLIF"
            )
        return resolve_adaptive_lif(self.authored_model, bindings)


@dataclass(frozen=True, kw_only=True)
class EscapeLIF(StandardNeuron):
    """Stable LIF whose spikes are generated by a voltage-dependent hazard.

    The conditional intensity is
    ``escape_rate * exp((v-v_escape)/delta_v)``. Lacuna samples the next spike
    from integrated hazard along the exact LIF trajectory. ``delta_v`` is not a
    timestep and the model performs no probability polling.
    """

    name: str = "escape_lif"
    tau_m: float = 20.0
    v_rest: float = -65.0
    resistance: float = 1.0
    drive: float = 0.0
    escape_rate: float = 0.01
    v_escape: float = -50.0
    delta_v: float = 2.0
    v_reset: float = -65.0
    refractory: float = 2.0

    def __post_init__(self) -> None:
        self._validate_definition()

    @property
    def defaults(self) -> Mapping[str, float]:
        return {
            "tau_m": self.tau_m,
            "v_rest": self.v_rest,
            "resistance": self.resistance,
            "drive": self.drive,
            "escape_rate": self.escape_rate,
            "v_escape": self.v_escape,
            "delta_v": self.delta_v,
            "v_reset": self.v_reset,
        }

    @property
    def positive_parameters(self) -> frozenset[str]:
        return frozenset({"tau_m", "resistance", "escape_rate", "delta_v"})

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("v",)

    def _default_initial(self, values: Mapping[str, float]) -> float:
        return values["v_rest"]

    @property
    def source(self) -> str:
        values = self.defaults
        return f"""neuron StandardEscapeLIF {{
    params {{
        tau_m : positive = {_literal(values['tau_m'])}
        v_rest = {_literal(values['v_rest'])}
        resistance : positive = {_literal(values['resistance'])}
        drive = {_literal(values['drive'])}
        escape_rate : positive = {_literal(values['escape_rate'])}
        v_escape = {_literal(values['v_escape'])}
        delta_v : positive = {_literal(values['delta_v'])}
        v_reset = {_literal(values['v_reset'])}
    }}
    state {{ v : membrane }}
    dynamics {{
        dv/dt = -(v-v_rest)/tau_m + resistance*drive/tau_m
    }}
    hazard {{ rate = escape_rate*exp((v-v_escape)/delta_v) }}
    reset {{ v <- v_reset }}
    refractory {{ duration = {_literal(self.refractory)} }}
}}
"""

    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedScalarLIF:
        if numerical is not None:
            raise ResolutionError(
                "numerical configuration does not apply to analytical EscapeLIF"
            )
        return resolve_escape_lif(self.authored_model, bindings)


@dataclass(frozen=True, kw_only=True)
class AdaptiveEscapeLIF(StandardNeuron):
    """Spike-triggered adaptive LIF with intrinsic exponential escape noise."""

    name: str = "adaptive_escape_lif"
    tau_m: float = 20.0
    tau_adaptation: float = 100.0
    v_rest: float = -65.0
    resistance: float = 1.0
    drive: float = 0.0
    adaptation_increment: float = 1.0
    escape_rate: float = 0.01
    v_escape: float = -50.0
    delta_v: float = 2.0
    v_reset: float = -65.0
    refractory: float = 2.0

    def __post_init__(self) -> None:
        self._validate_definition()

    @property
    def defaults(self) -> Mapping[str, float]:
        return {
            "tau_m": self.tau_m,
            "tau_adaptation": self.tau_adaptation,
            "v_rest": self.v_rest,
            "resistance": self.resistance,
            "drive": self.drive,
            "adaptation_increment": self.adaptation_increment,
            "escape_rate": self.escape_rate,
            "v_escape": self.v_escape,
            "delta_v": self.delta_v,
            "v_reset": self.v_reset,
        }

    @property
    def positive_parameters(self) -> frozenset[str]:
        return frozenset(
            {
                "tau_m",
                "tau_adaptation",
                "resistance",
                "escape_rate",
                "delta_v",
            }
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("v", "w")

    def _validate_values(self, values: Mapping[str, float]) -> None:
        super()._validate_values(values)
        if values["tau_m"] == values["tau_adaptation"]:
            raise ResolutionError(
                "tau_adaptation must differ from tau_m for the current analytical capability"
            )
        if values["adaptation_increment"] < 0.0:
            raise ResolutionError("adaptation_increment must be nonnegative")

    def _default_initial(self, values: Mapping[str, float]) -> tuple[float, float]:
        return (values["v_rest"], 0.0)

    @property
    def source(self) -> str:
        values = self.defaults
        return f"""neuron StandardAdaptiveEscapeLIF {{
    params {{
        tau_m : positive = {_literal(values['tau_m'])}
        tau_adaptation : positive = {_literal(values['tau_adaptation'])}
        v_rest = {_literal(values['v_rest'])}
        resistance : positive = {_literal(values['resistance'])}
        drive = {_literal(values['drive'])}
        adaptation_increment = {_literal(values['adaptation_increment'])}
        escape_rate : positive = {_literal(values['escape_rate'])}
        v_escape = {_literal(values['v_escape'])}
        delta_v : positive = {_literal(values['delta_v'])}
        v_reset = {_literal(values['v_reset'])}
    }}
    state {{
        v : membrane
        w : adaptation
    }}
    dynamics {{
        dv/dt = -(v-v_rest)/tau_m + resistance*(drive-w)/tau_m
        dw/dt = -w/tau_adaptation
    }}
    hazard {{ rate = escape_rate*exp((v-v_escape)/delta_v) }}
    reset {{
        v <- v_reset
        w <- w + adaptation_increment
    }}
    refractory {{ duration = {_literal(self.refractory)} }}
}}
"""

    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedAdaptiveLIF:
        if numerical is not None:
            raise ResolutionError(
                "numerical configuration does not apply to analytical "
                "AdaptiveEscapeLIF"
            )
        return resolve_adaptive_escape_lif(self.authored_model, bindings)


@dataclass(frozen=True, kw_only=True)
class AdEx(StandardNeuron):
    """Adaptive exponential integrate-and-fire neuron.

    Parameter names follow the usual AdEx terms while using descriptive Python
    spelling.  ``drive`` is an applied current in the same units as ``w`` and
    ``leak_conductance * voltage``.  This nonlinear family resolves to ``STEPPED``.
    """

    name: str = "adex"
    capacitance: float = 200.0
    leak_conductance: float = 10.0
    v_rest: float = -70.0
    v_rheobase: float = -50.0
    slope_factor: float = 2.0
    tau_adaptation: float = 30.0
    subthreshold_adaptation: float = 2.0
    spike_adaptation: float = 40.0
    drive: float = 0.0
    v_reset: float = -58.0
    v_spike: float = -40.0
    refractory: float = 2.0

    def __post_init__(self) -> None:
        self._validate_definition()

    @property
    def defaults(self) -> Mapping[str, float]:
        return {
            "capacitance": self.capacitance,
            "leak_conductance": self.leak_conductance,
            "v_rest": self.v_rest,
            "v_rheobase": self.v_rheobase,
            "slope_factor": self.slope_factor,
            "tau_adaptation": self.tau_adaptation,
            "subthreshold_adaptation": self.subthreshold_adaptation,
            "spike_adaptation": self.spike_adaptation,
            "drive": self.drive,
            "v_reset": self.v_reset,
            "v_spike": self.v_spike,
        }

    @property
    def positive_parameters(self) -> frozenset[str]:
        return frozenset(
            {
                "capacitance",
                "leak_conductance",
                "slope_factor",
                "tau_adaptation",
            }
        )

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("v", "w")

    def _validate_values(self, values: Mapping[str, float]) -> None:
        super()._validate_values(values)
        if values["v_reset"] >= values["v_spike"]:
            raise ResolutionError("v_reset must be strictly below v_spike")
        if values["v_rheobase"] >= values["v_spike"]:
            raise ResolutionError("v_rheobase must be strictly below v_spike")

    def _default_initial(self, values: Mapping[str, float]) -> tuple[float, float]:
        return (values["v_rest"], 0.0)

    @property
    def source(self) -> str:
        values = self.defaults
        return f"""neuron StandardAdEx {{
    params {{
        capacitance : positive = {_literal(values['capacitance'])}
        leak_conductance : positive = {_literal(values['leak_conductance'])}
        v_rest = {_literal(values['v_rest'])}
        v_rheobase = {_literal(values['v_rheobase'])}
        slope_factor : positive = {_literal(values['slope_factor'])}
        tau_adaptation : positive = {_literal(values['tau_adaptation'])}
        subthreshold_adaptation = {_literal(values['subthreshold_adaptation'])}
        spike_adaptation = {_literal(values['spike_adaptation'])}
        drive = {_literal(values['drive'])}
        v_reset = {_literal(values['v_reset'])}
        v_spike = {_literal(values['v_spike'])}
    }}
    state {{
        v : membrane
        w : adaptation
    }}
    dynamics {{
        dv/dt = (-leak_conductance*(v-v_rest) + leak_conductance*slope_factor*exp((v-v_rheobase)/slope_factor) - w + drive)/capacitance
        dw/dt = (subthreshold_adaptation*(v-v_rest)-w)/tau_adaptation
    }}
    threshold {{ v > v_spike }}
    reset {{
        v <- v_reset
        w <- w + spike_adaptation
    }}
    refractory {{ duration = {_literal(self.refractory)} }}
}}
"""

    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedSteppedNeuron:
        return resolve_stepped_neuron(
            self.authored_model, bindings, numerical=numerical
        )


@dataclass(frozen=True, kw_only=True)
class QIF(StandardNeuron):
    """Normalized quadratic integrate-and-fire neuron.

    The generated normal form is
    ``tau_m * dv/dt = quadratic_gain*(v-v_center)^2 + drive``.
    Defaults reproduce ``dv/dt = v^2 + 1`` with a spike cutoff at one.
    """

    name: str = "qif"
    tau_m: float = 1.0
    quadratic_gain: float = 1.0
    v_center: float = 0.0
    drive: float = 1.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory: float = 0.0

    def __post_init__(self) -> None:
        self._validate_definition()

    @property
    def defaults(self) -> Mapping[str, float]:
        return {
            "tau_m": self.tau_m,
            "quadratic_gain": self.quadratic_gain,
            "v_center": self.v_center,
            "drive": self.drive,
            "v_threshold": self.v_threshold,
            "v_reset": self.v_reset,
        }

    @property
    def positive_parameters(self) -> frozenset[str]:
        return frozenset({"tau_m", "quadratic_gain"})

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("v",)

    def _validate_values(self, values: Mapping[str, float]) -> None:
        super()._validate_values(values)
        if values["v_reset"] >= values["v_threshold"]:
            raise ResolutionError("v_reset must be strictly below v_threshold")

    def _default_initial(self, values: Mapping[str, float]) -> float:
        return values["v_center"]

    @property
    def source(self) -> str:
        values = self.defaults
        return f"""neuron StandardQIF {{
    params {{
        tau_m : positive = {_literal(values['tau_m'])}
        quadratic_gain : positive = {_literal(values['quadratic_gain'])}
        v_center = {_literal(values['v_center'])}
        drive = {_literal(values['drive'])}
        v_threshold = {_literal(values['v_threshold'])}
        v_reset = {_literal(values['v_reset'])}
    }}
    state {{ v : membrane }}
    dynamics {{
        dv/dt = (quadratic_gain*(v-v_center)*(v-v_center) + drive)/tau_m
    }}
    threshold {{ v > v_threshold }}
    reset {{ v <- v_reset }}
    refractory {{ duration = {_literal(self.refractory)} }}
}}
"""

    def _resolve(
        self,
        bindings: Mapping[str, float],
        numerical: NumericalConfig | None,
    ) -> ResolvedSteppedNeuron:
        return resolve_stepped_neuron(
            self.authored_model, bindings, numerical=numerical
        )


# Descriptive aliases are useful in public annotations and generated help while the
# concise scientific names remain the primary construction API.
LeakyIntegrateAndFire = LIF
IF = IntegrateAndFire
AdaptiveLeakyIntegrateAndFire = AdaptiveLIF
EscapeLeakyIntegrateAndFire = EscapeLIF
AdaptiveEscapeLeakyIntegrateAndFire = AdaptiveEscapeLIF
AdaptiveExponentialIntegrateAndFire = AdEx
QuadraticIntegrateAndFire = QIF


__all__ = [
    "StandardNeuron",
    "IntegrateAndFire",
    "IF",
    "LIF",
    "AdaptiveLIF",
    "EscapeLIF",
    "AdaptiveEscapeLIF",
    "AdEx",
    "QIF",
    "LeakyIntegrateAndFire",
    "AdaptiveLeakyIntegrateAndFire",
    "EscapeLeakyIntegrateAndFire",
    "AdaptiveEscapeLeakyIntegrateAndFire",
    "AdaptiveExponentialIntegrateAndFire",
    "QuadraticIntegrateAndFire",
]
