"""Typed authoring and resolved records for the initial Lacuna slice."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .expr import ExprDAG


class ParameterDomain(str, Enum):
    """Validation domain for a bound model parameter."""

    FINITE = "finite"
    POSITIVE = "positive"


class StateRole(str, Enum):
    """Semantic role of one neuron state variable."""

    MEMBRANE = "membrane"
    RECEPTOR = "receptor"
    ADAPTATION = "adaptation"
    OBSERVER = "observer"
    AUX = "aux"


class DispatchForm(str, Enum):
    """Execution class selected by equation resolution."""

    REACTIVE = "REACTIVE"
    CLOSED_FORM = "CLOSED_FORM"
    ROOT_FIND = "ROOT_FIND"
    STEPPED = "STEPPED"


class HazardKind(str, Enum):
    """Intrinsic stochastic-spike process selected by equation resolution."""

    EXPONENTIAL_VOLTAGE = "EXPONENTIAL_VOLTAGE"


class ReactiveMode(str, Enum):
    """Subthreshold state policy for deposit-triggered integrate-and-fire nodes."""

    HOLD = "HOLD"
    RESET_BEFORE_DEPOSIT = "RESET_BEFORE_DEPOSIT"


class SynapseTier(str, Enum):
    """State ownership strategy for a resolved synapse."""

    DELTA = "DELTA"
    FOLDED_SHARED = "FOLDED_SHARED"
    PER_EDGE = "PER_EDGE"


class NeuronPolarity(str, Enum):
    """Outgoing weight policy, with Dale typing unless MIXED is explicit."""

    EXCITATORY = "EXCITATORY"
    INHIBITORY = "INHIBITORY"
    MIXED = "MIXED"

    @property
    def sign(self) -> float:
        """Return the stored-weight multiplier, preserving signed MIXED values."""

        return -1.0 if self is NeuronPolarity.INHIBITORY else 1.0

    @property
    def runtime_code(self) -> int:
        """Return the stable C descriptor value."""

        return {
            NeuronPolarity.EXCITATORY: 0,
            NeuronPolarity.INHIBITORY: 1,
            NeuronPolarity.MIXED: 2,
        }[self]


@dataclass(frozen=True)
class ParameterDefinition:
    """Name, default value, and validation domain of a parameter."""

    name: str
    default: float
    domain: ParameterDomain = ParameterDomain.FINITE


@dataclass(frozen=True)
class StateDefinition:
    """Name and semantic role of a neuron state variable."""

    name: str
    role: StateRole


@dataclass(frozen=True)
class ThresholdDefinition:
    """Readout expression and level that define a spike crossing."""

    readout: str
    level: str
    direction: str = "rising"


@dataclass(frozen=True)
class HazardDefinition:
    """Authored conditional intensity for an intrinsically stochastic neuron."""

    rate: str


@dataclass(frozen=True)
class ResolvedHazard:
    """Runtime form of ``rate = exp(log_scale + voltage_gain * v)``.

    The integrated-hazard scheduler uses the exact deterministic state trajectory
    selected for the neuron.  These tolerances bound only numerical quadrature and
    inversion of that continuous trajectory. There is no simulation timestep.
    """

    kind: HazardKind
    log_scale: float
    voltage_gain: float
    relative_tolerance: float = 1e-10
    absolute_tolerance: float = 1e-12
    time_tolerance: float = 1e-10
    maximum_quadrature_depth: int = 20
    maximum_root_iterations: int = 96
    trajectory_limit_root: str | None = None
    trajectory_coefficient_roots: tuple[str, ...] = ()
    trajectory_rate_roots: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixedRefractory:
    """Fixed refractory interval and its state policy."""

    duration: float
    mode: str = "CLAMP_RESET"


@dataclass(frozen=True)
class NumericalConfig:
    """Deterministic adaptive integration contract for generic ODE nodes."""

    relative_tolerance: float = 1e-8
    absolute_tolerance: float = 1e-10
    initial_step: float = 1e-3
    minimum_step: float = 1e-12
    maximum_step: float = 0.25
    event_tolerance: float = 1e-9
    maximum_steps: int = 1_000_000
    maximum_rhs_evaluations: int = 7_000_000


@dataclass(frozen=True)
class NeuronModel:
    """Authored neuron equations and event maps."""

    name: str
    parameters: tuple[ParameterDefinition, ...]
    states: tuple[StateDefinition, ...]
    dynamics: Mapping[str, str]
    threshold: ThresholdDefinition | None
    reset: Mapping[str, str]
    refractory: FixedRefractory | None = None
    reactive: ReactiveMode | None = None
    hazard: HazardDefinition | None = None


@dataclass(frozen=True)
class SynapseModel:
    """Authoring record for one intrinsic synapse definition."""

    name: str
    parameters: tuple[ParameterDefinition, ...]
    states: tuple[str, ...]
    dynamics: Mapping[str, str]
    spike_target: str
    spike_update: str
    outputs: Mapping[str, str]


@dataclass(frozen=True)
class ResolvedScalarLIF:
    """Executable scalar LIF specialization produced by the resolver.

    Symbolic strings remain authoritative for caching and diagnostics. Numeric fields
    are the current guarded runtime binding passed to the C evaluator.
    """

    model_name: str
    model_hash: str
    resolution_key: str
    state_name: str
    readout_index: int
    a_expr: str
    b_expr: str
    bindings: Mapping[str, float]
    a: float
    b: float
    threshold: float
    reset: float
    refractory: float
    dispatch: DispatchForm
    binding_dag: ExprDAG
    propagation_dag: ExprDAG
    normal_roots: tuple[str]
    clamped_roots: tuple[str]
    reset_roots: tuple[str]
    root_hint: "ScalarLogRootHint"
    drive_parameters: tuple[str, ...]
    parameter_domains: Mapping[str, ParameterDomain]
    hazard: ResolvedHazard | None = None

    @property
    def asymptote(self) -> float:
        """Return the scalar trajectory's steady-state value."""

        return -self.b / self.a


@dataclass(frozen=True)
class RootFindHint:
    """Expression roots and tolerances for alpha-kernel crossing."""

    g_root: str
    g_prime_root: str
    extremum_root: str
    extremum_prime_root: str
    asymptote_root: str
    membrane_coefficient_root: str
    synapse_constant_root: str
    synapse_linear_root: str
    membrane_rate_root: str
    synapse_rate_root: str
    threshold_root: str
    relative_tolerance: float
    fastest_time_constant: float
    iteration_cap: int


@dataclass(frozen=True)
class ScalarLogRootHint:
    """DAG roots required by the scalar logarithmic crossing capability."""

    decay_root: str
    affine_root: str
    threshold_root: str


@dataclass(frozen=True)
class TwoExpRootHint:
    """Roots and tolerances for a certified two-real-exponential crossing."""

    g_root: str
    g_prime_root: str
    limit_root: str
    coefficient_one_root: str
    coefficient_two_root: str
    rate_one_root: str
    rate_two_root: str
    relative_tolerance: float
    fastest_time_constant: float
    iteration_cap: int


@dataclass(frozen=True)
class MultiExpRootHint:
    """Roots for a bounded sum of distinct stable real exponential modes."""

    limit_root: str
    coefficient_roots: tuple[str, ...]
    rate_roots: tuple[str, ...]
    relative_tolerance: float
    fastest_time_constant: float
    iteration_cap: int


@dataclass(frozen=True)
class ExpPolyRootHint:
    """Bounded stable real exponential-polynomial crossing description."""

    limit_root: str
    rate_roots: tuple[str, ...]
    coefficient_roots: tuple[tuple[str, ...], ...]
    relative_tolerance: float
    fastest_time_constant: float
    iteration_cap: int


@dataclass(frozen=True)
class ResolvedAdaptiveLIF:
    """Stable LIF with one independently decaying spike-triggered current."""

    model_name: str
    model_hash: str
    resolution_key: str
    state_names: tuple[str, str]
    readout_index: int
    bindings: Mapping[str, float]
    a: float
    b: float
    coupling: float
    adaptation_decay: float
    adaptation_jump: float
    threshold: float
    reset: float
    refractory: float
    dispatch: DispatchForm
    propagation_dag: ExprDAG
    normal_roots: tuple[str, str]
    clamped_roots: tuple[str, str]
    reset_roots: tuple[str, str]
    root_hint: TwoExpRootHint
    drive_parameters: tuple[str, ...]
    parameter_domains: Mapping[str, ParameterDomain]
    hazard: ResolvedHazard | None = None

    @property
    def tau_m(self) -> float:
        """Return the resolved membrane time constant."""

        return -1.0 / self.a

    @property
    def tau_w(self) -> float:
        """Return the resolved adaptation time constant."""

        return -1.0 / self.adaptation_decay


@dataclass(frozen=True)
class ResolvedSteppedNeuron:
    """Generic finite-dimensional ODE lowered for execution by the C stepper."""

    model_name: str
    model_hash: str
    resolution_key: str
    state_names: tuple[str, ...]
    readout_index: int
    bindings: Mapping[str, float]
    threshold: float
    reset: float
    refractory: float
    dispatch: DispatchForm
    propagation_dag: ExprDAG
    normal_roots: tuple[str, ...]
    clamped_roots: tuple[str, ...]
    reset_roots: tuple[str, ...]
    numerical: NumericalConfig
    drive_parameters: tuple[str, ...]
    parameter_domains: Mapping[str, ParameterDomain]


@dataclass(frozen=True)
class ResolvedReactiveIF:
    """Event-batched integrate-and-fire node with no autonomous crossing.

    ``HOLD`` retains subthreshold state between event timestamps.  The reset mode
    applies the authored reset map immediately before all deposits at a timestamp
    are summed, matching timestamp-leak processors such as RISP.
    """

    model_name: str
    model_hash: str
    resolution_key: str
    state_names: tuple[str, ...]
    readout_index: int
    bindings: Mapping[str, float]
    threshold: float
    reset: float
    refractory: float
    dispatch: DispatchForm
    propagation_dag: ExprDAG
    normal_roots: tuple[str, ...]
    clamped_roots: tuple[str, ...]
    reset_roots: tuple[str, ...]
    reactive_mode: ReactiveMode
    drive_parameters: tuple[str, ...]
    parameter_domains: Mapping[str, ParameterDomain]


@dataclass(frozen=True)
class ResolvedAlphaLIF:
    """Propagation-ready scalar LIF augmented by one folded alpha kernel."""

    neuron_name: str
    synapse_name: str
    model_hash: str
    resolution_key: str
    state_names: tuple[str, str, str]
    readout_index: int
    bindings: Mapping[str, float]
    a: float
    b: float
    coupling: float
    synaptic_decay: float
    threshold: float
    reset: float
    refractory: float
    dispatch: DispatchForm
    tier: SynapseTier
    propagation_dag: ExprDAG
    deposit_dag: ExprDAG
    normal_roots: tuple[str, str, str]
    clamped_roots: tuple[str, str, str]
    reset_roots: tuple[str, str, str]
    deposit_root: str
    deposit_index: int
    root_hint: RootFindHint | ExpPolyRootHint
    drive_parameters: tuple[str, ...]
    parameter_domains: Mapping[str, ParameterDomain]

    @property
    def tau_m(self) -> float:
        """Return the resolved membrane time constant."""

        return -1.0 / self.a

    @property
    def tau_s(self) -> float:
        """Return the resolved synaptic time constant."""

        return -1.0 / self.synaptic_decay


@dataclass(frozen=True)
class ResolvedPerEdgeLIF:
    """Scalar LIF driven by bounded groups of linear edge-scoped currents."""

    neuron_name: str
    model_hash: str
    resolution_key: str
    state_names: tuple[str, ...]
    readout_index: int
    bindings: Mapping[str, float]
    a: float
    b: float
    coupling: float
    synaptic_decays: tuple[float, ...]
    threshold: float
    reset: float
    refractory: float
    dispatch: DispatchForm
    tier: SynapseTier
    propagation_dag: ExprDAG
    normal_roots: tuple[str, ...]
    clamped_roots: tuple[str, ...]
    reset_roots: tuple[str, ...]
    root_hint: RootFindHint | MultiExpRootHint | ExpPolyRootHint
    drive_parameters: tuple[str, ...]
    parameter_domains: Mapping[str, ParameterDomain]
    edge_deposits: Mapping[int, tuple[int, float]]
    group_edge_ids: tuple[tuple[int, ...], ...]
    group_initials: tuple[float, ...]
