"""Model-neutral execution plans derived from resolved equations and graph wiring.

This module is deliberately independent of the C ABI.  It describes what must be
executed without choosing a concrete runtime representation.  The current runtime
continues to consume the established resolved records while this plan becomes the
stable boundary between mathematical resolution and backend lowering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .precision import PrecisionProfile, require_supported_precision
from .expr import ExprDAG
from .ir import (
    DispatchForm,
    ExpPolyRootHint,
    MultiExpRootHint,
    NeuronPolarity,
    NumericalConfig,
    ParameterDomain,
    ReactiveMode,
    RootFindHint,
    ScalarLogRootHint,
    TwoExpRootHint,
)
from .learning import LearningProgram, resolve_learning

if TYPE_CHECKING:
    from .graph import ResolvedGraph


class EvolutionMethod(str, Enum):
    """How a state program advances between event timestamps."""

    EXACT_EXPRESSIONS = "EXACT_EXPRESSIONS"
    EVENT_BATCHED = "EVENT_BATCHED"
    NUMERICAL_ODE = "NUMERICAL_ODE"


class CrossingMethod(str, Enum):
    """Mathematical procedure used to locate the next threshold crossing."""

    NONE = "NONE"
    SCALAR_LOG = "SCALAR_LOG"
    BRACKETED_EXTREMA = "BRACKETED_EXTREMA"
    TWO_EXPONENTIAL = "TWO_EXPONENTIAL"
    MULTI_EXPONENTIAL = "MULTI_EXPONENTIAL"
    EXPONENTIAL_POLYNOMIAL = "EXPONENTIAL_POLYNOMIAL"
    NUMERICAL_EVENT = "NUMERICAL_EVENT"
    INTEGRATED_HAZARD = "INTEGRATED_HAZARD"


class DepositOperation(str, Enum):
    """Operation performed when a connection delivery reaches its target."""

    ADD_STATE = "ADD_STATE"
    EVALUATE_PROGRAM = "EVALUATE_PROGRAM"


class ArithmeticMethod(str, Enum):
    """Equation-derived arithmetic operation used by the shared scheduler."""

    EXPRESSION_DAG = "EXPRESSION_DAG"
    SCALAR_AFFINE = "SCALAR_AFFINE"


@dataclass(frozen=True)
class DepositProgram:
    """Equation program used for non-additive spike deposits."""

    expressions: ExprDAG
    root: str
    target_state: int


@dataclass(frozen=True)
class DriveParameter:
    """One runtime-mutable equation parameter and its accepted public names."""

    name: str
    aliases: tuple[str, ...]
    parameter_index: int
    domain: ParameterDomain


@dataclass(frozen=True)
class EquationProgram:
    """Reusable equation program shared by nodes with one resolved structure."""

    id: int
    key: str
    state_names: tuple[str, ...]
    evolution: EvolutionMethod
    crossing: CrossingMethod
    propagation: ExprDAG
    normal_roots: tuple[str, ...]
    clamped_roots: tuple[str, ...]
    reset_roots: tuple[str, ...]
    readout_index: int
    drive_parameters: tuple[str, ...]
    drive_bindings: tuple[DriveParameter, ...]
    deposit: DepositProgram | None

    @property
    def state_count(self) -> int:
        """Return the number of physical states in this equation program."""

        return len(self.state_names)


@dataclass(frozen=True)
class ScalarAffineParameters:
    """Per-node numeric values for the compact one-state affine lowering."""

    decay: float
    drive: float
    reset: float


@dataclass(frozen=True)
class StateSlot:
    """One physically stored state value in the compiled node-state array."""

    index: int
    node: int
    local_index: int
    name: str
    initial: float


@dataclass(frozen=True)
class NodeExecution:
    """Per-node storage, bindings, and numeric boundary values."""

    index: int
    source_id: int
    program: int
    state_offset: int
    state_count: int
    readout_index: int
    parameter_values: tuple[float, ...]
    threshold: float
    refractory: float
    polarity: NeuronPolarity
    autonomous_crossing: bool
    crossing_hint: object | None
    numerical: NumericalConfig | None
    reactive_mode: ReactiveMode | None
    arithmetic: ArithmeticMethod
    scalar_affine: ScalarAffineParameters | None
    deposit_parameter_values: tuple[float, ...]
    hazard: object | None


@dataclass(frozen=True)
class ConnectionExecution:
    """Runtime-relevant connection data and its delivery operation."""

    index: int
    source_id: int
    pre: int
    post: int
    weight: float
    delay: float
    operation: DepositOperation
    target_state: int
    scale: float
    weight_group: int | None
    learning_program: int | None
    learning_parameter_values: tuple[float, ...]
    learning_weight_bounds: tuple[float, float] | None
    modulator: int | None

    @property
    def has_learning(self) -> bool:
        """Return whether this connection has an edge-learning program."""

        return self.learning_program is not None


@dataclass(frozen=True)
class InputExecution:
    """One named external input and its encoding/binding configuration."""

    id: str
    node: int
    mode: object
    parameter: str | None
    encoder: object


@dataclass(frozen=True)
class OutputExecution:
    """One named output tap and optional decoder configuration."""

    id: str
    node: int
    decoder: object | None


@dataclass(frozen=True)
class ModulatorExecution:
    """One named third-factor input and the compiled connections it targets."""

    id: str
    connections: tuple[int, ...]


@dataclass(frozen=True)
class NodeExecutionBatch:
    """Nodes sharing one equation program and runtime execution strategy."""

    id: int
    program: int
    evolution: EvolutionMethod
    crossing: CrossingMethod
    autonomous_crossing: bool
    reactive_mode: ReactiveMode | None
    arithmetic: ArithmeticMethod
    nodes: tuple[int, ...]


@dataclass(frozen=True)
class ConnectionExecutionBatch:
    """Connections sharing one delivery and learning execution strategy."""

    id: int
    operation: DepositOperation
    learning_program: int | None
    shared_weight: bool
    modulated: bool
    connections: tuple[int, ...]


@dataclass(frozen=True)
class PlanRequirements:
    """Features attached outside intrinsic state propagation in this first slice."""

    learning_connections: int
    modulators: int
    decoders: int


@dataclass(frozen=True)
class ExecutionPlan:
    """Backend-neutral physical plan for a fully resolved network graph."""

    programs: tuple[EquationProgram, ...]
    learning_programs: tuple[LearningProgram, ...]
    states: tuple[StateSlot, ...]
    nodes: tuple[NodeExecution, ...]
    connections: tuple[ConnectionExecution, ...]
    inputs: tuple[InputExecution, ...]
    outputs: tuple[OutputExecution, ...]
    modulators: tuple[ModulatorExecution, ...]
    node_batches: tuple[NodeExecutionBatch, ...]
    connection_batches: tuple[ConnectionExecutionBatch, ...]
    requirements: PlanRequirements
    precision: PrecisionProfile = PrecisionProfile.FLOAT64
    target_binding_key: str | None = None

    def __post_init__(self) -> None:
        # Reduced-precision plans require target-aware equation resolution first.
        object.__setattr__(self, "precision", require_supported_precision(self.precision))
        if self.precision is not PrecisionProfile.FLOAT64 and not self.target_binding_key:
            raise ValueError("reduced-precision plans require target-aware resolution")

    @property
    def precision_key(self) -> tuple[str, int, int, int]:
        """Identify the arithmetic contract, not the network or model identity."""

        return self.precision.cache_key

    @property
    def state_count(self) -> int:
        """Return the total number of physical state slots."""

        return len(self.states)

    @property
    def compact_scalar_delta_compatible(self) -> bool:
        """Whether the current compact scalar/delta C lowering can execute it."""

        if (
            self.requirements.learning_connections
            or self.requirements.modulators
            or self.requirements.decoders
        ):
            return False
        if any(
            node.state_count != 1
            or node.arithmetic is not ArithmeticMethod.SCALAR_AFFINE
            or node.scalar_affine is None
            or self.programs[node.program].evolution
            is not EvolutionMethod.EXACT_EXPRESSIONS
            or self.programs[node.program].crossing is not CrossingMethod.SCALAR_LOG
            for node in self.nodes
        ):
            return False
        return all(
            connection.operation is DepositOperation.ADD_STATE
            and connection.target_state
            == self.nodes[connection.post].readout_index
            and connection.scale == 1.0
            for connection in self.connections
        )


def _state_names(model: object) -> tuple[str, ...]:
    singular = getattr(model, "state_name", None)
    if singular is not None:
        return (str(singular),)
    return tuple(str(name) for name in getattr(model, "state_names"))


def _crossing_method(model: object) -> CrossingMethod:
    if getattr(model, "hazard", None) is not None:
        return CrossingMethod.INTEGRATED_HAZARD
    hint = getattr(model, "root_hint", None)
    if isinstance(hint, ScalarLogRootHint):
        return CrossingMethod.SCALAR_LOG
    if isinstance(hint, TwoExpRootHint):
        return CrossingMethod.TWO_EXPONENTIAL
    if isinstance(hint, MultiExpRootHint):
        return CrossingMethod.MULTI_EXPONENTIAL
    if isinstance(hint, ExpPolyRootHint):
        return CrossingMethod.EXPONENTIAL_POLYNOMIAL
    if isinstance(hint, RootFindHint):
        return CrossingMethod.BRACKETED_EXTREMA
    dispatch = getattr(model, "dispatch")
    if dispatch is DispatchForm.STEPPED:
        return CrossingMethod.NUMERICAL_EVENT
    if dispatch is DispatchForm.REACTIVE:
        return CrossingMethod.NONE
    raise ValueError("resolved equation program has no crossing procedure")


def _evolution_method(model: object) -> EvolutionMethod:
    if getattr(model, "dispatch") is DispatchForm.STEPPED:
        return EvolutionMethod.NUMERICAL_ODE
    if hasattr(model, "reactive_mode"):
        return EvolutionMethod.EVENT_BATCHED
    return EvolutionMethod.EXACT_EXPRESSIONS


def _initial_values(value: float | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(item) for item in value) if isinstance(value, tuple) else (float(value),)


def _program(model: object, program_id: int) -> EquationProgram:
    deposit_dag = getattr(model, "deposit_dag", None)
    deposit = (
        DepositProgram(
            expressions=deposit_dag,
            root=str(getattr(model, "deposit_root")),
            target_state=int(getattr(model, "deposit_index")),
        )
        if deposit_dag is not None
        else None
    )
    propagation = getattr(model, "propagation_dag")
    drive_parameters = tuple(getattr(model, "drive_parameters"))
    parameter_domains = getattr(model, "parameter_domains")
    drive_bindings = tuple(
        DriveParameter(
            name=name,
            aliases=(name, name.split(".", 1)[1]) if name.startswith("neuron.") else (name,),
            parameter_index=propagation.parameters.index(name),
            domain=parameter_domains[name],
        )
        for name in drive_parameters
    )
    return EquationProgram(
        id=program_id,
        key=str(getattr(model, "resolution_key")),
        state_names=_state_names(model),
        evolution=_evolution_method(model),
        crossing=_crossing_method(model),
        propagation=propagation,
        normal_roots=tuple(getattr(model, "normal_roots")),
        clamped_roots=tuple(getattr(model, "clamped_roots")),
        reset_roots=tuple(getattr(model, "reset_roots")),
        readout_index=int(getattr(model, "readout_index")),
        drive_parameters=drive_parameters,
        drive_bindings=drive_bindings,
        deposit=deposit,
    )


def lower_execution_plan(
    resolved: ResolvedGraph,
    *,
    precision: PrecisionProfile | str = PrecisionProfile.FLOAT64,
) -> ExecutionPlan:
    """Lower a resolved graph into the backend-neutral execution-plan IR."""

    from .graph import ResolvedGraph

    profile = require_supported_precision(precision)
    if not isinstance(resolved, ResolvedGraph):
        raise TypeError("lowering requires a ResolvedGraph")
    if profile is not resolved.precision:
        raise ValueError("execution plan precision differs from resolved graph precision")
    programs: list[EquationProgram] = []
    program_by_key: dict[str, int] = {}
    nodes: list[NodeExecution] = []
    states: list[StateSlot] = []
    state_offset = 0

    for index, (source_id, model, initial) in enumerate(
        zip(resolved.node_ids, resolved.models, resolved.initial_values)
    ):
        key = str(getattr(model, "resolution_key"))
        program_id = program_by_key.get(key)
        if program_id is None:
            program_id = len(programs)
            program_by_key[key] = program_id
            programs.append(_program(model, program_id))
        program = programs[program_id]
        values = _initial_values(initial)
        if len(values) != program.state_count:
            raise ValueError("resolved initial state does not match its equation program")
        for local_index, (name, value) in enumerate(zip(program.state_names, values)):
            states.append(
                StateSlot(state_offset + local_index, index, local_index, name, value)
            )
        dag = program.propagation
        bindings = getattr(model, "bindings")
        hint = getattr(model, "root_hint", None)
        deposit_parameters = (
            ()
            if program.deposit is None
            else tuple(
                float(bindings[name])
                for name in program.deposit.expressions.parameters
            )
        )
        scalar_affine = None
        if (
            program.state_count == 1
            and program.evolution is EvolutionMethod.EXACT_EXPRESSIONS
            and all(hasattr(model, name) for name in ("a", "b", "reset"))
        ):
            scalar_affine = ScalarAffineParameters(
                float(getattr(model, "a")),
                float(getattr(model, "b")),
                float(getattr(model, "reset")),
            )
        arithmetic = (
            ArithmeticMethod.SCALAR_AFFINE
            if scalar_affine is not None
            else ArithmeticMethod.EXPRESSION_DAG
        )
        nodes.append(
            NodeExecution(
                index=index,
                source_id=source_id,
                program=program_id,
                state_offset=state_offset,
                state_count=program.state_count,
                readout_index=program.readout_index,
                parameter_values=tuple(float(bindings[name]) for name in dag.parameters),
                threshold=float(getattr(model, "threshold")),
                refractory=float(getattr(model, "refractory")),
                polarity=resolved.polarities[index],
                autonomous_crossing=getattr(model, "dispatch")
                is not DispatchForm.REACTIVE,
                crossing_hint=hint,
                numerical=getattr(model, "numerical", None),
                reactive_mode=getattr(model, "reactive_mode", None),
                arithmetic=arithmetic,
                scalar_affine=scalar_affine,
                deposit_parameter_values=deposit_parameters,
                hazard=getattr(model, "hazard", None),
            )
        )
        state_offset += program.state_count

    modulator_by_edge = {
        edge: modulator
        for modulator, port in enumerate(resolved.graph.modulator_ports)
        for edge in port.edges
    }
    learning_programs: list[LearningProgram] = []
    learning_program_by_key: dict[str, int] = {}
    connections_list: list[ConnectionExecution] = []
    for index, (authored, edge) in enumerate(
        zip(resolved.graph.edges, resolved.edges)
    ):
        learning_program = None
        learning_parameter_values: tuple[float, ...] = ()
        learning_weight_bounds = None
        if authored.plasticity is not None:
            learned = resolve_learning(authored.plasticity)
            learning_program = learning_program_by_key.get(learned.program.key)
            if learning_program is None:
                learning_program = len(learning_programs)
                learning_program_by_key[learned.program.key] = learning_program
                learning_programs.append(learned.program)
            learning_parameter_values = learned.parameter_values
            learning_weight_bounds = learned.weight_bounds
        connections_list.append(
            ConnectionExecution(
                index=index,
                source_id=authored.id,
                pre=edge.pre,
                post=edge.post,
                weight=edge.weight,
                delay=edge.delay,
                operation=(
                    DepositOperation.ADD_STATE
                    if edge.deposit_kind.name == "STATE_ADD"
                    else DepositOperation.EVALUATE_PROGRAM
                ),
                target_state=edge.target,
                scale=edge.deposit_scale,
                weight_group=authored.weight_group,
                learning_program=learning_program,
                learning_parameter_values=learning_parameter_values,
                learning_weight_bounds=learning_weight_bounds,
                modulator=modulator_by_edge.get(authored.id),
            )
        )
    connections = tuple(connections_list)
    requirements = PlanRequirements(
        learning_connections=sum(edge.has_learning for edge in connections),
        modulators=len(resolved.graph.modulator_ports),
        decoders=sum(
            port.decoder is not None for port in resolved.graph.output_ports
        ),
    )
    node_by_source_id = {source_id: index for index, source_id in enumerate(resolved.node_ids)}
    connection_by_source_id = {
        connection.source_id: connection.index for connection in connections
    }
    inputs = tuple(
        InputExecution(
            id=port.id,
            node=node_by_source_id[port.node],
            mode=port.mode,
            parameter=port.parameter,
            encoder=port.encoder,
        )
        for port in resolved.graph.input_ports
    )
    outputs = tuple(
        OutputExecution(
            id=port.id,
            node=node_by_source_id[port.node],
            decoder=port.decoder,
        )
        for port in resolved.graph.output_ports
    )
    modulators = tuple(
        ModulatorExecution(
            id=port.id,
            connections=tuple(connection_by_source_id[edge] for edge in port.edges),
        )
        for port in resolved.graph.modulator_ports
    )
    node_batch_members: dict[
        tuple[
            int,
            EvolutionMethod,
            CrossingMethod,
            bool,
            ReactiveMode | None,
            ArithmeticMethod,
        ],
        list[int],
    ] = {}
    for node in nodes:
        program = programs[node.program]
        key = (
            node.program,
            program.evolution,
            program.crossing,
            node.autonomous_crossing,
            node.reactive_mode,
            node.arithmetic,
        )
        node_batch_members.setdefault(key, []).append(node.index)
    node_batches = tuple(
        NodeExecutionBatch(
            id=batch_id,
            program=key[0],
            evolution=key[1],
            crossing=key[2],
            autonomous_crossing=key[3],
            reactive_mode=key[4],
            arithmetic=key[5],
            nodes=tuple(members),
        )
        for batch_id, (key, members) in enumerate(node_batch_members.items())
    )
    connection_batch_members: dict[
        tuple[DepositOperation, int | None, bool, bool], list[int]
    ] = {}
    for connection in connections:
        key = (
            connection.operation,
            connection.learning_program,
            connection.weight_group is not None,
            connection.modulator is not None,
        )
        connection_batch_members.setdefault(key, []).append(connection.index)
    connection_batches = tuple(
        ConnectionExecutionBatch(
            id=batch_id,
            operation=key[0],
            learning_program=key[1],
            shared_weight=key[2],
            modulated=key[3],
            connections=tuple(members),
        )
        for batch_id, (key, members) in enumerate(connection_batch_members.items())
    )
    return ExecutionPlan(
        programs=tuple(programs),
        learning_programs=tuple(learning_programs),
        states=tuple(states),
        nodes=tuple(nodes),
        connections=connections,
        inputs=inputs,
        outputs=outputs,
        modulators=modulators,
        node_batches=node_batches,
        connection_batches=connection_batches,
        requirements=requirements,
        precision=profile,
        target_binding_key=resolved.target_binding_key,
    )


__all__ = [
    "ConnectionExecution",
    "ConnectionExecutionBatch",
    "CrossingMethod",
    "DepositOperation",
    "DepositProgram",
    "DriveParameter",
    "EquationProgram",
    "EvolutionMethod",
    "ExecutionPlan",
    "InputExecution",
    "ModulatorExecution",
    "NodeExecution",
    "NodeExecutionBatch",
    "OutputExecution",
    "PlanRequirements",
    "ScalarAffineParameters",
    "StateSlot",
    "lower_execution_plan",
]
