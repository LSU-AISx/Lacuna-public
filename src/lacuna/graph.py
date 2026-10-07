"""Flat graph IR, named ports, and deterministic self-contained serialization."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Sequence

from .codec import (
    BurstEncoder,
    DecodeEventKind,
    DecodeQuery,
    DecodeWindow,
    Decoder,
    DecoderBinding,
    DecoderQueryBinding,
    DecoderWindowBinding,
    Encoder,
    EmissionPolicy,
    HeldCurrentEncoder,
    LatencyBurstEncoder,
    NativeEventEncoder,
    PoissonRateEncoder,
    Presentation,
    RateDecoder,
    RateMode,
    RegularRateEncoder,
    TemporalSpikeMode,
    TemporalWeightDecoder,
    TTFSEncoder,
    TTFSDecoder,
)
from .dsl import parse_neuron, parse_synapse
from .precision import PrecisionProfile
from .errors import CapabilityError, DSLParseError, ResolutionError
from .ffi import (
    CompiledDecoderBank,
    CompiledGraph as CoreCompiledGraph,
    CoreEvaluator,
    DepositKind,
    IncrementalCompiledRun,
    MixedDriveUpdate,
    MixedEdge,
    MixedInputSpike,
    MixedRunResult,
    ModulationEvent,
    RecordingConfig,
    RunResult,
    StateInspection,
    StateInspectionRequest,
    StreamingEncoderRun,
    TraceRecord,
)
from .ir import (
    NeuronModel,
    NeuronPolarity,
    ParameterDomain,
    ResolvedAdaptiveLIF,
    ResolvedAlphaLIF,
    ResolvedPerEdgeLIF,
    ResolvedReactiveIF,
    ResolvedScalarLIF,
    ResolvedSteppedNeuron,
    StateRole,
    SynapseModel,
)
from .resolver import (
    PerEdgeSynapseInstance,
    resolve_adaptive_escape_lif,
    resolve_adaptive_lif,
    resolve_escape_lif,
    resolve_folded_alpha_lif,
    resolve_per_edge_lif,
    resolve_reactive_if,
    resolve_scalar_lif,
    resolve_stepped_neuron,
)
from .plasticity import (
    ModulatedSTDP,
    PairSTDP,
    PlasticityRule,
    SoftExcursionModulated,
    TripletSTDP,
    VoltageModulatedSTDP,
    plasticity_from_document,
    plasticity_to_document,
)

if TYPE_CHECKING:
    from .execution_plan import ExecutionPlan

GRAPH_SCHEMA_VERSION = 10
MIXED_GRAPH_SCHEMA_VERSION = 11
SUPPORTED_GRAPH_SCHEMA_VERSIONS = frozenset(
    (9, GRAPH_SCHEMA_VERSION, MIXED_GRAPH_SCHEMA_VERSION)
)


class InputMode(str, Enum):
    """External input behavior exposed by a named port."""

    SPIKE = "SPIKE"
    DRIVE = "DRIVE"


@dataclass(frozen=True)
class GraphModel:
    """Named neuron model source embedded in a graph."""

    id: str
    source: str


@dataclass(frozen=True)
class GraphSynapse:
    """Named synapse model source embedded in a graph."""

    id: str
    source: str


@dataclass(frozen=True)
class GraphNode:
    """Flat neuron instance with resolved authoring bindings."""

    id: int
    model: str
    initial: float | tuple[float, ...]
    bindings: Mapping[str, float]
    synapse: str | None = None
    receptor: str | None = None
    output: str | None = None
    synapse_bindings: Mapping[str, float] | None = None
    polarity: NeuronPolarity = NeuronPolarity.EXCITATORY


@dataclass(frozen=True)
class GraphEdge:
    """Directed connection with a magnitude or a signed MIXED-source weight."""

    id: int
    pre: int
    post: int
    weight: float
    delay: float = 0.0
    synapse: str | None = None
    receptor: str | None = None
    output: str | None = None
    synapse_bindings: Mapping[str, float] | None = None
    initial: float | tuple[float, ...] = 0.0
    plasticity: PlasticityRule | None = None
    weight_group: int | None = None


@dataclass(frozen=True)
class InputPort:
    """Named external input bound to one graph node."""

    id: str
    node: int
    mode: InputMode
    parameter: str | None = None
    encoder: Encoder = NativeEventEncoder()


@dataclass(frozen=True)
class OutputPort:
    """Named spike output with an optional decoder."""

    id: str
    node: int
    decoder: Decoder | None = None


@dataclass(frozen=True)
class ModulatorPort:
    """Named third-factor input scoped to an explicit set of graph edges."""

    id: str
    edges: tuple[int, ...]


@dataclass(frozen=True)
class SpikeInput:
    """Timestamped spike submitted through an input port."""

    t: float
    port: str
    value: float


@dataclass(frozen=True)
class DriveInput:
    """Timestamped parameter update submitted through an input port."""

    t: float
    port: str
    value: float


@dataclass(frozen=True)
class ScalarInput:
    """Scalar presentation encoded over a half-open interval."""

    t_start: float
    t_end: float
    port: str
    value: float


@dataclass(frozen=True)
class ModulationInput:
    """Timestamped third factor submitted through a modulator port."""

    t: float
    port: str
    value: float


@dataclass(frozen=True)
class PortSpike:
    """Output spike mapped back to its named port."""

    t: float
    port: str
    node: int


@dataclass(frozen=True)
class DecodedPort:
    """Final decoder value mapped back to its named output port."""

    port: str
    node: int
    valid: bool
    count: int
    value: float | None
    first_spike: float | None
    window: int = 0
    window_start: float | None = None
    window_end: float | None = None


@dataclass(frozen=True)
class DecodedPortEvent:
    """Streaming decoder event mapped to its named output port."""

    port: str
    node: int
    window: int
    kind: DecodeEventKind
    valid: bool
    emitted_at: float
    source_spike_time: float | None
    window_start: float
    window_end: float
    observed_through: float
    count: int
    value: float | None
    first_spike: float | None


@dataclass(frozen=True)
class GraphRunResult:
    """Core results projected onto graph ports and recording requests."""

    core: RunResult | MixedRunResult
    outputs: tuple[PortSpike, ...]
    decoded: tuple[DecodedPort, ...] = ()
    decoded_events: tuple[DecodedPortEvent, ...] = ()
    trace: tuple[TraceRecord, ...] = ()
    inspections: tuple[StateInspection, ...] = ()


@dataclass(frozen=True)
class Graph:
    """Serializable flat graph with named models, nodes, edges, and ports."""

    models: tuple[GraphModel, ...]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...] = ()
    input_ports: tuple[InputPort, ...] = ()
    output_ports: tuple[OutputPort, ...] = ()
    time_unit: str = "ms"
    synapses: tuple[GraphSynapse, ...] = ()
    modulator_ports: tuple[ModulatorPort, ...] = ()

    def resolve(self) -> "ResolvedGraph":
        """Validate and resolve the graph into executable model records."""

        return resolve_graph(self)

    def to_text(self) -> str:
        """Serialize the graph as deterministic JSON text."""

        return graph_to_text(self)

    @classmethod
    def from_text(cls, text: str) -> "Graph":
        """Load a graph from deterministic JSON text."""

        return graph_from_text(text)


@dataclass(frozen=True)
class ResolvedGraph:
    """Validated graph with executable node models and lowered edges."""

    graph: Graph
    node_ids: tuple[int, ...]
    models: tuple[
        ResolvedScalarLIF
        | ResolvedAlphaLIF
        | ResolvedAdaptiveLIF
        | ResolvedPerEdgeLIF
        | ResolvedReactiveIF
        | ResolvedSteppedNeuron,
        ...,
    ]
    initial_values: tuple[float | tuple[float, ...], ...]
    edges: tuple[MixedEdge, ...]
    _parsed_models: Mapping[str, NeuronModel]
    _parsed_synapses: Mapping[str, SynapseModel]
    _node_bindings: tuple[Mapping[str, float], ...]
    _node_synapse_bindings: tuple[Mapping[str, float], ...]
    _edge_synapse_bindings: tuple[Mapping[str, float], ...]
    _node_index: Mapping[int, int]
    _input_ports: Mapping[str, InputPort]
    _modulator_ports: Mapping[str, int]
    precision: PrecisionProfile = PrecisionProfile.FLOAT64
    target_binding_key: str | None = None

    @property
    def polarities(self) -> tuple[NeuronPolarity, ...]:
        """Return node polarities in compiled node order."""

        return tuple(node.polarity for node in self.graph.nodes)

    @property
    def effective_edges(self) -> tuple[MixedEdge, ...]:
        """Return signed diagnostics without changing stored edge weights."""

        return tuple(
            replace(
                edge,
                weight=(
                    edge.weight
                    * edge.deposit_scale
                    * self.graph.nodes[edge.pre].polarity.sign
                ),
                deposit_scale=1.0,
            )
            for edge in self.edges
        )

    def compile(self, core: CoreEvaluator) -> "CompiledResolvedGraph":
        """Prepare this graph once while preserving its named-port interface."""

        if self.precision is not core.precision:
            if self.precision is not PrecisionProfile.FLOAT64:
                raise ResolutionError("resolved graph and runtime precision differ")
            from .target_lowering import resolve_target_graph

            return resolve_target_graph(self.graph, core).compile(core)
        plan = self.execution_plan()
        compiled, decoders = self._compile_execution_plan(
            core, execution_plan=plan
        )
        return CompiledResolvedGraph(
            self,
            compiled,
            decoders,
            core=core,
            execution_plan=plan,
        )

    def execution_plan(self) -> "ExecutionPlan":
        """Derive the backend-neutral physical plan for this resolved graph."""

        from .execution_plan import lower_execution_plan

        return lower_execution_plan(self, precision=self.precision)

    @property
    def scalar_delta_eligible(self) -> bool:
        """Compatibility alias for the first compact C lowering."""

        return self.execution_plan().compact_scalar_delta_compatible

    def _compile_execution_plan(
        self,
        core: CoreEvaluator,
        *,
        execution_plan: "ExecutionPlan | None" = None,
    ) -> tuple[CoreCompiledGraph, CompiledDecoderBank | None]:
        """Compile the equation-derived graph and any decoder bank."""

        plan = self.execution_plan() if execution_plan is None else execution_plan
        compiled = core.compile_execution_plan(
            plan,
        )
        try:
            bindings = self._decoder_bindings()
            decoders = (
                core.compile_decoders(bindings, node_count=len(self.models))
                if bindings
                else None
            )
        except Exception:
            compiled.close()
            raise
        return compiled, decoders

    def _decoder_bindings(self) -> tuple[DecoderBinding, ...]:
        return tuple(
            DecoderBinding(self._node_index[port.node], port.decoder)
            for port in self.graph.output_ports
            if port.decoder is not None
        )

    def _encoded_ports(self) -> tuple[InputPort, ...]:
        return tuple(
            port
            for port in self.graph.input_ports
            if not isinstance(port.encoder, NativeEventEncoder)
        )

    def _prepare_presentations(
        self,
        scalar_inputs: Sequence[ScalarInput],
        encoded_ports: Sequence[InputPort],
    ) -> tuple[Presentation, ...]:
        encoded_port_index = {
            port.id: index for index, port in enumerate(encoded_ports)
        }
        presentations: list[Presentation] = []
        for event in scalar_inputs:
            port = self._input_ports.get(event.port)
            if port is None:
                raise ResolutionError(f"unknown input port '{event.port}'")
            if isinstance(port.encoder, NativeEventEncoder):
                raise ResolutionError(
                    f"input port '{port.id}' uses native event passthrough, not scalar encoding"
                )
            if (
                not math.isfinite(float(event.t_start))
                or not math.isfinite(float(event.t_end))
                or not float(event.t_start) < float(event.t_end)
                or not math.isfinite(float(event.value))
                or not 0.0 <= float(event.value) <= 1.0
            ):
                raise ResolutionError(
                    f"scalar input '{port.id}' requires finite start < end and value in [0, 1]"
                )
            presentations.append(
                Presentation(
                    float(event.t_start),
                    float(event.t_end),
                    encoded_port_index[port.id],
                    float(event.value),
                )
            )
        presentations.sort(key=lambda item: (item.t_start, item.encoder))
        return tuple(presentations)

    def _validate_native_inputs(
        self,
        spike_inputs: Sequence[SpikeInput],
        drive_inputs: Sequence[DriveInput],
    ) -> None:
        for event in spike_inputs:
            port = self._input_ports.get(event.port)
            if port is not None and not isinstance(port.encoder, NativeEventEncoder):
                raise ResolutionError(
                    f"input port '{port.id}' requires a scalar presentation for its encoder"
                )
        for event in drive_inputs:
            port = self._input_ports.get(event.port)
            if port is not None and not isinstance(port.encoder, NativeEventEncoder):
                raise ResolutionError(
                    f"input port '{port.id}' requires a scalar presentation for its encoder"
                )

    def _map_input_primitives(
        self,
        spike_inputs: Sequence[SpikeInput],
        drive_inputs: Sequence[DriveInput],
    ) -> tuple[list[MixedInputSpike], list[MixedDriveUpdate]]:
        mixed_inputs: list[MixedInputSpike] = []
        ordered_spikes = sorted(
            enumerate(spike_inputs),
            key=lambda item: (item[1].t, item[0]),
        )
        for _, event in ordered_spikes:
            port = self._input_ports.get(event.port)
            if port is None:
                raise ResolutionError(f"unknown input port '{event.port}'")
            if port.mode is not InputMode.SPIKE:
                raise ResolutionError(f"input port '{event.port}' is not a SPIKE port")
            index = self._node_index[port.node]
            model = self.models[index]
            value = float(event.value)
            if not math.isfinite(value):
                raise ResolutionError(f"spike input '{port.id}' must be finite")
            mixed_inputs.append(
                MixedInputSpike(
                    float(event.t),
                    index,
                    value,
                    DepositKind.PROGRAM
                    if isinstance(model, ResolvedAlphaLIF)
                    else DepositKind.STATE_ADD,
                    model.deposit_index
                    if isinstance(model, ResolvedAlphaLIF)
                    else model.readout_index,
                )
            )

        current_bindings = [dict(bindings) for bindings in self._node_bindings]
        ordered_drives = sorted(
            enumerate(drive_inputs),
            key=lambda item: (item[1].t, item[0]),
        )
        mixed_drives: list[MixedDriveUpdate] = []
        for _, event in ordered_drives:
            port = self._input_ports.get(event.port)
            if port is None:
                raise ResolutionError(f"unknown input port '{event.port}'")
            if port.mode is not InputMode.DRIVE or port.parameter is None:
                raise ResolutionError(f"input port '{event.port}' is not a DRIVE port")
            index = self._node_index[port.node]
            bindings = current_bindings[index]
            definition = next(
                item
                for item in self._parsed_models[self.graph.nodes[index].model].parameters
                if item.name == port.parameter
            )
            value = float(event.value)
            if not math.isfinite(value):
                raise ResolutionError(f"drive input '{port.id}' must be finite")
            if definition.domain is ParameterDomain.POSITIVE and value <= 0.0:
                raise ResolutionError(f"drive input '{port.id}' must be positive")
            bindings[port.parameter] = value
            mixed_drives.append(
                MixedDriveUpdate(float(event.t), index, value, port.parameter)
            )
        return mixed_inputs, mixed_drives

    def _prepare_modulations(
        self, modulation_inputs: Sequence[ModulationInput]
    ) -> tuple[ModulationEvent, ...]:
        events = []
        for item in modulation_inputs:
            if not isinstance(item, ModulationInput):
                raise ResolutionError("modulation inputs must contain ModulationInput values")
            modulator = self._modulator_ports.get(item.port)
            if modulator is None:
                raise ResolutionError(f"unknown modulator port '{item.port}'")
            t = float(item.t)
            value = float(item.value)
            if not math.isfinite(t) or t < 0.0 or not math.isfinite(value):
                raise ResolutionError("modulation time and value must be finite")
            events.append(ModulationEvent(t, modulator, value))
        return tuple(
            item for _, item in sorted(
                enumerate(events), key=lambda pair: (pair[1].t, pair[0])
            )
        )

    def _prepare_events(
        self,
        core: CoreEvaluator,
        spike_inputs: Sequence[SpikeInput],
        drive_inputs: Sequence[DriveInput],
        scalar_inputs: Sequence[ScalarInput] = (),
        *,
        encoder_seed: int = 0,
        encoder_spike_capacity: int = 4096,
    ) -> tuple[list[MixedInputSpike], list[MixedDriveUpdate]]:
        if not isinstance(encoder_seed, int) or not 0 <= encoder_seed <= 2**64 - 1:
            raise ResolutionError("encoder_seed must be an unsigned 64-bit integer")
        if encoder_spike_capacity < 0:
            raise ResolutionError("encoder_spike_capacity must be nonnegative")
        encoded_ports = self._encoded_ports()
        presentations = self._prepare_presentations(scalar_inputs, encoded_ports)
        encoded = core.encode_presentations(
            tuple(port.encoder for port in encoded_ports),
            presentations,
            seed=encoder_seed,
            spike_capacity=encoder_spike_capacity,
        )
        generated_spikes = [
            SpikeInput(item.t, encoded_ports[item.encoder].id, item.value)
            for item in sorted(
                encoded.spikes, key=lambda value: (value.t, value.encoder)
            )
        ]
        generated_drives = [
            DriveInput(item.t, encoded_ports[item.encoder].id, item.value)
            for item in sorted(
                encoded.drives, key=lambda value: (value.t, value.encoder)
            )
        ]
        self._validate_native_inputs(spike_inputs, drive_inputs)
        return self._map_input_primitives(
            (*spike_inputs, *generated_spikes),
            (*drive_inputs, *generated_drives),
        )

    def _project_outputs(
        self,
        result: RunResult | MixedRunResult,
        *,
        decode_schedule: tuple[DecoderWindowBinding, ...],
        allow_partial_decode: bool = False,
    ) -> GraphRunResult:
        ports_by_node: dict[int, list[str]] = {}
        for port in sorted(self.graph.output_ports, key=lambda item: item.id):
            ports_by_node.setdefault(self._node_index[port.node], []).append(port.id)
        outputs = tuple(
            PortSpike(spike.t, port, self.node_ids[spike.node])
            for spike in result.spikes
            for port in ports_by_node.get(spike.node, ())
        )
        trace = (
            tuple(self._project_trace_record(record) for record in result.trace)
            if isinstance(result, MixedRunResult)
            else ()
        )
        inspections = (
            tuple(
                self._project_inspection(item)
                for item in result.inspections
            )
            if isinstance(result, MixedRunResult)
            else ()
        )
        decoder_ports = tuple(
            port for port in self.graph.output_ports if port.decoder is not None
        )
        if not decoder_ports:
            return GraphRunResult(
                result, outputs, trace=trace, inspections=inspections
            )
        values = result.decoded if isinstance(result, MixedRunResult) else ()
        expected_values = len(decode_schedule)
        if len(values) != expected_values and not (
            allow_partial_decode and len(values) == 0
        ):
            raise ResolutionError(
                "decoder result count does not match the requested schedule"
            )
        decoded = tuple(
            DecodedPort(
                decoder_ports[value.decoder].id,
                decoder_ports[value.decoder].node,
                value.valid,
                value.count,
                value.value,
                value.first_spike,
                value.window,
                value.window_start,
                value.window_end,
            )
            for value in values
        )
        source_events = result.decoded_events if isinstance(result, MixedRunResult) else ()
        decoded_events = tuple(
            DecodedPortEvent(
                port=decoder_ports[event.decoder].id,
                node=decoder_ports[event.decoder].node,
                window=event.window,
                kind=event.kind,
                valid=event.valid,
                emitted_at=event.emitted_at,
                source_spike_time=event.source_spike_time,
                window_start=event.window_start,
                window_end=event.window_end,
                observed_through=event.observed_through,
                count=event.count,
                value=event.value,
                first_spike=event.first_spike,
            )
            for event in source_events
        )
        return GraphRunResult(
            result, outputs, decoded, decoded_events, trace, inspections
        )

    def _project_trace_record(self, record: TraceRecord) -> TraceRecord:
        model = self.models[record.node]
        names = (
            (model.state_name,)
            if isinstance(model, ResolvedScalarLIF)
            else model.state_names
        )
        return replace(
            record,
            node=self.node_ids[record.node],
            state_names=tuple(names[index] for index in record.state_indices),
        )

    def _project_inspection(self, item: StateInspection) -> StateInspection:
        model = self.models[item.node]
        names = (
            (model.state_name,)
            if isinstance(model, ResolvedScalarLIF)
            else model.state_names
        )
        return replace(
            item,
            node=self.node_ids[item.node],
            state_names=tuple(names[index] for index in item.state_indices),
        )

    def _inspection_requests(
        self, inspections: Sequence[StateInspectionRequest]
    ) -> tuple[StateInspectionRequest, ...]:
        normalized = tuple(inspections)
        projected: list[StateInspectionRequest] = []
        for request in normalized:
            if not isinstance(request, StateInspectionRequest):
                raise ResolutionError(
                    "inspections must contain StateInspectionRequest values"
                )
            try:
                node = self._node_index[request.node]
            except KeyError as exc:
                raise ResolutionError(
                    f"inspection references unknown graph node {request.node}"
                ) from exc
            projected.append(replace(request, node=node))
        return tuple(projected)

    def _recording_config(
        self, recording: RecordingConfig | None
    ) -> RecordingConfig | None:
        if recording is None:
            return None
        nodes = None
        if recording.nodes is not None:
            try:
                nodes = tuple(self._node_index[node] for node in recording.nodes)
            except KeyError as exc:
                raise ResolutionError(
                    f"recording references unknown graph node {exc.args[0]}"
                ) from exc
        consumer = recording.consumer
        if consumer is not None:
            source_consumer = consumer

            def projected_consumer(record: TraceRecord) -> None:
                """Map a low-level node index back to its authored identifier."""

                source_consumer(self._project_trace_record(record))

            consumer = projected_consumer
        return RecordingConfig(
            kinds=recording.kinds,
            nodes=nodes,
            capture_state=recording.capture_state,
            state_indices=recording.state_indices,
            capacity=recording.capacity,
            consumer=consumer,
        )

    def _decode_schedule(
        self,
        *,
        t_end: float,
        decode_start: float,
        decode_windows: (
            Sequence[DecodeWindow]
            | Mapping[str, Sequence[DecodeWindow]]
            | None
        ),
    ) -> tuple[DecoderWindowBinding, ...]:
        decoder_ports = tuple(
            port for port in self.graph.output_ports if port.decoder is not None
        )

        def validate_windows(
            windows: Sequence[DecodeWindow], context: str
        ) -> tuple[DecodeWindow, ...]:
            """Validate ordered half-open windows against the run horizon."""

            normalized = tuple(windows)
            previous_end = -math.inf
            for index, window in enumerate(normalized):
                if not isinstance(window, DecodeWindow) or (
                    not math.isfinite(float(window.t_start))
                    or not math.isfinite(float(window.t_end))
                    or not float(window.t_start) < float(window.t_end)
                    or float(window.t_end) > float(t_end)
                ):
                    raise ResolutionError(
                        f"{context} window {index} requires finite "
                        "start < end <= run end"
                    )
                if float(window.t_end) < previous_end:
                    raise ResolutionError(
                        f"{context} windows must be ordered by nondecreasing end time"
                    )
                previous_end = float(window.t_end)
            return normalized

        if isinstance(decode_windows, Mapping):
            if decode_start != 0.0:
                raise ResolutionError(
                    "decode_start cannot be combined with explicit decode_windows"
                )
            known = {port.id for port in decoder_ports}
            unknown = set(decode_windows) - known
            if unknown:
                raise ResolutionError(
                    f"decode_windows names unknown decoder ports: {sorted(unknown)}"
                )
            sparse: list[DecoderWindowBinding] = []
            for decoder, port in enumerate(decoder_ports):
                windows = validate_windows(
                    decode_windows.get(port.id, ()),
                    f"output port '{port.id}' decoder",
                )
                sparse.extend(
                    DecoderWindowBinding(
                        decoder, window, item.t_start, item.t_end
                    )
                    for window, item in enumerate(windows)
                )
            if not sparse:
                raise ResolutionError(
                    "decode_windows must schedule at least one decoder window"
                )
            return tuple(sparse)

        if decode_windows is None:
            common = (DecodeWindow(float(decode_start), float(t_end)),)
        else:
            if decode_start != 0.0:
                raise ResolutionError(
                    "decode_start cannot be combined with explicit decode_windows"
                )
            common = tuple(decode_windows)
        common = validate_windows(common, "decoder")
        if not common:
            raise ResolutionError("decode_windows must contain at least one DecodeWindow")
        return tuple(
            DecoderWindowBinding(decoder, window, item.t_start, item.t_end)
            for window, item in enumerate(common)
            for decoder in range(len(decoder_ports))
        )

    def _decoder_queries(
        self,
        schedule: Sequence[DecoderWindowBinding],
        decode_queries: Mapping[str, Sequence[DecodeQuery]] | None,
    ) -> tuple[DecoderQueryBinding, ...]:
        if decode_queries is None:
            return ()
        if not isinstance(decode_queries, Mapping):
            raise ResolutionError("decode_queries must map output ports to queries")
        decoder_ports = tuple(
            port for port in self.graph.output_ports if port.decoder is not None
        )
        known = {port.id for port in decoder_ports}
        unknown = set(decode_queries) - known
        if unknown:
            raise ResolutionError(
                f"decode_queries names unknown decoder ports: {sorted(unknown)}"
            )
        scheduled = {
            (binding.decoder, binding.window): binding for binding in schedule
        }
        result: list[DecoderQueryBinding] = []
        for decoder, port in enumerate(decoder_ports):
            queries = tuple(decode_queries.get(port.id, ()))
            if queries and port.decoder.emission is not EmissionPolicy.ON_QUERY:
                raise ResolutionError(
                    f"output port '{port.id}' decoder is not configured ON_QUERY"
                )
            for index, query in enumerate(queries):
                if (
                    not isinstance(query, DecodeQuery)
                    or not isinstance(query.window, int)
                    or not 0 <= query.window <= 2**32 - 1
                    or not math.isfinite(float(query.t))
                ):
                    raise ResolutionError(
                        f"output port '{port.id}' query {index} requires an "
                        "unsigned window id and finite time"
                    )
                binding = scheduled.get((decoder, query.window))
                if binding is None:
                    raise ResolutionError(
                        f"output port '{port.id}' query {index} references "
                        f"unscheduled window {query.window}"
                    )
                if not binding.t_start <= query.t <= binding.t_end:
                    raise ResolutionError(
                        f"output port '{port.id}' query {index} must lie inside "
                        "its assigned window (the end boundary is allowed)"
                    )
                if isinstance(port.decoder, RateDecoder):
                    effective_start = (
                        port.decoder.origin
                        if port.decoder.mode is RateMode.CUMULATIVE
                        else binding.t_start
                    )
                    if not query.t > effective_start:
                        raise ResolutionError(
                            f"output port '{port.id}' rate query {index} must "
                            "follow its effective window start"
                        )
                result.append(
                    DecoderQueryBinding(decoder, query.window, float(query.t))
                )
        return tuple(result)

    def run(
        self,
        core: CoreEvaluator,
        *,
        spike_inputs: Sequence[SpikeInput] = (),
        drive_inputs: Sequence[DriveInput] = (),
        scalar_inputs: Sequence[ScalarInput] = (),
        modulation_inputs: Sequence[ModulationInput] = (),
        t_end: float,
        decode_start: float = 0.0,
        decode_windows: (
            Sequence[DecodeWindow]
            | Mapping[str, Sequence[DecodeWindow]]
            | None
        ) = None,
        decode_queries: Mapping[str, Sequence[DecodeQuery]] | None = None,
        encoder_seed: int = 0,
        encoder_spike_capacity: int = 4096,
        decoder_event_capacity: int = 4096,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
        return_final_state: bool = True,
    ) -> GraphRunResult:
        """Compile and execute one graph run through named ports."""

        if self.precision is not PrecisionProfile.FLOAT64 or core.precision is not PrecisionProfile.FLOAT64:
            with self.compile(core) as compiled:
                return compiled.run(
                    spike_inputs=spike_inputs, drive_inputs=drive_inputs,
                    scalar_inputs=scalar_inputs, modulation_inputs=modulation_inputs,
                    t_end=t_end, decode_start=decode_start, decode_windows=decode_windows,
                    decode_queries=decode_queries, encoder_seed=encoder_seed,
                    encoder_spike_capacity=encoder_spike_capacity,
                    decoder_event_capacity=decoder_event_capacity,
                    queue_capacity=queue_capacity, output_capacity=output_capacity,
                    same_time_cascade_limit=same_time_cascade_limit,
                    stochastic_seed=stochastic_seed, recording=recording,
                    inspections=inspections, return_final_state=return_final_state,
                )
        mixed_inputs, mixed_drives = self._prepare_events(
            core,
            spike_inputs,
            drive_inputs,
            scalar_inputs,
            encoder_seed=encoder_seed,
            encoder_spike_capacity=encoder_spike_capacity,
        )

        decoder_bindings = self._decoder_bindings()
        core_recording = self._recording_config(recording)
        core_inspections = self._inspection_requests(inspections)
        modulations = self._prepare_modulations(modulation_inputs)
        modulator_by_edge = {
            edge: index
            for index, port in enumerate(self.graph.modulator_ports)
            for edge in port.edges
        }
        plasticity = tuple(edge.plasticity for edge in self.graph.edges)
        modulators = tuple(modulator_by_edge.get(edge.id) for edge in self.graph.edges)
        weight_groups = tuple(edge.weight_group for edge in self.graph.edges)
        if decoder_bindings:
            schedule = self._decode_schedule(
                t_end=t_end,
                decode_start=decode_start,
                decode_windows=decode_windows,
            )
            queries = self._decoder_queries(schedule, decode_queries)
            with core.compile_mixed(
                self.models,
                edges=self.edges,
                polarities=self.polarities,
                plasticity=plasticity,
                modulators=modulators,
                weight_groups=weight_groups,
            ) as compiled:
                with core.compile_decoders(
                    decoder_bindings, node_count=len(self.models)
                ) as decoder_bank:
                    with decoder_bank.create_run(
                        schedule=schedule,
                        queries=queries,
                        event_capacity=decoder_event_capacity,
                    ) as decoder_run:
                        result = compiled.run(
                            self.initial_values,
                            inputs=mixed_inputs,
                            drive_updates=mixed_drives,
                            modulations=modulations,
                            t_end=t_end,
                            queue_capacity=queue_capacity,
                            output_capacity=output_capacity,
                            same_time_cascade_limit=same_time_cascade_limit,
                            stochastic_seed=stochastic_seed,
                            decoder_run=decoder_run,
                            recording=core_recording,
                            inspections=core_inspections,
                            return_final_state=return_final_state,
                        )
        else:
            if decode_queries:
                raise ResolutionError("decode_queries requires at least one decoder")
            result: RunResult | MixedRunResult = core.run_mixed(
                self.models,
                self.initial_values,
                edges=self.edges,
                polarities=self.polarities,
                inputs=mixed_inputs,
                drive_updates=mixed_drives,
                modulations=modulations,
                plasticity=plasticity,
                modulators=modulators,
                weight_groups=weight_groups,
                t_end=t_end,
                queue_capacity=queue_capacity,
                output_capacity=output_capacity,
                same_time_cascade_limit=same_time_cascade_limit,
                stochastic_seed=stochastic_seed,
                recording=core_recording,
                inspections=core_inspections,
                return_final_state=return_final_state,
            )
        return self._project_outputs(
            result,
            decode_schedule=schedule if decoder_bindings else (),
        )


class CompiledResolvedGraph:
    """Prepared graph retaining the high-level named input/output port API."""

    def __init__(
        self,
        resolved: ResolvedGraph,
        compiled: CoreCompiledGraph | None = None,
        decoders: CompiledDecoderBank | None = None,
        *,
        core: CoreEvaluator | None = None,
        execution_plan: "ExecutionPlan | None" = None,
    ):
        self.resolved = resolved
        self.compiled = compiled
        self.decoders = decoders
        self._core = core if core is not None else compiled._core
        self.execution_plan = (
            resolved.execution_plan() if execution_plan is None else execution_plan
        )
        self._last_execution_path: str | None = None
        self._closed = False

    @property
    def preferred_execution_path(self) -> str:
        """Return the equation-derived sparse execution path."""

        return "compiled_sparse"

    @property
    def last_execution_path(self) -> str | None:
        """Return the path selected by the most recent run."""

        return self._last_execution_path

    def _ensure_compiled_scheduler(self) -> CoreCompiledGraph:
        if self._closed:
            raise RuntimeError("compiled graph is closed")
        if self.compiled is None:
            self.compiled, self.decoders = self.resolved._compile_execution_plan(
                self._core
            )
        return self.compiled

    @property
    def closed(self) -> bool:
        """Return whether compiled resources have been released."""

        return self._closed

    def close(self) -> None:
        """Release compiled graph and decoder resources once."""

        if self._closed:
            return
        if self.compiled is not None:
            self.compiled.close()
        if self.decoders is not None:
            self.decoders.close()
        self._closed = True

    def __enter__(self) -> "CompiledResolvedGraph":
        if self.closed:
            raise RuntimeError("compiled graph is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def run(
        self,
        *,
        spike_inputs: Sequence[SpikeInput] = (),
        drive_inputs: Sequence[DriveInput] = (),
        scalar_inputs: Sequence[ScalarInput] = (),
        modulation_inputs: Sequence[ModulationInput] = (),
        t_end: float,
        decode_start: float = 0.0,
        decode_windows: (
            Sequence[DecodeWindow]
            | Mapping[str, Sequence[DecodeWindow]]
            | None
        ) = None,
        decode_queries: Mapping[str, Sequence[DecodeQuery]] | None = None,
        encoder_seed: int = 0,
        encoder_spike_capacity: int = 4096,
        decoder_event_capacity: int = 4096,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
        return_final_state: bool = True,
    ) -> GraphRunResult:
        """Execute one run while reusing compiled graph resources."""

        if self.closed:
            raise RuntimeError("compiled graph is closed")
        mixed_inputs, mixed_drives = self.resolved._prepare_events(
            self._core,
            spike_inputs,
            drive_inputs,
            scalar_inputs,
            encoder_seed=encoder_seed,
            encoder_spike_capacity=encoder_spike_capacity,
        )
        core_recording = self.resolved._recording_config(recording)
        core_inspections = self.resolved._inspection_requests(inspections)
        modulations = self.resolved._prepare_modulations(modulation_inputs)
        compiled = self._ensure_compiled_scheduler()
        self._last_execution_path = "compiled_sparse"
        if self.decoders is None:
            if decode_queries:
                raise ResolutionError("decode_queries requires at least one decoder")
            result = compiled.run(
                self.resolved.initial_values,
                inputs=mixed_inputs,
                drive_updates=mixed_drives,
                modulations=modulations,
                t_end=t_end,
                queue_capacity=queue_capacity,
                output_capacity=output_capacity,
                same_time_cascade_limit=same_time_cascade_limit,
                stochastic_seed=stochastic_seed,
                recording=core_recording,
                inspections=core_inspections,
                return_final_state=return_final_state,
            )
        else:
            schedule = self.resolved._decode_schedule(
                t_end=t_end,
                decode_start=decode_start,
                decode_windows=decode_windows,
            )
            queries = self.resolved._decoder_queries(schedule, decode_queries)
            with self.decoders.create_run(
                schedule=schedule,
                queries=queries,
                event_capacity=decoder_event_capacity,
            ) as decoder_run:
                result = compiled.run(
                    self.resolved.initial_values,
                    inputs=mixed_inputs,
                    drive_updates=mixed_drives,
                    modulations=modulations,
                    t_end=t_end,
                    queue_capacity=queue_capacity,
                    output_capacity=output_capacity,
                    same_time_cascade_limit=same_time_cascade_limit,
                    stochastic_seed=stochastic_seed,
                    decoder_run=decoder_run,
                    recording=core_recording,
                    inspections=core_inspections,
                    return_final_state=return_final_state,
                )
        return self.resolved._project_outputs(
            result,
            decode_schedule=schedule if self.decoders is not None else (),
        )

    def create_incremental_run(
        self,
        *,
        t_end: float,
        decode_start: float = 0.0,
        decode_windows: (
            Sequence[DecodeWindow]
            | Mapping[str, Sequence[DecodeWindow]]
            | None
        ) = None,
        decode_queries: Mapping[str, Sequence[DecodeQuery]] | None = None,
        decoder_event_capacity: int = 4096,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        encoder_seed: int = 0,
        encoder_spike_capacity: int = 4096,
        encoder_drive_capacity: int = 4096,
        return_final_state: bool = True,
    ) -> "IncrementalResolvedGraphRun":
        """Open one incremental run on the compiled sparse scheduler."""

        if self.closed:
            raise RuntimeError("compiled graph is closed")
        arguments = dict(
            t_end=t_end,
            decode_start=decode_start,
            decode_windows=decode_windows,
            decode_queries=decode_queries,
            decoder_event_capacity=decoder_event_capacity,
            queue_capacity=queue_capacity,
            output_capacity=output_capacity,
            same_time_cascade_limit=same_time_cascade_limit,
            stochastic_seed=stochastic_seed,
            encoder_seed=encoder_seed,
            encoder_spike_capacity=encoder_spike_capacity,
            encoder_drive_capacity=encoder_drive_capacity,
            return_final_state=return_final_state,
        )
        return self._create_incremental_run(**arguments)

    def _create_incremental_run(
        self,
        *,
        t_end: float,
        decode_start: float = 0.0,
        decode_windows: (
            Sequence[DecodeWindow]
            | Mapping[str, Sequence[DecodeWindow]]
            | None
        ) = None,
        decode_queries: Mapping[str, Sequence[DecodeQuery]] | None = None,
        decoder_event_capacity: int = 4096,
        queue_capacity: int = 4096,
        output_capacity: int = 4096,
        same_time_cascade_limit: int = 1024,
        stochastic_seed: int = 0,
        encoder_seed: int = 0,
        encoder_spike_capacity: int = 4096,
        encoder_drive_capacity: int = 4096,
        return_final_state: bool = True,
    ) -> "IncrementalResolvedGraphRun":
        """Open a named-port incremental run for raw and scalar inputs."""

        compiled = self._ensure_compiled_scheduler()
        self._last_execution_path = "compiled_sparse"

        decoder_run = None
        schedule: tuple[DecoderWindowBinding, ...] = ()
        if self.decoders is not None:
            schedule = self.resolved._decode_schedule(
                t_end=t_end,
                decode_start=decode_start,
                decode_windows=decode_windows,
            )
            queries = self.resolved._decoder_queries(schedule, decode_queries)
            decoder_run = self.decoders.create_run(
                schedule=schedule,
                queries=queries,
                event_capacity=decoder_event_capacity,
            )
        elif decode_queries:
            raise ResolutionError("decode_queries requires at least one decoder")
        core_run = None
        encoder_run = None
        encoded_ports = self.resolved._encoded_ports()
        try:
            core_run = compiled.create_incremental_run(
                self.resolved.initial_values,
                t_end=t_end,
                queue_capacity=queue_capacity,
                output_capacity=output_capacity,
                same_time_cascade_limit=same_time_cascade_limit,
                stochastic_seed=stochastic_seed,
                decoder_run=decoder_run,
                return_final_state=return_final_state,
            )
            if encoded_ports:
                encoder_run = self._core.create_streaming_encoder_run(
                    tuple(port.encoder for port in encoded_ports),
                    initial_frontier=core_run.frontier,
                    seed=encoder_seed,
                    spike_capacity=encoder_spike_capacity,
                    drive_capacity=encoder_drive_capacity,
                )
        except Exception:
            if encoder_run is not None:
                encoder_run.close()
            if core_run is not None:
                core_run.close()
            if decoder_run is not None:
                decoder_run.close()
            raise
        return IncrementalResolvedGraphRun(
            self.resolved,
            core_run,
            decoder_run,
            schedule,
            encoded_ports,
            encoder_run,
        )


class IncrementalResolvedGraphRun:
    """Named-port wrapper over resumable raw-event and scalar C execution."""

    def __init__(
        self,
        resolved: ResolvedGraph,
        core_run: IncrementalCompiledRun,
        decoder_run,
        decode_schedule: tuple[DecoderWindowBinding, ...],
        encoded_ports: tuple[InputPort, ...],
        encoder_run: StreamingEncoderRun | None,
    ):
        self.resolved = resolved
        self.core_run = core_run
        self.decoder_run = decoder_run
        self.decode_schedule = decode_schedule
        self.encoded_ports = encoded_ports
        self.encoder_run = encoder_run
        self._failed = False

    @property
    def closed(self) -> bool:
        """Return whether the underlying incremental run is closed."""

        return self.core_run.closed

    @property
    def frontier(self) -> float:
        """Return the settled open simulation frontier."""

        return self.core_run.frontier

    @property
    def finished(self) -> bool:
        """Return whether the final horizon has been sealed."""

        return self.core_run.finished

    @property
    def cumulative_stats(self):
        """Return counters accumulated across all advances."""

        return self.core_run.cumulative_stats

    def close(self) -> None:
        """Release network, encoder, and decoder run resources."""

        self.core_run.close()
        if self.encoder_run is not None:
            self.encoder_run.close()
        if self.decoder_run is not None:
            self.decoder_run.close()

    def reset_episode(self) -> None:
        """Reset network/encoder state and traces while retaining learned weights."""

        if self.closed:
            raise RuntimeError("incremental graph run is closed")
        if self._failed:
            raise RuntimeError("incremental graph run is unusable after a failed advance")
        try:
            self.core_run.reset_episode(self.resolved.initial_values)
            if self.encoder_run is not None:
                self.encoder_run.reset_episode()
        except Exception:
            self._failed = True
            raise

    def __enter__(self) -> "IncrementalResolvedGraphRun":
        if self.closed:
            raise RuntimeError("incremental graph run is closed")
        if self._failed:
            raise RuntimeError("incremental graph run is unusable after a failed advance")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def advance_until(
        self,
        until: float,
        *,
        spike_inputs: Sequence[SpikeInput] = (),
        drive_inputs: Sequence[DriveInput] = (),
        scalar_inputs: Sequence[ScalarInput] = (),
        modulation_inputs: Sequence[ModulationInput] = (),
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
    ) -> GraphRunResult:
        """Advance to an open boundary and return outputs from this segment."""

        mixed_inputs, mixed_drives = self._prepare_incremental_inputs(
            until,
            spike_inputs,
            drive_inputs,
            scalar_inputs,
            seal=False,
        )
        try:
            result = self.core_run.advance_until(
                until,
                inputs=mixed_inputs,
                drive_updates=mixed_drives,
                modulations=self.resolved._prepare_modulations(modulation_inputs),
                recording=self.resolved._recording_config(recording),
                inspections=self.resolved._inspection_requests(inspections),
            )
        except Exception:
            self._failed = True
            raise
        return self.resolved._project_outputs(
            result,
            decode_schedule=self.decode_schedule,
            allow_partial_decode=True,
        )

    def finish(
        self,
        *,
        spike_inputs: Sequence[SpikeInput] = (),
        drive_inputs: Sequence[DriveInput] = (),
        scalar_inputs: Sequence[ScalarInput] = (),
        modulation_inputs: Sequence[ModulationInput] = (),
        recording: RecordingConfig | None = None,
        inspections: Sequence[StateInspectionRequest] = (),
    ) -> GraphRunResult:
        """Seal the final horizon and return remaining outputs."""

        mixed_inputs, mixed_drives = self._prepare_incremental_inputs(
            self.core_run.t_end,
            spike_inputs,
            drive_inputs,
            scalar_inputs,
            seal=True,
        )
        try:
            result = self.core_run.finish(
                inputs=mixed_inputs,
                drive_updates=mixed_drives,
                modulations=self.resolved._prepare_modulations(modulation_inputs),
                recording=self.resolved._recording_config(recording),
                inspections=self.resolved._inspection_requests(inspections),
            )
        except Exception:
            self._failed = True
            raise
        return self.resolved._project_outputs(
            result,
            decode_schedule=self.decode_schedule,
            allow_partial_decode=True,
        )

    def _prepare_incremental_inputs(
        self,
        until: float,
        spike_inputs: Sequence[SpikeInput],
        drive_inputs: Sequence[DriveInput],
        scalar_inputs: Sequence[ScalarInput],
        *,
        seal: bool,
    ) -> tuple[list[MixedInputSpike], list[MixedDriveUpdate]]:
        if self._failed:
            raise RuntimeError("incremental graph run is unusable after a failed advance")
        self.resolved._validate_native_inputs(spike_inputs, drive_inputs)
        self.resolved._map_input_primitives(spike_inputs, drive_inputs)
        presentations = self.resolved._prepare_presentations(
            scalar_inputs, self.encoded_ports
        )
        if presentations and self.encoder_run is None:
            raise ResolutionError("this graph has no scalar-encoded input ports")
        if self.encoder_run is None:
            generated_spikes: tuple[SpikeInput, ...] = ()
            generated_drives: tuple[DriveInput, ...] = ()
        else:
            encoded = (
                self.encoder_run.finish(until, presentations)
                if seal
                else self.encoder_run.advance_until(until, presentations)
            )
            generated_spikes = tuple(
                SpikeInput(
                    item.t,
                    self.encoded_ports[item.encoder].id,
                    item.value,
                )
                for item in sorted(
                    encoded.spikes, key=lambda value: (value.t, value.encoder)
                )
            )
            generated_drives = tuple(
                DriveInput(
                    item.t,
                    self.encoded_ports[item.encoder].id,
                    item.value,
                )
                for item in sorted(
                    encoded.drives, key=lambda value: (value.t, value.encoder)
                )
            )
        return self.resolved._map_input_primitives(
            (*spike_inputs, *generated_spikes),
            (*drive_inputs, *generated_drives),
        )


def _unique(values: Sequence[object], key, label: str) -> None:
    identifiers = [key(value) for value in values]
    if len(identifiers) != len(set(identifiers)):
        raise ResolutionError(f"{label} identifiers must be unique")


def _finite_codec_values(label: str, **values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ResolutionError(f"{label} {name} must be finite")


def _validate_encoder(port: InputPort) -> None:
    encoder = port.encoder
    label = f"input port '{port.id}' encoder"
    if not isinstance(
        encoder,
        (
            NativeEventEncoder,
            RegularRateEncoder,
            PoissonRateEncoder,
            TTFSEncoder,
            BurstEncoder,
            LatencyBurstEncoder,
            HeldCurrentEncoder,
        ),
    ):
        raise ResolutionError(f"{label} has unsupported type {type(encoder).__name__}")
    if isinstance(encoder, NativeEventEncoder):
        return
    if isinstance(encoder, HeldCurrentEncoder):
        if port.mode is not InputMode.DRIVE:
            raise ResolutionError(f"{label} HELD_CURRENT requires a DRIVE port")
        _finite_codec_values(
            label,
            gain=encoder.gain,
            offset=encoder.offset,
            baseline=encoder.baseline,
        )
        return
    if port.mode is not InputMode.SPIKE:
        raise ResolutionError(f"{label} {encoder.kind.name} requires a SPIKE port")
    _finite_codec_values(label, amplitude=encoder.amplitude)
    if isinstance(encoder, (RegularRateEncoder, PoissonRateEncoder, BurstEncoder)):
        _finite_codec_values(
            label, min_rate=encoder.min_rate, max_rate=encoder.max_rate
        )
        if encoder.min_rate < 0.0 or encoder.max_rate < encoder.min_rate:
            raise ResolutionError(
                f"{label} rates must satisfy 0 <= min_rate <= max_rate"
            )
        if isinstance(encoder, BurstEncoder):
            _finite_codec_values(label, duration=encoder.duration)
            if encoder.duration <= 0.0:
                raise ResolutionError(f"{label} duration must be positive")
        return
    if isinstance(encoder, (TTFSEncoder, LatencyBurstEncoder)):
        _finite_codec_values(
            label,
            min_latency=encoder.min_latency,
            max_latency=encoder.max_latency,
            silence_threshold=encoder.silence_threshold,
        )
        if encoder.min_latency < 0.0 or encoder.max_latency < encoder.min_latency:
            raise ResolutionError(
                f"{label} latencies must satisfy 0 <= min_latency <= max_latency"
            )
        if not 0.0 <= encoder.silence_threshold <= 1.0:
            raise ResolutionError(
                f"{label} silence_threshold must lie in normalized range [0, 1]"
            )
        if isinstance(encoder, LatencyBurstEncoder):
            _finite_codec_values(label, rate=encoder.rate, duration=encoder.duration)
            if encoder.rate <= 0.0 or encoder.duration <= 0.0:
                raise ResolutionError(f"{label} rate and duration must be positive")
        return
    raise ResolutionError(f"{label} has unsupported type {type(encoder).__name__}")


def _validate_decoder(port: OutputPort) -> None:
    decoder = port.decoder
    label = f"output port '{port.id}' decoder"
    if decoder is None:
        return
    if not isinstance(decoder.emission, EmissionPolicy):
        raise ResolutionError(f"{label} has an invalid emission policy")
    if isinstance(decoder, TTFSDecoder):
        if decoder.emission not in (
            EmissionPolicy.ON_EVENT,
            EmissionPolicy.ON_WINDOW_CLOSE,
            EmissionPolicy.ON_QUERY,
        ):
            raise ResolutionError(
                f"{label} supports ON_EVENT, ON_WINDOW_CLOSE, or ON_QUERY emission"
            )
        return
    if isinstance(decoder, RateDecoder):
        if decoder.emission not in (
            EmissionPolicy.ON_WINDOW_CLOSE,
            EmissionPolicy.ON_QUERY,
        ):
            raise ResolutionError(
                f"{label} supports ON_WINDOW_CLOSE or ON_QUERY emission"
            )
        _finite_codec_values(label, origin=decoder.origin)
        if not isinstance(decoder.mode, RateMode):
            raise ResolutionError(f"{label} has an invalid rate mode")
        if decoder.mode is RateMode.SLIDING:
            if decoder.width is None:
                raise ResolutionError(f"{label} sliding mode requires width")
            _finite_codec_values(label, width=decoder.width)
            if decoder.width <= 0.0:
                raise ResolutionError(f"{label} sliding width must be positive")
            if decoder.emission is EmissionPolicy.ON_QUERY:
                raise ResolutionError(
                    f"{label} sliding ON_QUERY requires timestamp retention and "
                    "is reserved for a future bounded-state implementation"
                )
        return
    if isinstance(decoder, TemporalWeightDecoder):
        _finite_codec_values(label, tau=decoder.tau)
        if decoder.tau <= 0.0:
            raise ResolutionError(f"{label} tau must be positive")
        if not isinstance(decoder.spikes, TemporalSpikeMode):
            raise ResolutionError(f"{label} has an invalid spike selection")
        return
    raise ResolutionError(f"{label} has unsupported type {type(decoder).__name__}")


def resolve_graph(graph: Graph) -> ResolvedGraph:
    """Validate, canonicalize, and lower an authored graph."""

    return _resolve_graph(graph)


def _resolve_graph(
    graph: Graph,
    *,
    parsed_models: Mapping[str, NeuronModel] | None = None,
    parsed_synapse_models: Mapping[str, SynapseModel] | None = None,
) -> ResolvedGraph:
    """Share resolution with detached, non-executable precision preflight."""

    if not graph.time_unit or any(character.isspace() for character in graph.time_unit):
        raise ResolutionError("time_unit must be a nonempty token")
    if not graph.models:
        raise ResolutionError("graph must contain at least one model")
    if not graph.nodes:
        raise ResolutionError("graph must contain at least one node")
    _unique(graph.models, lambda item: item.id, "model")
    _unique(graph.synapses, lambda item: item.id, "synapse")
    _unique(graph.nodes, lambda item: item.id, "node")
    _unique(graph.edges, lambda item: item.id, "edge")
    _unique(graph.input_ports, lambda item: item.id, "input port")
    _unique(graph.output_ports, lambda item: item.id, "output port")
    _unique(graph.modulator_ports, lambda item: item.id, "modulator port")
    ordinary_ports = {item.id for item in graph.input_ports} | {
        item.id for item in graph.output_ports
    }
    duplicate_ports = ordinary_ports & {item.id for item in graph.modulator_ports}
    if duplicate_ports:
        raise ResolutionError(
            "modulator port identifiers overlap another port: "
            + ", ".join(sorted(duplicate_ports))
        )

    # Parse each unique source once before resolving individual node bindings.
    parsed: dict[str, NeuronModel] = {}
    for item in graph.models:
        try:
            parsed[item.id] = (
                parse_neuron(item.source)
                if parsed_models is None else parsed_models[item.id]
            )
        except DSLParseError as exc:
            raise ResolutionError(f"model '{item.id}' failed to parse: {exc}") from exc

    parsed_synapses: dict[str, SynapseModel] = {}
    for item in graph.synapses:
        try:
            parsed_synapses[item.id] = (
                parse_synapse(item.source)
                if parsed_synapse_models is None else parsed_synapse_models[item.id]
            )
        except DSLParseError as exc:
            raise ResolutionError(f"synapse '{item.id}' failed to parse: {exc}") from exc

    ordered_nodes = tuple(sorted(graph.nodes, key=lambda item: item.id))
    if any(node.id < 0 for node in ordered_nodes):
        raise ResolutionError("node identifiers must be nonnegative")
    for node in ordered_nodes:
        if not isinstance(node.polarity, NeuronPolarity):
            raise ResolutionError(
                f"node {node.id} polarity must be "
                "EXCITATORY, INHIBITORY, or MIXED"
            )
    node_index = {node.id: index for index, node in enumerate(ordered_nodes)}
    ordered_edges = tuple(sorted(graph.edges, key=lambda item: item.id))
    edge_by_id = {edge.id: edge for edge in ordered_edges}
    modulator_by_edge: dict[int, int] = {}
    ordered_modulators = tuple(sorted(graph.modulator_ports, key=lambda item: item.id))
    for modulator_index, port in enumerate(ordered_modulators):
        if not isinstance(port.id, str) or not port.id:
            raise ResolutionError("modulator port id must be nonempty")
        if not port.edges:
            raise ResolutionError(f"modulator port '{port.id}' must target an edge")
        if len(port.edges) != len(set(port.edges)):
            raise ResolutionError(f"modulator port '{port.id}' repeats an edge")
        for edge_id in port.edges:
            edge = edge_by_id.get(edge_id)
            if edge is None:
                raise ResolutionError(
                    f"modulator port '{port.id}' references unknown edge {edge_id}"
                )
            if not isinstance(
                edge.plasticity,
                (
                    ModulatedSTDP,
                    VoltageModulatedSTDP,
                    SoftExcursionModulated,
                ),
            ):
                raise ResolutionError(
                    f"modulator port '{port.id}' targets edge {edge_id} without "
                    "modulated plasticity"
                )
            if edge_id in modulator_by_edge:
                raise ResolutionError(
                    f"modulated edge {edge_id} is targeted by more than one modulator"
                )
            modulator_by_edge[edge_id] = modulator_index
    for edge in ordered_edges:
        if (
            edge.id < 0
            or edge.pre not in node_index
            or edge.post not in node_index
        ):
            raise ResolutionError(f"edge {edge.id} references an invalid node")
        if (
            edge.plasticity is not None
            and ordered_nodes[node_index[edge.pre]].polarity
            is NeuronPolarity.MIXED
        ):
            raise ResolutionError(
                f"edge {edge.id} online plasticity is not supported for "
                "MIXED presynaptic neurons"
            )
        if edge.weight_group is not None and (
            not isinstance(edge.weight_group, int)
            or isinstance(edge.weight_group, bool)
            or edge.weight_group < 0
        ):
            raise ResolutionError(
                f"edge {edge.id} weight_group must be a nonnegative integer or None"
            )
        if edge.plasticity is not None and not isinstance(
            edge.plasticity,
            (
                PairSTDP,
                TripletSTDP,
                ModulatedSTDP,
                VoltageModulatedSTDP,
                SoftExcursionModulated,
            ),
        ):
            raise ResolutionError(
                f"edge {edge.id} has unsupported plasticity type "
                f"{type(edge.plasticity).__name__}"
            )
        if edge.plasticity is not None:
            lower, upper = edge.plasticity.bounds
            if not lower <= edge.weight <= upper:
                raise ResolutionError(
                    f"edge {edge.id} initial weight {edge.weight} lies outside "
                    f"plasticity bounds [{lower}, {upper}]"
                )
        if isinstance(
            edge.plasticity,
            (
                ModulatedSTDP,
                VoltageModulatedSTDP,
                SoftExcursionModulated,
            ),
        ) and edge.id not in modulator_by_edge:
            raise ResolutionError(
                f"modulated edge {edge.id} is not targeted by a modulator port"
            )
    shared_groups: dict[int, list[GraphEdge]] = {}
    for edge in ordered_edges:
        if edge.weight_group is not None:
            shared_groups.setdefault(edge.weight_group, []).append(edge)
    for group, members in shared_groups.items():
        reference = members[0]
        reference_polarity = ordered_nodes[node_index[reference.pre]].polarity
        reference_semantics = (
            reference.weight,
            reference.synapse,
            reference.receptor,
            reference.output,
            reference.synapse_bindings,
            reference.initial,
            reference.plasticity,
            reference_polarity,
        )
        for member in members[1:]:
            semantics = (
                member.weight,
                member.synapse,
                member.receptor,
                member.output,
                member.synapse_bindings,
                member.initial,
                member.plasticity,
                ordered_nodes[node_index[member.pre]].polarity,
            )
            if semantics != reference_semantics:
                raise ResolutionError(
                    f"weight group {group} must use one initial weight, polarity, "
                    "synapse definition, and plasticity rule"
                )
    incoming_stateful: dict[int, list[GraphEdge]] = {}
    edge_synapse_bindings: list[Mapping[str, float]] = []
    edge_bindings_by_id: dict[int, Mapping[str, float]] = {}
    for edge in ordered_edges:
        if edge.id < 0 or edge.pre not in node_index or edge.post not in node_index:
            raise ResolutionError(f"edge {edge.id} references an invalid node")
        edge_initial = (
            tuple(float(value) for value in edge.initial)
            if isinstance(edge.initial, (tuple, list))
            else (float(edge.initial),)
        )
        if (
            not math.isfinite(float(edge.weight))
            or not math.isfinite(float(edge.delay))
            or not edge_initial
            or any(not math.isfinite(value) for value in edge_initial)
        ):
            raise ResolutionError(f"edge {edge.id} values must be finite")
        if edge.delay < 0.0:
            raise ResolutionError(f"edge {edge.id} delay must be nonnegative")
        if (
            edge.weight < 0.0
            and ordered_nodes[node_index[edge.pre]].polarity
            is not NeuronPolarity.MIXED
        ):
            raise ResolutionError(
                f"edge {edge.id} weight must be a nonnegative magnitude; "
                "signed weights require a MIXED presynaptic neuron"
            )
        if edge.synapse is None:
            if (
                edge.receptor is not None
                or edge.output is not None
                or edge.synapse_bindings
                or len(edge_initial) != 1
                or edge_initial[0] != 0.0
            ):
                raise ResolutionError(
                    f"edge {edge.id} has synapse fields but no synapse"
                )
            edge_synapse_bindings.append({})
            edge_bindings_by_id[edge.id] = {}
            continue
        if edge.synapse not in parsed_synapses:
            raise ResolutionError(
                f"edge {edge.id} references unknown synapse '{edge.synapse}'"
            )
        if edge.receptor is None or edge.output is None:
            raise ResolutionError(
                f"stateful edge {edge.id} requires receptor and output mappings"
            )
        values = {
            definition.name: definition.default
            for definition in parsed_synapses[edge.synapse].parameters
        }
        supplied = dict(edge.synapse_bindings or {})
        unknown = set(supplied).difference(values)
        if unknown:
            raise ResolutionError(
                f"edge {edge.id} has unknown synapse binding(s): "
                + ", ".join(sorted(unknown))
            )
        values.update(supplied)
        edge_synapse_bindings.append(values)
        edge_bindings_by_id[edge.id] = values
        incoming_stateful.setdefault(edge.post, []).append(edge)

    resolved_models: list[
        ResolvedScalarLIF
        | ResolvedAlphaLIF
        | ResolvedAdaptiveLIF
        | ResolvedPerEdgeLIF
        | ResolvedReactiveIF
        | ResolvedSteppedNeuron
    ] = []
    bindings: list[Mapping[str, float]] = []
    synapse_bindings: list[Mapping[str, float]] = []
    initial_values: list[float | tuple[float, ...]] = []

    for node in ordered_nodes:
        if node.model not in parsed:
            raise ResolutionError(f"node {node.id} references unknown model '{node.model}'")
        stateful_edges = tuple(incoming_stateful.get(node.id, ()))
        if stateful_edges:
            if parsed[node.model].hazard is not None:
                raise CapabilityError(
                    f"node {node.id} intrinsic hazards currently support delta "
                    "synapses; filtered-current hazard lowering is a later capability"
                )
            if node.synapse is not None:
                raise CapabilityError(
                    f"node {node.id} cannot mix legacy node-scoped and edge-scoped synapses"
                )
            if node.receptor is not None or node.output is not None or node.synapse_bindings:
                raise ResolutionError(
                    f"node {node.id} has node-scoped synapse fields but no node synapse"
                )
            if isinstance(node.initial, (tuple, list)):
                raise ResolutionError(
                    f"edge-scoped analytical node {node.id} initial membrane value must be scalar; "
                    "edge initial states belong on their edges"
                )
            resolved = resolve_per_edge_lif(
                parsed[node.model],
                tuple(
                    PerEdgeSynapseInstance(
                        edge.id,
                        parsed_synapses[edge.synapse],
                        edge.receptor,
                        edge.output,
                        edge_bindings_by_id[edge.id],
                        tuple(float(value) for value in edge.initial)
                        if isinstance(edge.initial, (tuple, list))
                        else float(edge.initial),
                    )
                    for edge in stateful_edges
                    if edge.synapse is not None
                    and edge.receptor is not None
                    and edge.output is not None
                ),
                neuron_bindings=node.bindings,
            )
            initial = (float(node.initial), *resolved.group_initials)
            if any(not math.isfinite(value) for value in initial):
                raise ResolutionError(f"node {node.id} initial state must be finite")
            node_bindings = {
                definition.name: resolved.bindings[f"neuron.{definition.name}"]
                for definition in parsed[node.model].parameters
            }
            node_synapse_bindings = {}
        elif node.synapse is None:
            if node.receptor is not None or node.output is not None or node.synapse_bindings:
                raise ResolutionError(
                    f"node {node.id} has synapse mapping fields but no synapse"
                )
            authored = parsed[node.model]
            is_adaptive = any(
                state.role is StateRole.ADAPTATION for state in authored.states
            )
            if authored.reactive is not None:
                resolved = resolve_reactive_if(authored, node.bindings)
            elif authored.hazard is not None:
                resolved = (
                    resolve_adaptive_escape_lif(authored, node.bindings)
                    if is_adaptive
                    else resolve_escape_lif(authored, node.bindings)
                )
            else:
                try:
                    resolved = (
                        resolve_adaptive_lif(authored, node.bindings)
                        if is_adaptive
                        else resolve_scalar_lif(authored, node.bindings)
                    )
                except CapabilityError:
                    resolved = resolve_stepped_neuron(authored, node.bindings)
            if isinstance(resolved, ResolvedSteppedNeuron):
                state_count = len(resolved.state_names)
                if isinstance(node.initial, (tuple, list)):
                    initial = tuple(float(value) for value in node.initial)
                    if len(initial) != state_count:
                        raise ResolutionError(
                            f"stepped node {node.id} initial state must contain "
                            f"{state_count} values in declared state order"
                        )
                else:
                    initial = (float(node.initial),) + (0.0,) * (state_count - 1)
                if any(not math.isfinite(value) for value in initial):
                    raise ResolutionError(f"node {node.id} initial state must be finite")
            elif isinstance(resolved, ResolvedAdaptiveLIF):
                if isinstance(node.initial, (tuple, list)):
                    initial = tuple(float(value) for value in node.initial)
                    if len(initial) != 2:
                        raise ResolutionError(
                            f"adaptive node {node.id} initial state must contain [v, w]"
                        )
                else:
                    initial = (float(node.initial), 0.0)
                if any(not math.isfinite(value) for value in initial):
                    raise ResolutionError(f"node {node.id} initial state must be finite")
            else:
                if isinstance(node.initial, (tuple, list)):
                    raise ResolutionError(f"scalar node {node.id} initial value must be scalar")
                initial = float(node.initial)
                if not math.isfinite(initial):
                    raise ResolutionError(f"node {node.id} initial value must be finite")
            node_bindings = dict(resolved.bindings)
            node_synapse_bindings: Mapping[str, float] = {}
        else:
            if parsed[node.model].hazard is not None:
                raise CapabilityError(
                    f"node {node.id} intrinsic hazards currently support delta "
                    "synapses; folded-alpha hazard lowering is a later capability"
                )
            if node.synapse not in parsed_synapses:
                raise ResolutionError(
                    f"node {node.id} references unknown synapse '{node.synapse}'"
                )
            if node.receptor is None or node.output is None:
                raise ResolutionError(
                    f"folded-alpha node {node.id} requires receptor and output mappings"
                )
            resolved = resolve_folded_alpha_lif(
                parsed[node.model],
                parsed_synapses[node.synapse],
                receptor=node.receptor,
                output=node.output,
                neuron_bindings=node.bindings,
                synapse_bindings=node.synapse_bindings or {},
            )
            if isinstance(node.initial, (tuple, list)):
                values = tuple(float(value) for value in node.initial)
                if len(values) != 3:
                    raise ResolutionError(
                        f"folded-alpha node {node.id} initial state must contain [v, s, z]"
                    )
                initial = values
            else:
                initial = (float(node.initial), 0.0, 0.0)
            if any(not math.isfinite(value) for value in initial):
                raise ResolutionError(f"node {node.id} initial state must be finite")
            node_bindings = {
                definition.name: resolved.bindings[f"neuron.{definition.name}"]
                for definition in parsed[node.model].parameters
            }
            node_synapse_bindings = {
                definition.name: resolved.bindings[f"synapse.{definition.name}"]
                for definition in parsed_synapses[node.synapse].parameters
            }
        membrane_initial = (
            initial[resolved.readout_index] if isinstance(initial, tuple) else initial
        )
        if (
            not isinstance(resolved, ResolvedReactiveIF)
            and getattr(resolved, "hazard", None) is None
            and membrane_initial >= resolved.threshold
        ):
            raise ResolutionError(f"node {node.id} initial value must be below threshold")
        resolved_models.append(resolved)
        bindings.append(node_bindings)
        synapse_bindings.append(node_synapse_bindings)
        initial_values.append(initial)

    resolved_edges: list[MixedEdge] = []
    for edge in ordered_edges:
        post_model = resolved_models[node_index[edge.post]]
        if edge.synapse is not None:
            if not isinstance(post_model, ResolvedPerEdgeLIF):
                raise ResolutionError(
                    f"edge {edge.id} has stateful synapse data but its post node "
                    "did not resolve an edge-scoped analytical cluster"
                )
            deposit_kind = DepositKind.STATE_ADD
            target, deposit_scale = post_model.edge_deposits[edge.id]
        elif isinstance(post_model, ResolvedAlphaLIF):
            deposit_kind = DepositKind.PROGRAM
            target = post_model.deposit_index
        else:
            deposit_kind = DepositKind.STATE_ADD
            target = post_model.readout_index
        resolved_edges.append(
            MixedEdge(
                node_index[edge.pre],
                node_index[edge.post],
                float(edge.weight),
                float(edge.delay),
                deposit_kind,
                target,
                deposit_scale if edge.synapse is not None else 1.0,
            )
        )

    input_ports = {port.id: port for port in graph.input_ports}
    drive_bindings: set[tuple[int, str]] = set()
    for port in graph.input_ports:
        if port.node not in node_index:
            raise ResolutionError(f"input port '{port.id}' references an invalid node")
        _validate_encoder(port)
        node = ordered_nodes[node_index[port.node]]
        if port.mode is InputMode.SPIKE and port.parameter is not None:
            raise ResolutionError(f"SPIKE input port '{port.id}' cannot bind a parameter")
        if port.mode is InputMode.DRIVE:
            if port.parameter is None:
                raise ResolutionError(f"DRIVE input port '{port.id}' requires a parameter")
            drive_binding = (port.node, port.parameter)
            if drive_binding in drive_bindings:
                raise ResolutionError(
                    f"DRIVE input port '{port.id}' duplicates the node/parameter "
                    "binding of another input port"
                )
            drive_bindings.add(drive_binding)
            parameter_names = {item.name for item in parsed[node.model].parameters}
            if port.parameter not in parameter_names:
                raise ResolutionError(
                    f"DRIVE input port '{port.id}' references unknown parameter '{port.parameter}'"
                )
            definition = next(
                item
                for item in parsed[node.model].parameters
                if item.name == port.parameter
            )
            if (
                isinstance(port.encoder, HeldCurrentEncoder)
                and definition.domain is ParameterDomain.POSITIVE
                and min(
                    float(port.encoder.baseline),
                    float(port.encoder.offset),
                    float(port.encoder.offset + port.encoder.gain),
                )
                <= 0.0
            ):
                raise ResolutionError(
                    f"input port '{port.id}' HELD_CURRENT can produce a "
                    "nonpositive value for its positive parameter"
                )
            resolved = resolved_models[node_index[port.node]]
            drive_name = (
                f"neuron.{port.parameter}"
                if isinstance(resolved, (ResolvedAlphaLIF, ResolvedPerEdgeLIF))
                else port.parameter
            )
            if drive_name not in resolved.drive_parameters:
                detail = (
                    "is not an approved RHS-only runtime binding"
                    if isinstance(resolved, ResolvedSteppedNeuron)
                    else "changes more than the affine b term"
                )
                raise CapabilityError(
                    f"drive port '{port.id}' {detail}; "
                    "that requires a future structural boundary event"
                )
    for port in graph.output_ports:
        if port.node not in node_index:
            raise ResolutionError(f"output port '{port.id}' references an invalid node")
        _validate_decoder(port)

    canonical_graph = Graph(
        models=tuple(sorted(graph.models, key=lambda item: item.id)),
        nodes=ordered_nodes,
        edges=ordered_edges,
        input_ports=tuple(sorted(graph.input_ports, key=lambda item: item.id)),
        output_ports=tuple(sorted(graph.output_ports, key=lambda item: item.id)),
        time_unit=graph.time_unit,
        synapses=tuple(sorted(graph.synapses, key=lambda item: item.id)),
        modulator_ports=ordered_modulators,
    )
    return ResolvedGraph(
        graph=canonical_graph,
        node_ids=tuple(node.id for node in ordered_nodes),
        models=tuple(resolved_models),
        initial_values=tuple(initial_values),
        edges=tuple(resolved_edges),
        _parsed_models=parsed,
        _parsed_synapses=parsed_synapses,
        _node_bindings=tuple(bindings),
        _node_synapse_bindings=tuple(synapse_bindings),
        _edge_synapse_bindings=tuple(edge_synapse_bindings),
        _node_index=node_index,
        _input_ports=input_ports,
        _modulator_ports={port.id: index for index, port in enumerate(ordered_modulators)},
    )


def _hex(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ResolutionError("serialized numeric values must be finite")
    return number.hex()


def _unhex(value: object, context: str) -> float:
    if not isinstance(value, str):
        raise ResolutionError(f"{context} must be a hexadecimal float string")
    normalized = value.lower().lstrip("+-")
    if not normalized.startswith("0x") or "p" not in normalized:
        raise ResolutionError(f"{context} must use C99 hexadecimal-float syntax")
    try:
        number = float.fromhex(value)
    except (OverflowError, ValueError) as exc:
        raise ResolutionError(f"invalid hexadecimal float in {context}") from exc
    if not math.isfinite(number):
        raise ResolutionError(f"{context} must be finite")
    return number


def _plasticity_document(rule: PlasticityRule) -> dict[str, object]:
    document = plasticity_to_document(rule)
    encoded: dict[str, object] = {}
    for name, value in document["parameters"].items():  # type: ignore[union-attr]
        if isinstance(value, bool):
            encoded[name] = value
        elif name == "bounds":
            encoded[name] = [_hex(item) for item in value]  # type: ignore[arg-type]
        else:
            encoded[name] = _hex(value)  # type: ignore[arg-type]
    return {"kind": document["kind"], "parameters": encoded}


def _plasticity_from_document(value: object, context: str) -> PlasticityRule:
    if not isinstance(value, Mapping) or not isinstance(value.get("parameters"), Mapping):
        raise ResolutionError(f"{context} plasticity must be a kind/parameters object")
    decoded: dict[str, object] = {}
    for name, item in value["parameters"].items():
        if isinstance(item, bool):
            decoded[name] = item
        elif name == "bounds":
            if not isinstance(item, list):
                raise ResolutionError(f"{context} plasticity bounds must be an array")
            decoded[name] = tuple(
                _unhex(component, f"{context} plasticity bound") for component in item
            )
        else:
            decoded[name] = _unhex(item, f"{context} plasticity {name}")
    return plasticity_from_document({"kind": value.get("kind"), "parameters": decoded})


def _source_hash(source: str) -> str:
    if not isinstance(source, str):
        raise ResolutionError("model and synapse sources must be strings")
    normalized = source.strip() + "\n"
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeError as exc:
        raise ResolutionError("model and synapse sources must be valid UTF-8") from exc
    return hashlib.sha256(encoded).hexdigest()


def _codec_hash(document: Mapping[str, object]) -> str:
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _encoder_document(encoder: Encoder) -> dict[str, object]:
    if isinstance(encoder, NativeEventEncoder):
        return {"kind": "NATIVE_EVENT"}
    if isinstance(encoder, RegularRateEncoder):
        return {
            "kind": "REGULAR_RATE",
            "min_rate": _hex(encoder.min_rate),
            "max_rate": _hex(encoder.max_rate),
            "amplitude": _hex(encoder.amplitude),
        }
    if isinstance(encoder, PoissonRateEncoder):
        return {
            "kind": "POISSON_RATE",
            "min_rate": _hex(encoder.min_rate),
            "max_rate": _hex(encoder.max_rate),
            "amplitude": _hex(encoder.amplitude),
        }
    if isinstance(encoder, TTFSEncoder):
        return {
            "kind": "TTFS",
            "min_latency": _hex(encoder.min_latency),
            "max_latency": _hex(encoder.max_latency),
            "amplitude": _hex(encoder.amplitude),
            "silence_threshold": _hex(encoder.silence_threshold),
        }
    if isinstance(encoder, BurstEncoder):
        return {
            "kind": "BURST",
            "min_rate": _hex(encoder.min_rate),
            "max_rate": _hex(encoder.max_rate),
            "duration": _hex(encoder.duration),
            "amplitude": _hex(encoder.amplitude),
        }
    if isinstance(encoder, LatencyBurstEncoder):
        return {
            "kind": "LATENCY_BURST",
            "min_latency": _hex(encoder.min_latency),
            "max_latency": _hex(encoder.max_latency),
            "rate": _hex(encoder.rate),
            "duration": _hex(encoder.duration),
            "amplitude": _hex(encoder.amplitude),
            "silence_threshold": _hex(encoder.silence_threshold),
        }
    if isinstance(encoder, HeldCurrentEncoder):
        return {
            "kind": "HELD_CURRENT",
            "gain": _hex(encoder.gain),
            "offset": _hex(encoder.offset),
            "baseline": _hex(encoder.baseline),
        }
    raise ResolutionError(f"unsupported encoder type {type(encoder).__name__}")


def _decoder_document(decoder: Decoder) -> dict[str, object]:
    if isinstance(decoder, RateDecoder):
        return {
            "kind": "RATE",
            "mode": decoder.mode.name,
            "width": None if decoder.width is None else _hex(decoder.width),
            "origin": _hex(decoder.origin),
            "emission": decoder.emission.name,
        }
    if isinstance(decoder, TTFSDecoder):
        return {
            "kind": "TTFS",
            "normalize": decoder.normalize,
            "emission": decoder.emission.name,
        }
    if isinstance(decoder, TemporalWeightDecoder):
        return {
            "kind": "TEMPORAL_WEIGHT",
            "tau": _hex(decoder.tau),
            "spikes": decoder.spikes.value,
            "normalize": decoder.normalize,
            "emission": decoder.emission.name,
        }
    raise ResolutionError(f"unsupported decoder type {type(decoder).__name__}")


def _encoder_from_document(document: Mapping[str, object], context: str) -> Encoder:
    kind = document["kind"]
    if kind == "NATIVE_EVENT":
        return NativeEventEncoder()
    if kind in ("REGULAR_RATE", "POISSON_RATE"):
        encoder_type = RegularRateEncoder if kind == "REGULAR_RATE" else PoissonRateEncoder
        return encoder_type(
            _unhex(document["min_rate"], f"{context} min_rate"),
            _unhex(document["max_rate"], f"{context} max_rate"),
            _unhex(document["amplitude"], f"{context} amplitude"),
        )
    if kind == "TTFS":
        return TTFSEncoder(
            _unhex(document["min_latency"], f"{context} min_latency"),
            _unhex(document["max_latency"], f"{context} max_latency"),
            _unhex(document["amplitude"], f"{context} amplitude"),
            _unhex(document["silence_threshold"], f"{context} silence_threshold"),
        )
    if kind == "BURST":
        return BurstEncoder(
            _unhex(document["min_rate"], f"{context} min_rate"),
            _unhex(document["max_rate"], f"{context} max_rate"),
            _unhex(document["duration"], f"{context} duration"),
            _unhex(document["amplitude"], f"{context} amplitude"),
        )
    if kind == "LATENCY_BURST":
        return LatencyBurstEncoder(
            _unhex(document["min_latency"], f"{context} min_latency"),
            _unhex(document["max_latency"], f"{context} max_latency"),
            _unhex(document["rate"], f"{context} rate"),
            _unhex(document["duration"], f"{context} duration"),
            _unhex(document["amplitude"], f"{context} amplitude"),
            _unhex(document["silence_threshold"], f"{context} silence_threshold"),
        )
    if kind == "HELD_CURRENT":
        return HeldCurrentEncoder(
            _unhex(document["gain"], f"{context} gain"),
            _unhex(document["offset"], f"{context} offset"),
            _unhex(document["baseline"], f"{context} baseline"),
        )
    raise ResolutionError(f"{context} has unknown kind '{kind}'")


def _decoder_from_document(document: Mapping[str, object], context: str) -> Decoder:
    kind = document["kind"]
    if kind == "RATE":
        width = document.get("width")
        return RateDecoder(
            RateMode[str(document["mode"])],
            None if width is None else _unhex(width, f"{context} width"),
            _unhex(document["origin"], f"{context} origin"),
            EmissionPolicy[str(document["emission"])],
        )
    if kind == "TTFS":
        return TTFSDecoder(
            bool(document["normalize"]),
            EmissionPolicy[str(document["emission"])],
        )
    if kind == "TEMPORAL_WEIGHT":
        return TemporalWeightDecoder(
            _unhex(document["tau"], f"{context} tau"),
            TemporalSpikeMode(str(document["spikes"])),
            bool(document["normalize"]),
            EmissionPolicy[str(document["emission"])],
        )
    raise ResolutionError(f"{context} has unknown kind '{kind}'")


def graph_to_text(graph: Graph) -> str:
    """Serialize a validated graph with hashes for embedded model sources."""

    resolved = graph.resolve()
    document = {
        "schema": (
            MIXED_GRAPH_SCHEMA_VERSION
            if any(
                node.polarity is NeuronPolarity.MIXED for node in graph.nodes
            )
            else GRAPH_SCHEMA_VERSION
        ),
        "time_unit": resolved.graph.time_unit,
        "models": [
            {
                "id": model.id,
                "model_hash": _source_hash(model.source),
                "source": model.source.strip() + "\n",
            }
            for model in resolved.graph.models
        ],
        "synapses": [
            {
                "id": synapse.id,
                "synapse_hash": _source_hash(synapse.source),
                "source": synapse.source.strip() + "\n",
            }
            for synapse in resolved.graph.synapses
        ],
        "nodes": [
            {
                "id": node.id,
                "model": node.model,
                "polarity": node.polarity.value,
                "bindings": {
                    name: _hex(value)
                    for name, value in sorted(resolved._node_bindings[index].items())
                },
                "initial": (
                    _hex(resolved.initial_values[index][0])
                    if isinstance(resolved.models[index], ResolvedPerEdgeLIF)
                    else [_hex(value) for value in resolved.initial_values[index]]
                    if isinstance(resolved.initial_values[index], tuple)
                    else _hex(resolved.initial_values[index])
                ),
                "synapse": node.synapse,
                "receptor": node.receptor,
                "output": node.output,
                "synapse_bindings": {
                    name: _hex(value)
                    for name, value in sorted(
                        resolved._node_synapse_bindings[index].items()
                    )
                },
            }
            for index, node in enumerate(resolved.graph.nodes)
        ],
        "edges": [
            {
                "id": edge.id,
                "pre": edge.pre,
                "post": edge.post,
                "weight": _hex(edge.weight),
                "delay": _hex(edge.delay),
                "synapse": edge.synapse,
                "receptor": edge.receptor,
                "output": edge.output,
                "synapse_bindings": {
                    name: _hex(value)
                    for name, value in sorted(
                        resolved._edge_synapse_bindings[index].items()
                    )
                },
                "initial": (
                    [_hex(value) for value in edge.initial]
                    if isinstance(edge.initial, (tuple, list))
                    else _hex(edge.initial)
                ),
                "plasticity": (
                    None
                    if edge.plasticity is None
                    else _plasticity_document(edge.plasticity)
                ),
                **(
                    {"weight_group": edge.weight_group}
                    if edge.weight_group is not None
                    else {}
                ),
            }
            for index, edge in enumerate(resolved.graph.edges)
        ],
        "input_ports": [
            {
                "id": port.id,
                "node": port.node,
                "mode": port.mode.value,
                "parameter": port.parameter,
                "encoder": _encoder_document(port.encoder),
                "encoder_hash": _codec_hash(_encoder_document(port.encoder)),
            }
            for port in resolved.graph.input_ports
        ],
        "output_ports": [
            {
                "id": port.id,
                "node": port.node,
                "decoder": (
                    None if port.decoder is None else _decoder_document(port.decoder)
                ),
                "decoder_hash": (
                    None
                    if port.decoder is None
                    else _codec_hash(_decoder_document(port.decoder))
                ),
            }
            for port in resolved.graph.output_ports
        ],
        "modulator_ports": [
            {"id": port.id, "edges": list(port.edges)}
            for port in resolved.graph.modulator_ports
        ],
    }
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def graph_from_text(text: str) -> Graph:
    """Load a graph after validating its schema and embedded source hashes."""

    if not isinstance(text, str):
        raise ResolutionError("graph document must be text")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResolutionError(f"invalid graph document: {exc.msg}") from exc
    except RecursionError as exc:
        raise ResolutionError("invalid graph document: nesting is too deep") from exc
    if (
        not isinstance(document, dict)
        or not isinstance(document.get("schema"), int)
        or isinstance(document.get("schema"), bool)
        or document.get("schema") not in SUPPORTED_GRAPH_SCHEMA_VERSIONS
    ):
        raise ResolutionError(
            "unsupported graph schema; expected one of "
            + ", ".join(str(value) for value in sorted(SUPPORTED_GRAPH_SCHEMA_VERSIONS))
        )
    try:
        raw_models = document["models"]
        raw_synapses = document.get("synapses", [])
        for item in raw_models:
            if item.get("model_hash") != _source_hash(item["source"]):
                raise ResolutionError(f"model hash mismatch for '{item['id']}'")
        for item in raw_synapses:
            if item.get("synapse_hash") != _source_hash(item["source"]):
                raise ResolutionError(f"synapse hash mismatch for '{item['id']}'")
        for item in document.get("input_ports", []):
            if item.get("encoder_hash") != _codec_hash(item["encoder"]):
                raise ResolutionError(f"encoder hash mismatch for '{item['id']}'")
        for item in document.get("output_ports", []):
            decoder = item.get("decoder")
            expected = None if decoder is None else _codec_hash(decoder)
            if item.get("decoder_hash") != expected:
                raise ResolutionError(f"decoder hash mismatch for '{item['id']}'")

        nodes: list[GraphNode] = []
        for item in document["nodes"]:
            if (
                item.get("polarity") == NeuronPolarity.MIXED.value
                and document["schema"] < MIXED_GRAPH_SCHEMA_VERSION
            ):
                raise ResolutionError("MIXED polarity requires graph schema 11")
            raw_initial = item["initial"]
            initial = (
                tuple(
                    _unhex(value, f"node {item['id']} initial component")
                    for value in raw_initial
                )
                if isinstance(raw_initial, list)
                else _unhex(raw_initial, f"node {item['id']} initial")
            )
            nodes.append(
                GraphNode(
                    id=item["id"],
                    model=item["model"],
                    bindings={
                        name: _unhex(value, f"node {item['id']} binding {name}")
                        for name, value in item["bindings"].items()
                    },
                    initial=initial,
                    polarity=NeuronPolarity(item["polarity"]),
                    synapse=item.get("synapse"),
                    receptor=item.get("receptor"),
                    output=item.get("output"),
                    synapse_bindings={
                        name: _unhex(
                            value, f"node {item['id']} synapse binding {name}"
                        )
                        for name, value in item.get("synapse_bindings", {}).items()
                    },
                )
            )
        graph = Graph(
            models=tuple(GraphModel(item["id"], item["source"]) for item in raw_models),
            nodes=tuple(nodes),
            edges=tuple(
                GraphEdge(
                    id=item["id"],
                    pre=item["pre"],
                    post=item["post"],
                    weight=_unhex(item["weight"], f"edge {item['id']} weight"),
                    delay=_unhex(item["delay"], f"edge {item['id']} delay"),
                    synapse=item.get("synapse"),
                    receptor=item.get("receptor"),
                    output=item.get("output"),
                    synapse_bindings={
                        name: _unhex(
                            value, f"edge {item['id']} synapse binding {name}"
                        )
                        for name, value in item.get("synapse_bindings", {}).items()
                    },
                    initial=(
                        tuple(
                            _unhex(
                                value,
                                f"edge {item['id']} initial component",
                            )
                            for value in item.get("initial", [])
                        )
                        if isinstance(item.get("initial", "0x0.0p+0"), list)
                        else _unhex(
                            item.get("initial", "0x0.0p+0"),
                            f"edge {item['id']} initial",
                        )
                    ),
                    plasticity=(
                        None
                        if item.get("plasticity") is None
                        else _plasticity_from_document(
                            item["plasticity"], f"edge {item['id']}"
                        )
                    ),
                    weight_group=item.get("weight_group"),
                )
                for item in document.get("edges", [])
            ),
            input_ports=tuple(
                InputPort(
                    id=item["id"],
                    node=item["node"],
                    mode=InputMode(item["mode"]),
                    parameter=item.get("parameter"),
                    encoder=_encoder_from_document(
                        item["encoder"], f"input port '{item['id']}' encoder"
                    ),
                )
                for item in document.get("input_ports", [])
            ),
            output_ports=tuple(
                OutputPort(
                    item["id"],
                    item["node"],
                    None
                    if item.get("decoder") is None
                    else _decoder_from_document(
                        item["decoder"], f"output port '{item['id']}' decoder"
                    ),
                )
                for item in document.get("output_ports", [])
            ),
            time_unit=document["time_unit"],
            synapses=tuple(
                GraphSynapse(item["id"], item["source"]) for item in raw_synapses
            ),
            modulator_ports=tuple(
                ModulatorPort(item["id"], tuple(item["edges"]))
                for item in document.get("modulator_ports", [])
            ),
        )
    except ResolutionError:
        raise
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise ResolutionError(f"malformed graph document: {exc}") from exc
    try:
        return graph.resolve().graph
    except ResolutionError:
        raise
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise ResolutionError(f"malformed graph document: {exc}") from exc
