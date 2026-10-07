"""High-level execution façade and structured run data."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from .codec import DecodeQuery, DecodeWindow
from .errors import ResolutionError
from .ffi import (
    AugmentedState,
    CoreEvaluator,
    PlasticityState,
    RecordingConfig,
    ScalarState,
    StateInspectionRequest,
    TraceKind,
)
from .graph import (
    CompiledResolvedGraph,
    DriveInput,
    GraphRunResult,
    ModulationInput,
    ResolvedGraph,
    ScalarInput,
    SpikeInput,
)
from .ir import ResolvedScalarLIF
from .network import Network
from .precision import PrecisionProfile
from .recording import RecordingPlan, RunOptions, StateRecording, TraceRecording
from .tracefile import TraceArtifactMetadata, TraceArtifactWriter


def _network_precision(network: Network) -> PrecisionProfile | None:
    """Read reserved provenance without accepting contradictory profile fields."""

    if "lacuna_precision" not in network.metadata:
        return None
    try:
        return PrecisionProfile.from_record(network.metadata["lacuna_precision"])
    except (TypeError, ValueError) as exc:
        raise ResolutionError(f"invalid lacuna_precision metadata: {exc}") from exc


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{context} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise ResolutionError(f"{context} must be finite")
    return result


def _is_sequence(value: object) -> bool:
    return (
        not isinstance(value, (str, bytes, bytearray, Mapping))
        and hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
    )


@dataclass(frozen=True)
class SpikeTrain:
    """Timestamped spike values submitted to one input port."""

    times: tuple[float, ...] | Sequence[float]
    values: float | tuple[float, ...] | Sequence[float] = 1.0


@dataclass(frozen=True)
class DriveSeries:
    """Timestamped drive replacements submitted to one input port."""

    times: tuple[float, ...] | Sequence[float]
    values: tuple[float, ...] | Sequence[float]


@dataclass(frozen=True)
class ModulationSeries:
    """Timestamped third-factor values sent to one named modulator port."""

    times: tuple[float, ...] | Sequence[float]
    values: float | tuple[float, ...] | Sequence[float] = 1.0


@dataclass(frozen=True)
class ScalarPresentation:
    """Scalar input value presented over a half-open interval."""

    t_start: float
    t_end: float
    value: float


@dataclass(frozen=True)
class NetworkSpike:
    """Output spike expressed with an authored network node identifier."""

    t: float
    node: int


@dataclass(frozen=True)
class StateSample:
    """Selected state values observed at one settled simulation time."""

    t: float
    node: int
    names: tuple[str, ...]
    values: tuple[float, ...]
    generation: int
    clamped: bool


@dataclass(frozen=True)
class FinalState:
    """Complete final state for one authored network node."""

    node: int
    names: tuple[str, ...]
    values: tuple[float, ...]
    t: float


@dataclass(frozen=True)
class SpikeSeries:
    """Immutable output spikes with selection and conversion helpers."""

    events: tuple[NetworkSpike, ...]
    _network: Network = field(init=False, repr=False, compare=False)

    def __init__(self, events: Sequence[NetworkSpike], network: Network):
        object.__setattr__(self, "events", tuple(events))
        object.__setattr__(self, "_network", network)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    @property
    def times(self) -> tuple[float, ...]:
        """Return event times in output order."""

        return tuple(item.t for item in self.events)

    @property
    def nodes(self) -> tuple[int, ...]:
        """Return authored node identifiers in output order."""

        return tuple(item.node for item in self.events)

    def select(self, target: object) -> "SpikeSeries":
        """Return spikes emitted by the selected network nodes."""

        selected = frozenset(self._network.node_ids(target))
        return SpikeSeries(
            tuple(item for item in self.events if item.node in selected),
            self._network,
        )

    def to_numpy(self):
        """Return a structured NumPy array with time and node fields."""

        try:
            import numpy
        except ImportError as exc:
            raise RuntimeError("NumPy is required for to_numpy()") from exc
        result = numpy.empty(
            len(self.events), dtype=[("time", "f8"), ("node", "i8")]
        )
        result["time"] = self.times
        result["node"] = self.nodes
        return result

    def to_pandas(self):
        """Return spikes as a pandas data frame."""

        try:
            import pandas
        except ImportError as exc:
            raise RuntimeError("pandas is required for to_pandas()") from exc
        return pandas.DataFrame({"time": self.times, "node": self.nodes})


@dataclass(frozen=True)
class StateSeries:
    """Immutable explicit-time state samples with selection helpers."""

    samples: tuple[StateSample, ...]
    _network: Network = field(init=False, repr=False, compare=False)

    def __init__(self, samples: Sequence[StateSample], network: Network):
        object.__setattr__(self, "samples", tuple(samples))
        object.__setattr__(self, "_network", network)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def select(self, target: object, variable: str | None = None) -> "StateSeries":
        """Select network nodes and optionally one named state variable."""

        selected = frozenset(self._network.node_ids(target))
        result = []
        for sample in self.samples:
            if sample.node not in selected:
                continue
            if variable is None:
                result.append(sample)
                continue
            if variable not in sample.names:
                continue
            index = sample.names.index(variable)
            result.append(
                replace(
                    sample,
                    names=(variable,),
                    values=(sample.values[index],),
                )
            )
        return StateSeries(tuple(result), self._network)

    def to_pandas(self):
        """Return one row per state sample as a pandas data frame."""

        try:
            import pandas
        except ImportError as exc:
            raise RuntimeError("pandas is required for to_pandas()") from exc
        rows = []
        for sample in self.samples:
            row = {
                "time": sample.t,
                "node": sample.node,
                "generation": sample.generation,
                "clamped": sample.clamped,
            }
            row.update(dict(zip(sample.names, sample.values)))
            rows.append(row)
        return pandas.DataFrame(rows)


@dataclass(frozen=True)
class SimulationResult:
    """High-level spikes, state, decoding, learning, and runtime results."""

    network: Network
    precision: PrecisionProfile
    spikes: SpikeSeries
    states: StateSeries
    final_states: tuple[FinalState, ...]
    port_spikes: tuple[object, ...]
    decoded: tuple[object, ...]
    decoded_events: tuple[object, ...]
    trace: tuple[object, ...]
    trace_path: Path | None
    stats: object
    weights: tuple[float, ...]
    plasticity: tuple[PlasticityState, ...]
    raw: GraphRunResult = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        network: Network,
        raw: GraphRunResult,
        *,
        spike_targets: object | None = None,
        trace_path: Path | None = None,
        resolved: ResolvedGraph | None = None,
    ) -> None:
        provenance = _network_precision(network)
        if resolved is None:
            if provenance not in (None, PrecisionProfile.FLOAT64):
                raise ResolutionError(
                    "reduced-precision results require a target-resolved graph"
                )
            resolved = network.graph.resolve()
        if provenance is not None and provenance is not resolved.precision:
            raise ResolutionError("result precision does not match network provenance")
        events = tuple(
            NetworkSpike(item.t, resolved.node_ids[item.node])
            for item in raw.core.spikes
        )
        spikes = SpikeSeries(events, network)
        if spike_targets is not None:
            spikes = spikes.select(spike_targets)
        samples = StateSeries(
            tuple(
                StateSample(
                    item.t,
                    item.node,
                    item.state_names,
                    item.values,
                    item.generation,
                    item.clamped,
                )
                for item in raw.inspections
            ),
            network,
        )
        final = []
        for node, model, state in zip(
            resolved.node_ids, resolved.models, raw.core.states
        ):
            names = (
                (model.state_name,)
                if isinstance(model, ResolvedScalarLIF)
                else tuple(model.state_names)
            )
            if isinstance(state, ScalarState):
                values = (state.value,)
            elif isinstance(state, AugmentedState):
                values = state.values
            else:
                values = tuple(state.values)
            final.append(FinalState(node, names, tuple(values), state.t_last))
        object.__setattr__(self, "network", network)
        object.__setattr__(self, "precision", resolved.precision)
        object.__setattr__(self, "spikes", spikes)
        object.__setattr__(self, "states", samples)
        object.__setattr__(self, "final_states", tuple(final))
        object.__setattr__(self, "port_spikes", raw.outputs)
        object.__setattr__(self, "decoded", raw.decoded)
        object.__setattr__(self, "decoded_events", raw.decoded_events)
        object.__setattr__(self, "trace", raw.trace)
        object.__setattr__(self, "trace_path", trace_path)
        object.__setattr__(self, "stats", raw.core.stats)
        object.__setattr__(self, "weights", tuple(raw.core.weights))
        object.__setattr__(self, "plasticity", tuple(raw.core.plasticity))
        object.__setattr__(self, "raw", raw)

    def learned_network(self) -> Network:
        """Freeze the run's final weights into a new immutable network."""

        if len(self.weights) != len(self.network.graph.edges):
            raise RuntimeError("run did not return one weight for every graph edge")
        weights_by_id = {
            edge.id: weight
            for edge, weight in zip(
                sorted(self.network.graph.edges, key=lambda edge: edge.id),
                self.weights,
            )
        }
        edges = tuple(
            replace(edge, weight=weights_by_id[edge.id])
            for edge in self.network.graph.edges
        )
        return replace(
            self.network,
            graph=replace(self.network.graph, edges=edges),
            metadata={
                **self.network.metadata,
                "lacuna_precision": self.precision.to_record(),
            },
            _owner=object(),
        )

    def save_learned_network(self, path: str | os.PathLike[str]) -> Network:
        """Persist final learned weights and return the frozen snapshot."""

        network = self.learned_network()
        network.save(path)
        return network


@dataclass(frozen=True)
class _PreparedRecording:
    config: RecordingConfig | None
    inspections: tuple[StateInspectionRequest, ...]
    spike_targets: object | None
    trace: TraceRecording | None


class Engine:
    """Own the C evaluator and compile immutable networks for repeated runs."""

    def __init__(
        self,
        library: str | os.PathLike[str] | None = None,
        *,
        precision: PrecisionProfile | str = PrecisionProfile.FLOAT64,
        time_precision: str | None = None,
    ) -> None:
        self.core = CoreEvaluator(
            library, precision=precision, time_precision=time_precision
        )

    @property
    def precision(self) -> PrecisionProfile:
        """Return the validated precision of the native evaluator."""

        return self.core.precision

    def compile(self, network: Network) -> "CompiledNetwork":
        """Resolve and compile a network for repeated execution."""

        if not isinstance(network, Network):
            raise ResolutionError("compile() requires an immutable Network")
        provenance = _network_precision(network)
        if provenance is not None and provenance is not self.precision:
            raise ResolutionError(
                f"network precision {provenance.value!r} does not match "
                f"engine precision {self.precision.value!r}"
            )
        if self.precision is PrecisionProfile.FLOAT64:
            resolved = network.graph.resolve()
        else:
            from .target_lowering import resolve_target_graph

            resolved = resolve_target_graph(network.graph, self.core)
        return CompiledNetwork(network, resolved.compile(self.core))


class CompiledNetwork:
    """Reusable compiled C graph with high-level inputs, recording, and results."""

    def __init__(self, network: Network, compiled: CompiledResolvedGraph) -> None:
        self.network = network
        self.compiled = compiled

    @property
    def precision(self) -> PrecisionProfile:
        """Return the arithmetic contract used to compile this network."""

        return self.execution_plan.precision

    @property
    def closed(self) -> bool:
        """Return whether compiled resources were released."""

        return self.compiled.closed

    @property
    def preferred_execution_path(self) -> str:
        """Capability-selected path before run-specific fallbacks are considered."""

        return self.compiled.preferred_execution_path

    @property
    def execution_plan(self):
        """Backend-neutral plan derived from this network's equations and wiring."""

        return self.compiled.execution_plan

    def compiled_graph_image(self) -> bytes:
        """Return the compiler-free C graph deployment image."""

        return self.compiled._ensure_compiled_scheduler().to_bytes()

    def save_compiled_graph_image(
        self,
        path: str | os.PathLike[str],
    ) -> Path:
        """Write the C graph deployment image without changing the JSON model."""

        return self.compiled._ensure_compiled_scheduler().save_image(path)

    @property
    def last_execution_path(self) -> str | None:
        """Path used by the most recently executed run, if any."""

        return self.compiled.last_execution_path

    def close(self) -> None:
        """Release compiled resources once."""

        self.compiled.close()

    def __enter__(self) -> "CompiledNetwork":
        if self.closed:
            raise RuntimeError("compiled network is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _model_names(self, node: int) -> tuple[str, ...]:
        resolved = self.compiled.resolved
        try:
            index = resolved.node_ids.index(node)
        except ValueError as exc:
            raise ResolutionError(f"unknown graph node {node}") from exc
        model = resolved.models[index]
        return (
            (model.state_name,)
            if isinstance(model, ResolvedScalarLIF)
            else tuple(model.state_names)
        )

    def _indices(self, node: int, variables: tuple[str, ...] | None):
        if variables is None:
            return None
        names = self._model_names(node)
        missing = [name for name in variables if name not in names]
        if missing:
            raise ResolutionError(
                f"node {node} does not expose state variable(s): {', '.join(missing)}"
            )
        return tuple(names.index(name) for name in variables)

    def _recording(
        self, plan: RecordingPlan | None, t_end: float
    ) -> _PreparedRecording:
        if plan is None:
            return _PreparedRecording(None, (), None, None)
        if not isinstance(plan, RecordingPlan):
            raise ResolutionError("recording must be a RecordingPlan")
        inspections = []
        for state in plan.states:
            assert isinstance(state, StateRecording)
            nodes = self.network.node_ids(state.targets)
            times = state.sampling.times(t_end)
            for time in times:
                for node in nodes:
                    inspections.append(
                        StateInspectionRequest(
                            time,
                            node,
                            self._indices(node, state.variables),
                        )
                    )
        inspections.sort(key=lambda item: (item.t, item.node))
        trace = plan.trace
        config = None
        if trace is not None:
            nodes = self.network.node_ids(trace.targets)
            kinds = None
            if trace.kinds is not None:
                normalized = []
                for item in trace.kinds:
                    if isinstance(item, TraceKind):
                        normalized.append(item)
                    elif isinstance(item, str):
                        try:
                            normalized.append(TraceKind[item.upper()])
                        except KeyError as exc:
                            raise ResolutionError(f"unknown trace kind '{item}'") from exc
                    else:
                        raise ResolutionError("trace kinds must be TraceKind or names")
                kinds = frozenset(normalized)
            state_indices = None
            if trace.variables is not None:
                by_node = tuple(self._indices(node, trace.variables) for node in nodes)
                if by_node and any(value != by_node[0] for value in by_node[1:]):
                    raise ResolutionError(
                        "trace variable names must occupy the same state indices on all selected nodes"
                    )
                state_indices = None if not by_node else by_node[0]
            config = RecordingConfig(
                kinds=kinds,
                nodes=nodes,
                capture_state=trace.capture_state,
                state_indices=state_indices,
                capacity=trace.capacity,
            )
        spike_targets = None if plan.spikes is None else plan.spikes.targets
        return _PreparedRecording(config, tuple(inspections), spike_targets, trace)

    def _mapped_inputs(
        self, inputs: Mapping[str, object] | None
    ) -> tuple[
        list[SpikeInput],
        list[DriveInput],
        list[ScalarInput],
        list[ModulationInput],
    ]:
        spikes: list[SpikeInput] = []
        drives: list[DriveInput] = []
        scalars: list[ScalarInput] = []
        modulations: list[ModulationInput] = []
        input_ports = {port.id for port in self.network.graph.input_ports}
        modulator_ports = {port.id for port in self.network.graph.modulator_ports}
        known = input_ports | modulator_ports
        for port, value in (inputs or {}).items():
            if port not in known:
                raise ResolutionError(f"unknown input port '{port}'")
            values = tuple(value) if _is_sequence(value) and not isinstance(
                value, (SpikeTrain, DriveSeries, ModulationSeries)
            ) else (value,)
            for item in values:
                if port in modulator_ports:
                    if not isinstance(item, ModulationSeries):
                        raise ResolutionError(
                            f"modulator port '{port}' requires ModulationSeries"
                        )
                    times = tuple(
                        _finite(time, "modulation time") for time in item.times
                    )
                    amplitudes = (
                        tuple(
                            _finite(amplitude, "modulation value")
                            for amplitude in item.values
                        )
                        if _is_sequence(item.values)
                        else (_finite(item.values, "modulation value"),) * len(times)
                    )
                    if len(amplitudes) != len(times):
                        raise ResolutionError(
                            "modulation times and values must have equal length"
                        )
                    modulations.extend(
                        ModulationInput(time, port, amplitude)
                        for time, amplitude in zip(times, amplitudes)
                    )
                    continue
                if isinstance(item, ModulationSeries):
                    raise ResolutionError(
                        f"input port '{port}' does not accept ModulationSeries"
                    )
                if isinstance(item, ScalarPresentation):
                    scalars.append(
                        ScalarInput(
                            _finite(item.t_start, "presentation start"),
                            _finite(item.t_end, "presentation end"),
                            port,
                            _finite(item.value, "presentation value"),
                        )
                    )
                elif isinstance(item, SpikeTrain):
                    times = tuple(_finite(time, "spike time") for time in item.times)
                    amplitudes = (
                        tuple(_finite(amplitude, "spike value") for amplitude in item.values)
                        if _is_sequence(item.values)
                        else (_finite(item.values, "spike value"),) * len(times)
                    )
                    if len(amplitudes) != len(times):
                        raise ResolutionError("spike times and values must have equal length")
                    spikes.extend(
                        SpikeInput(time, port, amplitude)
                        for time, amplitude in zip(times, amplitudes)
                    )
                elif isinstance(item, DriveSeries):
                    times = tuple(_finite(time, "drive time") for time in item.times)
                    amplitudes = tuple(
                        _finite(amplitude, "drive value") for amplitude in item.values
                    )
                    if len(amplitudes) != len(times):
                        raise ResolutionError("drive times and values must have equal length")
                    drives.extend(
                        DriveInput(time, port, amplitude)
                        for time, amplitude in zip(times, amplitudes)
                    )
                else:
                    raise ResolutionError(
                        f"input port '{port}' received unsupported data {type(item).__name__}"
                    )
        return spikes, drives, scalars, modulations

    def run(
        self,
        duration: float,
        *,
        inputs: Mapping[str, object] | None = None,
        spike_inputs: Sequence[SpikeInput] = (),
        drive_inputs: Sequence[DriveInput] = (),
        scalar_inputs: Sequence[ScalarInput] = (),
        modulation_inputs: Sequence[ModulationInput] = (),
        recording: RecordingPlan | None = None,
        seed: int = 0,
        options: RunOptions | None = None,
        decode_start: float = 0.0,
        decode_windows: Sequence[DecodeWindow] | Mapping[str, Sequence[DecodeWindow]] | None = None,
        decode_queries: Mapping[str, Sequence[DecodeQuery]] | None = None,
    ) -> SimulationResult:
        """Execute one high-level run with inputs and recording options."""

        t_end = _finite(duration, "duration")
        if t_end < 0.0:
            raise ResolutionError("duration must be nonnegative")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**64 - 1:
            raise ResolutionError("seed must be an unsigned 64-bit integer")
        limits = options or RunOptions()
        (
            generated_spikes,
            generated_drives,
            generated_scalars,
            generated_modulations,
        ) = self._mapped_inputs(inputs)
        prepared = self._recording(recording, t_end)
        output_capacity = limits.output_capacity
        if recording is not None and recording.spikes is None:
            output_capacity = 0

        def execute(config: RecordingConfig | None) -> GraphRunResult:
            """Execute with the prepared high-level inputs and one trace sink."""

            return self.compiled.run(
                spike_inputs=(*spike_inputs, *generated_spikes),
                drive_inputs=(*drive_inputs, *generated_drives),
                scalar_inputs=(*scalar_inputs, *generated_scalars),
                modulation_inputs=(*modulation_inputs, *generated_modulations),
                t_end=t_end,
                decode_start=decode_start,
                decode_windows=decode_windows,
                decode_queries=decode_queries,
                encoder_seed=seed,
                encoder_spike_capacity=limits.encoder_spike_capacity,
                decoder_event_capacity=limits.decoder_event_capacity,
                queue_capacity=limits.queue_capacity,
                output_capacity=output_capacity,
                same_time_cascade_limit=limits.same_time_cascade_limit,
                stochastic_seed=seed,
                recording=config,
                inspections=prepared.inspections,
                return_final_state=limits.return_final_state,
            )

        trace_path = None
        if prepared.trace is not None and prepared.trace.path is not None:
            trace_path = Path(prepared.trace.path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            assert prepared.config is not None
            metadata = TraceArtifactMetadata.from_resolved_graph(
                self.compiled.resolved,
                prepared.config,
                extra={
                    "network_name": self.network.name,
                    "duration": t_end,
                    "encoder_seed": seed,
                    **({"lacuna_precision": self.precision.to_record()}
                       if self.precision is not PrecisionProfile.FLOAT64 else {}),
                },
            )
            with TraceArtifactWriter(
                trace_path,
                metadata,
                chunk_records=prepared.trace.chunk_records,
                compress=prepared.trace.compress,
            ) as writer:
                raw = execute(replace(prepared.config, capacity=0, consumer=writer))
                writer.set_summary({"run_stats": asdict(raw.core.stats)})
        else:
            raw = execute(prepared.config)
        return SimulationResult(
            self.network,
            raw,
            spike_targets=prepared.spike_targets,
            trace_path=trace_path,
            resolved=self.compiled.resolved,
        )

    def start_run(
        self,
        duration: float,
        *,
        recording: RecordingPlan | None = None,
        seed: int = 0,
        options: RunOptions | None = None,
        decode_start: float = 0.0,
        decode_windows: Sequence[DecodeWindow] | Mapping[str, Sequence[DecodeWindow]] | None = None,
        decode_queries: Mapping[str, Sequence[DecodeQuery]] | None = None,
    ) -> "IncrementalSimulationRun":
        """Open a resumable run while retaining C scheduler and codec state."""

        t_end = _finite(duration, "duration")
        if t_end < 0.0:
            raise ResolutionError("duration must be nonnegative")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**64 - 1:
            raise ResolutionError("seed must be an unsigned 64-bit integer")
        limits = options or RunOptions()
        prepared = self._recording(recording, t_end)
        output_capacity = limits.output_capacity
        if recording is not None and recording.spikes is None:
            output_capacity = 0
        raw = self.compiled.create_incremental_run(
            t_end=t_end,
            decode_start=decode_start,
            decode_windows=decode_windows,
            decode_queries=decode_queries,
            decoder_event_capacity=limits.decoder_event_capacity,
            queue_capacity=limits.queue_capacity,
            output_capacity=output_capacity,
            same_time_cascade_limit=limits.same_time_cascade_limit,
            stochastic_seed=seed,
            encoder_seed=seed,
            encoder_spike_capacity=limits.encoder_spike_capacity,
            encoder_drive_capacity=limits.encoder_drive_capacity,
            return_final_state=limits.return_final_state,
        )
        writer = None
        trace_path = None
        try:
            if prepared.trace is not None and prepared.trace.path is not None:
                trace_path = Path(prepared.trace.path)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                assert prepared.config is not None
                metadata = TraceArtifactMetadata.from_resolved_graph(
                    self.compiled.resolved,
                    prepared.config,
                    extra={
                        "network_name": self.network.name,
                        "duration": t_end,
                        "encoder_seed": seed,
                        "incremental": True,
                        **({"lacuna_precision": self.precision.to_record()}
                           if self.precision is not PrecisionProfile.FLOAT64 else {}),
                    },
                )
                writer = TraceArtifactWriter(
                    trace_path,
                    metadata,
                    chunk_records=prepared.trace.chunk_records,
                    compress=prepared.trace.compress,
                )
        except Exception:
            raw.close()
            raise
        return IncrementalSimulationRun(
            self,
            raw,
            prepared,
            writer,
            trace_path,
        )

    def visualizer(
        self,
        duration: float,
        *,
        step: float = 1.0,
        sample_interval: float = 0.1,
        seed: int = 0,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        """Create a controllable interactive visualizer over a fresh live run."""

        from .visualization import NetworkVisualizer

        return NetworkVisualizer(
            self,
            duration,
            step=step,
            sample_interval=sample_interval,
            seed=seed,
            host=host,
            port=port,
        )

    def visualize(
        self,
        duration: float,
        *,
        step: float = 1.0,
        sample_interval: float = 0.1,
        seed: int = 0,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
        block: bool = True,
    ):
        """Open the interactive visualizer, optionally blocking until it closes."""

        viewer = self.visualizer(
            duration,
            step=step,
            sample_interval=sample_interval,
            seed=seed,
            host=host,
            port=port,
        )
        viewer.start(open_browser=open_browser)
        if block:
            viewer.wait()
        return viewer


class IncrementalSimulationRun:
    """High-level resumable run with half-open advances and one final boundary."""

    def __init__(
        self,
        parent: CompiledNetwork,
        raw,
        recording: _PreparedRecording,
        writer: TraceArtifactWriter | None,
        trace_path: Path | None,
    ) -> None:
        self.parent = parent
        self.raw = raw
        self.recording = recording
        self.writer = writer
        self.trace_path = trace_path
        self._inspection_cursor = 0
        self._failed = False

    @property
    def frontier(self) -> float:
        """Return the settled open simulation frontier."""

        return self.raw.frontier

    @property
    def finished(self) -> bool:
        """Return whether the final horizon has been sealed."""

        return self.raw.finished

    @property
    def closed(self) -> bool:
        """Return whether incremental resources were released."""

        return self.raw.closed

    def __enter__(self) -> "IncrementalSimulationRun":
        if self.closed:
            raise RuntimeError("incremental simulation run is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self._failed = True
        self.close()

    def close(self) -> None:
        """Release incremental execution and trace resources."""

        self.raw.close()
        if self.writer is not None and not self.writer.closed:
            self.writer.abort()

    def _config(self) -> RecordingConfig | None:
        config = self.recording.config
        if config is None or self.writer is None:
            return config
        return replace(config, capacity=0, consumer=self.writer)

    def _inspections(
        self, boundary: float, *, inclusive: bool
    ) -> tuple[StateInspectionRequest, ...]:
        values = self.recording.inspections
        start = self._inspection_cursor
        end = start
        while end < len(values) and (
            values[end].t <= boundary if inclusive else values[end].t < boundary
        ):
            end += 1
        self._inspection_cursor = end
        return values[start:end]

    def _result(self, raw: GraphRunResult) -> SimulationResult:
        return SimulationResult(
            self.parent.network,
            raw,
            spike_targets=self.recording.spike_targets,
            trace_path=self.trace_path,
            resolved=self.parent.compiled.resolved,
        )

    def advance(
        self,
        until: float,
        *,
        inputs: Mapping[str, object] | None = None,
        spike_inputs: Sequence[SpikeInput] = (),
        drive_inputs: Sequence[DriveInput] = (),
        scalar_inputs: Sequence[ScalarInput] = (),
        modulation_inputs: Sequence[ModulationInput] = (),
        inspections: Sequence[StateInspectionRequest] = (),
    ) -> SimulationResult:
        """Advance to an open boundary and return this segment's results."""

        if self._failed:
            raise RuntimeError("incremental simulation run is unusable after failure")
        (
            generated_spikes,
            generated_drives,
            generated_scalars,
            generated_modulations,
        ) = self.parent._mapped_inputs(inputs)
        try:
            raw = self.raw.advance_until(
                _finite(until, "advance boundary"),
                spike_inputs=(*spike_inputs, *generated_spikes),
                drive_inputs=(*drive_inputs, *generated_drives),
                scalar_inputs=(*scalar_inputs, *generated_scalars),
                modulation_inputs=(*modulation_inputs, *generated_modulations),
                recording=self._config(),
                inspections=(
                    *self._inspections(float(until), inclusive=False),
                    *tuple(inspections),
                ),
            )
        except Exception:
            self._failed = True
            if self.writer is not None and not self.writer.closed:
                self.writer.abort()
            raise
        return self._result(raw)

    def finish(
        self,
        *,
        inputs: Mapping[str, object] | None = None,
        spike_inputs: Sequence[SpikeInput] = (),
        drive_inputs: Sequence[DriveInput] = (),
        scalar_inputs: Sequence[ScalarInput] = (),
        modulation_inputs: Sequence[ModulationInput] = (),
        inspections: Sequence[StateInspectionRequest] = (),
    ) -> SimulationResult:
        """Seal the final horizon and return remaining results."""

        if self._failed:
            raise RuntimeError("incremental simulation run is unusable after failure")
        (
            generated_spikes,
            generated_drives,
            generated_scalars,
            generated_modulations,
        ) = self.parent._mapped_inputs(inputs)
        try:
            raw = self.raw.finish(
                spike_inputs=(*spike_inputs, *generated_spikes),
                drive_inputs=(*drive_inputs, *generated_drives),
                scalar_inputs=(*scalar_inputs, *generated_scalars),
                modulation_inputs=(*modulation_inputs, *generated_modulations),
                recording=self._config(),
                inspections=(
                    *self._inspections(self.raw.core_run.t_end, inclusive=True),
                    *tuple(inspections),
                ),
            )
            if self.writer is not None:
                self.writer.set_summary(
                    {"run_stats": asdict(self.raw.cumulative_stats)}
                )
                self.writer.close()
        except Exception:
            self._failed = True
            if self.writer is not None and not self.writer.closed:
                self.writer.abort()
            raise
        return self._result(raw)


__all__ = [
    "SpikeTrain",
    "DriveSeries",
    "ModulationSeries",
    "ScalarPresentation",
    "NetworkSpike",
    "StateSample",
    "FinalState",
    "SpikeSeries",
    "StateSeries",
    "SimulationResult",
    "Engine",
    "CompiledNetwork",
    "IncrementalSimulationRun",
]
