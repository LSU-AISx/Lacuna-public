"""Offline, C-evaluated state reconstruction from persisted traces."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from .ffi import (
    AugmentedState,
    CoreEvaluator,
    StateInspectionRequest,
    TraceKind,
    TraceRecord,
)
from .expr import root_variable_dependencies
from .graph import Graph, InputMode, ResolvedGraph
from .ir import ResolvedScalarLIF, ResolvedSteppedNeuron
from .precision import PrecisionProfile
from .tracefile import TraceArtifactIdentityError, TraceArtifactReader


class ReconstructionError(ValueError):
    """Base error for an invalid or unsupported offline state query."""


class ReconstructionUnavailableError(ReconstructionError):
    """The recording does not contain enough causal state for a query."""


class ReconstructionArtifactError(ReconstructionError):
    """A valid trace file violates the stronger reconstruction contract."""


class ReconstructionSupport(str, Enum):
    """Accuracy available for an offline state query."""

    EXACT = "EXACT"
    EVENT_ONLY = "EVENT_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


class ReconstructionSource(str, Enum):
    """Source used to produce one reconstructed state sample."""

    INITIAL_STATE = "INITIAL_STATE"
    RECORDED_EVENT = "RECORDED_EVENT"
    ANALYTICAL_PROPAGATION = "ANALYTICAL_PROPAGATION"
    NUMERICAL_PROPAGATION = "NUMERICAL_PROPAGATION"


@dataclass(frozen=True)
class ReconstructionDiagnostic:
    """Required and missing trace data for one reconstruction target."""

    node: int
    state_indices: tuple[int, ...]
    state_names: tuple[str, ...]
    support: ReconstructionSupport
    required_state_names: tuple[str, ...]
    missing_state_names: tuple[str, ...]
    required_trace_kinds: tuple[TraceKind, ...]
    missing_trace_kinds: tuple[TraceKind, ...]
    reasons: tuple[str, ...]

    @property
    def exact_between_events(self) -> bool:
        """Return whether continuous state can be recovered exactly."""

        return self.support is ReconstructionSupport.EXACT


@dataclass(frozen=True)
class ReconstructedState:
    """Offline state sample and the event anchor used to reconstruct it."""

    t: float
    node: int
    state_indices: tuple[int, ...]
    state_names: tuple[str, ...]
    values: tuple[float, ...]
    anchor_time: float
    clamped: bool
    source: ReconstructionSource


@dataclass(frozen=True)
class _PreparedQuery:
    position: int
    t: float
    node: int
    index: int
    selected: tuple[int, ...]
    names: tuple[str, ...]
    diagnostic: ReconstructionDiagnostic


_STATE_TRANSITION_KINDS = frozenset(
    {
        TraceKind.DRIVE_UPDATE,
        TraceKind.REFRACTORY_RELEASE,
        TraceKind.DEPOSIT_APPLY,
        TraceKind.RESET,
        TraceKind.REFRACTORY_ENTER,
        TraceKind.FINAL_STATE,
    }
)


class TraceReconstructor:
    """Answer explicit-time state queries from one complete trace artifact."""

    def __init__(
        self,
        core: CoreEvaluator,
        resolved: ResolvedGraph | Graph,
        reader: TraceArtifactReader,
    ) -> None:
        if not isinstance(core, CoreEvaluator):
            raise TypeError("core must be a CoreEvaluator")
        if not isinstance(resolved, (ResolvedGraph, Graph)):
            raise TypeError("resolved must be a ResolvedGraph or Graph")
        if not isinstance(reader, TraceArtifactReader):
            raise TypeError("reader must be a TraceArtifactReader")
        if reader.closed:
            raise ReconstructionError("trace artifact reader is closed")
        if not reader.complete:
            raise ReconstructionUnavailableError(
                "offline reconstruction requires a complete trace artifact"
            )
        profile = reader.metadata.precision
        if core.precision is not profile:
            raise TraceArtifactIdentityError(
                "trace precision does not match the C evaluator"
            )
        if isinstance(resolved, Graph):
            if profile is PrecisionProfile.FLOAT64:
                resolved = resolved.resolve()
            else:
                from .target_lowering import resolve_target_graph

                resolved = resolve_target_graph(resolved, core)
        if resolved.precision is not profile:
            raise TraceArtifactIdentityError(
                "trace precision does not match the resolved graph"
            )
        graph_hash = hashlib.sha256(
            resolved.graph.to_text().encode("utf-8")
        ).hexdigest()
        if reader.metadata.graph_sha256 != graph_hash:
            raise TraceArtifactIdentityError(
                "trace artifact graph hash does not match the resolved graph"
            )
        if reader.metadata.time_unit != resolved.graph.time_unit:
            raise TraceArtifactIdentityError(
                "trace artifact time unit does not match the resolved graph"
            )
        metadata_nodes = {item.node: item for item in reader.metadata.nodes}
        if set(metadata_nodes) != set(resolved.node_ids):
            raise TraceArtifactIdentityError(
                "trace artifact node identities do not match the resolved graph"
            )
        for node, model in zip(resolved.node_ids, resolved.models):
            item = metadata_nodes[node]
            names = self._state_names(model)
            if (
                item.state_names != names
                or item.model_hash != model.model_hash
                or item.resolution_key != model.resolution_key
            ):
                raise TraceArtifactIdentityError(
                    f"trace artifact model identity does not match graph node {node}"
                )
        self.core = core
        self.resolved = resolved
        self.reader = reader
        self._node_index = {
            node: index for index, node in enumerate(resolved.node_ids)
        }

    @staticmethod
    def _state_names(model) -> tuple[str, ...]:
        return (
            (model.state_name,)
            if isinstance(model, ResolvedScalarLIF)
            else tuple(model.state_names)
        )

    def _selection(
        self, node: int, state_indices: Iterable[int] | None
    ) -> tuple[int, tuple[int, ...], tuple[str, ...]]:
        if not isinstance(node, int) or isinstance(node, bool) or node not in self._node_index:
            raise ReconstructionError("reconstruction references an unknown graph node")
        index = self._node_index[node]
        names = self._state_names(self.resolved.models[index])
        selected = (
            tuple(range(len(names)))
            if state_indices is None
            else tuple(state_indices)
        )
        if (
            len(selected) != len(set(selected))
            or any(
                not isinstance(item, int)
                or isinstance(item, bool)
                or not 0 <= item < len(names)
                for item in selected
            )
        ):
            raise ReconstructionError(
                "reconstruction state indices must be unique valid local indices"
            )
        return index, selected, names

    def _required_kinds(self, node: int, index: int) -> tuple[TraceKind, ...]:
        model = self.resolved.models[index]
        required = {TraceKind.RESET, TraceKind.FINAL_STATE}
        if any(edge.post == node for edge in self.resolved.graph.edges) or any(
            port.node == node and port.mode is InputMode.SPIKE
            for port in self.resolved.graph.input_ports
        ):
            required.add(TraceKind.DEPOSIT_APPLY)
        if any(
            port.node == node and port.mode is InputMode.DRIVE
            for port in self.resolved.graph.input_ports
        ):
            required.add(TraceKind.DRIVE_UPDATE)
        if model.refractory > 0.0:
            required.update(
                {TraceKind.REFRACTORY_ENTER, TraceKind.REFRACTORY_RELEASE}
            )
        return tuple(sorted(required, key=int))

    def _required_state_indices(
        self, index: int, selected: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Find the local anchor variables needed by selected propagation roots."""

        model = self.resolved.models[index]
        dag = model.propagation_dag
        state_count = len(self._state_names(model))
        if isinstance(model, ResolvedSteppedNeuron):
            return tuple(range(state_count))
        expected_variables = ("Delta",) + tuple(
            f"x{item}" for item in range(state_count)
        )
        if dag.variables != expected_variables:
            raise ReconstructionArtifactError(
                "resolved propagation variable layout is not canonical"
            )
        required = set(selected)
        for roots in (model.normal_roots, model.clamped_roots):
            if len(roots) != state_count:
                raise ReconstructionArtifactError(
                    "resolved propagation root layout is not canonical"
                )
            for state_index in selected:
                for variable in root_variable_dependencies(
                    dag, roots[state_index]
                ):
                    if variable == "Delta":
                        continue
                    if not variable.startswith("x") or not variable[1:].isdigit():
                        raise ReconstructionArtifactError(
                            f"propagation root depends on unknown variable '{variable}'"
                        )
                    variable_index = int(variable[1:])
                    if not 0 <= variable_index < state_count:
                        raise ReconstructionArtifactError(
                            f"propagation root depends on invalid state variable '{variable}'"
                        )
                    required.add(variable_index)
        return tuple(sorted(required))

    def diagnostic(
        self,
        node: int,
        state_indices: Iterable[int] | None = None,
    ) -> ReconstructionDiagnostic:
        """Report whether selected state can be reconstructed between events."""

        if self.reader.closed:
            raise ReconstructionError("trace artifact reader is closed")
        index, selected, names = self._selection(node, state_indices)
        metadata = self.reader.metadata
        recorded_nodes = (
            set(self.resolved.node_ids)
            if metadata.recorded_nodes is None
            else set(metadata.recorded_nodes)
        )
        recorded_indices = (
            set(range(len(names)))
            if metadata.state_indices is None
            else set(metadata.state_indices)
        )
        required_kinds = self._required_kinds(node, index)
        recorded_kinds = {TraceKind[name] for name in metadata.recorded_kinds}
        missing_kinds = tuple(
            kind for kind in required_kinds if kind not in recorded_kinds
        )
        required_indices = self._required_state_indices(index, selected)
        missing_indices = tuple(
            item for item in required_indices if item not in recorded_indices
        )
        missing_requested = tuple(
            item for item in selected if item not in recorded_indices
        )
        reasons = []
        if node not in recorded_nodes:
            reasons.append("the node was excluded from the recording")
        if not metadata.capture_state:
            reasons.append("state capture was disabled")
        if missing_requested:
            reasons.append(
                "requested state was not recorded: "
                + ", ".join(names[item] for item in missing_requested)
            )
        if missing_kinds:
            reasons.append(
                "causal transition kinds were not recorded: "
                + ", ".join(kind.name for kind in missing_kinds)
            )
        unavailable = (
            node not in recorded_nodes
            or not metadata.capture_state
            or bool(missing_requested)
            or bool(missing_kinds)
        )
        if unavailable:
            support = ReconstructionSupport.UNAVAILABLE
        elif missing_indices:
            support = ReconstructionSupport.EVENT_ONLY
            reasons.append(
                "between-event propagation also requires: "
                + ", ".join(names[item] for item in missing_indices)
            )
        else:
            support = ReconstructionSupport.EXACT
        return ReconstructionDiagnostic(
            node=node,
            state_indices=selected,
            state_names=tuple(names[item] for item in selected),
            support=support,
            required_state_names=tuple(names[item] for item in required_indices),
            missing_state_names=tuple(names[item] for item in missing_indices),
            required_trace_kinds=required_kinds,
            missing_trace_kinds=missing_kinds,
            reasons=tuple(reasons),
        )

    def reconstruct_one(
        self,
        t: float,
        node: int,
        state_indices: Iterable[int] | None = None,
    ) -> ReconstructedState:
        """Reconstruct one node state at one explicit time."""

        selected = None if state_indices is None else tuple(state_indices)
        return self.reconstruct(
            (StateInspectionRequest(t, node, selected),)
        )[0]

    def reconstruct(
        self, requests: Sequence[StateInspectionRequest]
    ) -> tuple[ReconstructedState, ...]:
        """Reconstruct ordered state queries using shared trace scans."""

        if self.reader.closed:
            raise ReconstructionError("trace artifact reader is closed")
        normalized = tuple(requests)
        if any(not isinstance(item, StateInspectionRequest) for item in normalized):
            raise ReconstructionError(
                "reconstruction requests must be StateInspectionRequest values"
            )
        prepared = []
        for position, request in enumerate(normalized):
            if (
                not isinstance(request.t, (int, float))
                or isinstance(request.t, bool)
                or not math.isfinite(request.t)
                or request.t < 0.0
            ):
                raise ReconstructionError(
                    "reconstruction times must be finite and nonnegative"
                )
            index, selected, names = self._selection(
                request.node, request.state_indices
            )
            diagnostic = self.diagnostic(request.node, selected)
            if diagnostic.support is ReconstructionSupport.UNAVAILABLE:
                detail = "; ".join(diagnostic.reasons) or "insufficient recording"
                raise ReconstructionUnavailableError(
                    f"node {request.node} cannot be reconstructed: {detail}"
                )
            prepared.append(
                _PreparedQuery(
                    position,
                    self.resolved.precision.round_time(request.t),
                    request.node,
                    index,
                    selected,
                    names,
                    diagnostic,
                )
            )
        if not prepared:
            return ()
        by_node: dict[int, list[_PreparedQuery]] = {}
        for query in prepared:
            by_node.setdefault(query.node, []).append(query)
        results: list[ReconstructedState | None] = [None] * len(prepared)
        for node, queries in by_node.items():
            reconstructed = self._reconstruct_node(
                sorted(queries, key=lambda item: (item.t, item.position)),
                self.reader.iter_records(nodes=(node,)),
            )
            for query, value in reconstructed:
                results[query.position] = value
        if any(item is None for item in results):
            raise ReconstructionArtifactError(
                "reconstruction did not produce every requested result"
            )
        return tuple(item for item in results if item is not None)

    def _reconstruct_node(
        self,
        queries: Sequence[_PreparedQuery],
        records: Iterable[TraceRecord],
    ) -> tuple[tuple[_PreparedQuery, ReconstructedState], ...]:
        first = queries[0]
        node = first.node
        index = first.index
        names = first.names
        model = self.resolved.models[index]
        initial = self.resolved.initial_values[index]
        initial_values = (
            (float(initial),)
            if isinstance(initial, (int, float))
            else tuple(map(float, initial))
        )
        if len(initial_values) != len(names):
            raise ReconstructionArtifactError(
                f"node {node} initial state layout is invalid"
            )
        bindings = dict(model.bindings)
        clamped = False
        anchor_time = 0.0
        anchor_values = initial_values
        required = set(first.diagnostic.required_trace_kinds)
        expected_recorded = (
            set(range(len(names)))
            if self.reader.metadata.state_indices is None
            else set(self.reader.metadata.state_indices)
        )
        direct_time: float | None = None
        direct_values: dict[int, float] | None = None
        final_count = 0
        t_end: float | None = None
        cursor = 0
        reconstructed: list[tuple[_PreparedQuery, ReconstructedState]] = []

        def materialize(query: _PreparedQuery) -> ReconstructedState:
            """Propagate the current anchor or use an exact recorded sample."""

            if query.index != index or query.names != names:
                raise ReconstructionArtifactError(
                    f"node {node} query layouts disagree"
                )
            if (
                direct_time == query.t
                and direct_values is not None
                and set(query.selected).issubset(direct_values)
            ):
                return ReconstructedState(
                    t=query.t,
                    node=node,
                    state_indices=query.selected,
                    state_names=tuple(names[item] for item in query.selected),
                    values=tuple(direct_values[item] for item in query.selected),
                    anchor_time=query.t,
                    clamped=clamped,
                    source=ReconstructionSource.RECORDED_EVENT,
                )
            if query.t == 0.0 and anchor_time == 0.0:
                return ReconstructedState(
                    t=query.t,
                    node=node,
                    state_indices=query.selected,
                    state_names=tuple(names[item] for item in query.selected),
                    values=tuple(initial_values[item] for item in query.selected),
                    anchor_time=0.0,
                    clamped=clamped,
                    source=ReconstructionSource.INITIAL_STATE,
                )
            if query.diagnostic.support is ReconstructionSupport.EVENT_ONLY:
                raise ReconstructionUnavailableError(
                    f"node {node} has only event-time state at t={query.t}; "
                    "the query falls between recorded state anchors"
                )
            if isinstance(model, ResolvedSteppedNeuron):
                stepped, _ = self.core.advance_stepped(
                    model,
                    AugmentedState(anchor_values, anchor_time),
                    query.t,
                    clamped=clamped,
                    parameter_bindings=bindings,
                )
                values = tuple(stepped.values[item] for item in query.selected)
                source = ReconstructionSource.NUMERICAL_PROPAGATION
            else:
                values = self.core.advance_analytical_selected(
                    model,
                    AugmentedState(anchor_values, anchor_time),
                    query.t,
                    query.selected,
                    clamped=clamped,
                    parameter_bindings=bindings,
                )
                source = ReconstructionSource.ANALYTICAL_PROPAGATION
            return ReconstructedState(
                t=query.t,
                node=node,
                state_indices=query.selected,
                state_names=tuple(names[item] for item in query.selected),
                values=values,
                anchor_time=anchor_time,
                clamped=clamped,
                source=source,
            )

        for record in records:
            if final_count:
                raise ReconstructionArtifactError(
                    f"node {node} has records after its final-state record"
                )
            while cursor < len(queries) and queries[cursor].t < record.t:
                query = queries[cursor]
                reconstructed.append((query, materialize(query)))
                cursor += 1
            if direct_time != record.t:
                direct_time = record.t
                direct_values = None
            state_map = dict(zip(record.state_indices, record.after))
            if record.kind in required and record.kind in _STATE_TRANSITION_KINDS:
                if not expected_recorded.issubset(state_map):
                    raise ReconstructionArtifactError(
                        f"node {node} {record.kind.name} record lacks required state"
                    )
            if record.kind is TraceKind.DRIVE_UPDATE:
                if record.subject is None or record.subject >= len(
                    model.propagation_dag.parameters
                ):
                    raise ReconstructionArtifactError(
                        f"node {node} drive update has an invalid parameter binding"
                    )
                parameter = model.propagation_dag.parameters[record.subject]
                if parameter not in model.drive_parameters:
                    raise ReconstructionArtifactError(
                        f"node {node} drive update targets an unapproved parameter"
                    )
                bindings[parameter] = record.value
            elif record.kind is TraceKind.REFRACTORY_ENTER:
                clamped = True
            elif record.kind is TraceKind.REFRACTORY_RELEASE:
                clamped = False
            if expected_recorded.issubset(state_map):
                merged = list(anchor_values)
                for state_index, value in state_map.items():
                    merged[state_index] = value
                anchor_values = tuple(merged)
                anchor_time = record.t
            if state_map:
                direct_values = state_map
            if record.kind is TraceKind.FINAL_STATE:
                final_count += 1
                t_end = record.t
        if final_count != 1 or t_end is None:
            raise ReconstructionArtifactError(
                f"node {node} requires exactly one final-state record"
            )
        if queries[-1].t > t_end:
            raise ReconstructionError(
                f"reconstruction time {queries[-1].t} "
                f"exceeds artifact end time {t_end}"
            )
        while cursor < len(queries):
            query = queries[cursor]
            reconstructed.append((query, materialize(query)))
            cursor += 1
        return tuple(reconstructed)
