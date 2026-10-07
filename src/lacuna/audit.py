"""Independent invariant checks for complete mixed-network causal traces.

The auditor intentionally does not propagate neuron state or solve crossings.  It
checks scheduler facts that can be derived from the immutable topology, the
complete causal record, and the result returned by the C evaluator.
"""

from __future__ import annotations

import math
import hashlib
import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Iterable, Sequence

from .ffi import (
    MixedEdge,
    MixedRunResult,
    TraceKind,
    TracePhase,
    TraceRecord,
)
from .ir import (
    NeuronPolarity,
    ResolvedAdaptiveLIF,
    ResolvedAlphaLIF,
    ResolvedPerEdgeLIF,
    ResolvedReactiveIF,
    ResolvedScalarLIF,
    ResolvedSteppedNeuron,
)

if TYPE_CHECKING:
    from .graph import GraphRunResult, ResolvedGraph
    from .tracefile import TraceArtifactReader

ExecutableModel = (
    ResolvedScalarLIF
    | ResolvedAlphaLIF
    | ResolvedAdaptiveLIF
    | ResolvedPerEdgeLIF
    | ResolvedReactiveIF
    | ResolvedSteppedNeuron
)


class TraceAuditError(AssertionError):
    """A complete causal trace violated one independently checkable invariant."""

    def __init__(
        self,
        invariant: str,
        message: str,
        *,
        sequence: int | None = None,
    ) -> None:
        location = "" if sequence is None else f" at trace sequence {sequence}"
        super().__init__(f"{invariant}{location}: {message}")
        self.invariant = invariant
        self.sequence = sequence


@dataclass(frozen=True)
class TraceAuditReport:
    """Counts and checks established by :func:`audit_causal_trace`."""

    record_count: int
    input_records: int
    delivery_records: int
    drive_records: int
    modulation_records: int
    spike_records: int
    stale_prediction_records: int
    confirmed_prediction_records: int
    final_state_records: int
    checks: tuple[str, ...]


_PHASE_BY_KIND = {
    TraceKind.INPUT_SPIKE: TracePhase.DEPOSIT,
    TraceKind.DELIVERY: TracePhase.DEPOSIT,
    TraceKind.DRIVE_UPDATE: TracePhase.BOUNDARY,
    TraceKind.REFRACTORY_RELEASE: TracePhase.BOUNDARY,
    TraceKind.STALE_PREDICTION: TracePhase.PREDICTION,
    TraceKind.PREDICTION_CONFIRMED: TracePhase.PREDICTION,
    TraceKind.DEPOSIT_APPLY: TracePhase.DEPOSIT,
    TraceKind.SPIKE: TracePhase.FIRE,
    TraceKind.RESET: TracePhase.FIRE,
    TraceKind.REFRACTORY_ENTER: TracePhase.FIRE,
    TraceKind.FINAL_STATE: TracePhase.FINAL,
    TraceKind.MODULATION: TracePhase.BOUNDARY,
}

_STATEFUL_KINDS = frozenset(
    {
        TraceKind.DRIVE_UPDATE,
        TraceKind.REFRACTORY_RELEASE,
        TraceKind.PREDICTION_CONFIRMED,
        TraceKind.DEPOSIT_APPLY,
        TraceKind.SPIKE,
        TraceKind.RESET,
        TraceKind.REFRACTORY_ENTER,
        TraceKind.FINAL_STATE,
    }
)

_SUBJECT_KINDS = frozenset(
    {
        TraceKind.INPUT_SPIKE,
        TraceKind.DELIVERY,
        TraceKind.DRIVE_UPDATE,
        TraceKind.MODULATION,
    }
)

_CHECKS = (
    "trace_sequence_and_time",
    "trace_phase_and_payload_contract",
    "full_state_snapshot_contract",
    "same_time_state_continuity",
    "spike_reset_pairing",
    "refractory_state_machine",
    "same_time_deposit_causality",
    "delivery_conservation",
    "prediction_generation_consistency",
    "runtime_accounting_from_trace",
    "final_state_agreement",
)

_PERSISTED_CHECK = "persisted_artifact_identity_and_completeness"


def _fail(
    invariant: str,
    message: str,
    record: TraceRecord | None = None,
) -> None:
    raise TraceAuditError(
        invariant,
        message,
        sequence=None if record is None else record.sequence,
    )


def _state_count(model: ExecutableModel) -> int:
    return len(model.normal_roots)


def _close(left: float, right: float) -> bool:
    return left == right or math.isclose(
        left,
        right,
        rel_tol=2e-12,
        abs_tol=2e-12 * max(1.0, abs(left), abs(right)),
    )


def _crossing_threshold_tolerance(model: ExecutableModel) -> float:
    root_hint = getattr(model, "root_hint", None)
    numerical = getattr(model, "numerical", None)
    if numerical is not None:
        return max(
            2e-12,
            8.0
            * (
                float(numerical.absolute_tolerance)
                + float(numerical.relative_tolerance)
                * max(1.0, abs(model.threshold))
            ),
        )
    relative = float(
        getattr(
            root_hint,
            "relative_tolerance",
            2e-12,
        )
    )
    return max(2e-12, relative) * max(1.0, abs(model.threshold))


def _check_record_contract(
    records: Sequence[TraceRecord],
    models: Sequence[ExecutableModel],
    t_end: float,
) -> None:
    previous_t = -math.inf
    last_after_at_time: dict[tuple[float, int], tuple[float, ...]] = {}
    final_started = False

    for expected_sequence, record in enumerate(records):
        if record.sequence != expected_sequence:
            _fail(
                "trace_sequence_and_time",
                f"expected sequence {expected_sequence}, got {record.sequence}",
                record,
            )
        if (
            not math.isfinite(record.t)
            or record.t < 0.0
            or record.t > t_end
            or record.t < previous_t
        ):
            _fail(
                "trace_sequence_and_time",
                f"invalid or nonchronological time {record.t!r}",
                record,
            )
        previous_t = record.t
        if not 0 <= record.node < len(models):
            _fail(
                "trace_phase_and_payload_contract",
                f"node {record.node} is outside [0, {len(models)})",
                record,
            )
        if record.phase is not _PHASE_BY_KIND[record.kind]:
            _fail(
                "trace_phase_and_payload_contract",
                f"{record.kind.name} has phase {record.phase.name}",
                record,
            )
        if not math.isfinite(record.value):
            _fail(
                "trace_phase_and_payload_contract",
                "record value is not finite",
                record,
            )
        if (record.kind in _SUBJECT_KINDS) != (record.subject is not None):
            _fail(
                "trace_phase_and_payload_contract",
                f"{record.kind.name} has an invalid subject payload",
                record,
            )
        if final_started and record.kind is not TraceKind.FINAL_STATE:
            _fail(
                "trace_phase_and_payload_contract",
                "a non-final record follows final-state emission",
                record,
            )
        final_started = final_started or record.kind is TraceKind.FINAL_STATE

        expected_count = _state_count(models[record.node])
        if record.kind in _STATEFUL_KINDS:
            expected_indices = tuple(range(expected_count))
            if (
                record.state_indices != expected_indices
                or len(record.before) != expected_count
                or len(record.after) != expected_count
            ):
                _fail(
                    "full_state_snapshot_contract",
                    f"{record.kind.name} does not contain all {expected_count} states",
                    record,
                )
            if any(
                not math.isfinite(value)
                for value in (*record.before, *record.after)
            ):
                _fail(
                    "full_state_snapshot_contract",
                    "state snapshot contains a non-finite value",
                    record,
                )
            key = (record.t, record.node)
            previous_after = last_after_at_time.get(key)
            if previous_after is not None and record.before != previous_after:
                _fail(
                    "same_time_state_continuity",
                    "before-state does not match the preceding same-time after-state",
                    record,
                )
            last_after_at_time[key] = record.after
        elif record.state_indices or record.before or record.after:
            _fail(
                "full_state_snapshot_contract",
                f"{record.kind.name} unexpectedly contains state",
                record,
            )


def _check_spikes_and_refractory(
    records: Sequence[TraceRecord],
    models: Sequence[ExecutableModel],
    result: MixedRunResult,
) -> None:
    traced_spikes = tuple(
        (record.t, record.node)
        for record in records
        if record.kind is TraceKind.SPIKE
    )
    result_spikes = tuple((spike.t, spike.node) for spike in result.spikes)
    if traced_spikes != result_spikes:
        _fail(
            "spike_reset_pairing",
            "spike records do not exactly match the returned output spikes",
        )

    confirmed = {
        (record.t, record.node, record.generation)
        for record in records
        if record.kind is TraceKind.PREDICTION_CONFIRMED
    }
    clamped: dict[int, tuple[float, int]] = {}
    for index, record in enumerate(records):
        model = models[record.node]
        if record.kind is TraceKind.SPIKE:
            if record.node in clamped:
                _fail(
                    "refractory_state_machine",
                    f"node {record.node} fired while refractory-clamped",
                    record,
                )
            readout = record.before[model.readout_index]
            autonomous = (record.t, record.node, record.generation) in confirmed
            tolerance = _crossing_threshold_tolerance(model) if autonomous else 0.0
            if readout < model.threshold - tolerance:
                _fail(
                    "spike_reset_pairing",
                    "spike readout is below threshold",
                    record,
                )
            if index + 1 >= len(records):
                _fail("spike_reset_pairing", "spike has no reset record", record)
            reset = records[index + 1]
            if (
                reset.kind is not TraceKind.RESET
                or reset.t != record.t
                or reset.node != record.node
                or reset.generation != record.generation
            ):
                _fail(
                    "spike_reset_pairing",
                    "spike is not immediately followed by its atomic reset",
                    record,
                )
            if not _close(reset.after[model.readout_index], model.reset):
                _fail(
                    "spike_reset_pairing",
                    "reset readout does not match the model reset value",
                    reset,
                )
            if model.refractory > 0.0:
                if index + 2 >= len(records):
                    _fail(
                        "refractory_state_machine",
                        "refractory spike has no clamp-entry record",
                        record,
                    )
                enter = records[index + 2]
                if (
                    enter.kind is not TraceKind.REFRACTORY_ENTER
                    or enter.t != record.t
                    or enter.node != record.node
                ):
                    _fail(
                        "refractory_state_machine",
                        "reset is not immediately followed by refractory entry",
                        record,
                    )
        elif record.kind is TraceKind.RESET:
            if index == 0 or records[index - 1].kind is not TraceKind.SPIKE:
                _fail(
                    "spike_reset_pairing",
                    "reset is not paired with an immediately preceding spike",
                    record,
                )
        elif record.kind is TraceKind.REFRACTORY_ENTER:
            if model.refractory <= 0.0:
                _fail(
                    "refractory_state_machine",
                    "non-refractory model entered a clamp",
                    record,
                )
            if record.node in clamped:
                _fail(
                    "refractory_state_machine",
                    "node entered refractory while already clamped",
                    record,
                )
            expected_release = record.t + model.refractory
            if not _close(record.value, expected_release):
                _fail(
                    "refractory_state_machine",
                    f"release time {record.value} does not match {expected_release}",
                    record,
                )
            clamped[record.node] = (record.value, record.generation)
        elif record.kind is TraceKind.REFRACTORY_RELEASE:
            active = clamped.get(record.node)
            if active is None:
                _fail(
                    "refractory_state_machine",
                    "release has no active refractory interval",
                    record,
                )
            release_time, generation = active
            if not _close(record.t, release_time) or record.generation != generation:
                _fail(
                    "refractory_state_machine",
                    "release time or generation does not match clamp entry",
                    record,
                )
            del clamped[record.node]


def _check_deposits(records: Sequence[TraceRecord]) -> None:
    pending: Counter[tuple[float, int]] = Counter()
    for record in records:
        key = (record.t, record.node)
        if record.kind in (TraceKind.INPUT_SPIKE, TraceKind.DELIVERY):
            pending[key] += 1
        elif record.kind is TraceKind.DEPOSIT_APPLY:
            if pending[key] == 0:
                _fail(
                    "same_time_deposit_causality",
                    "deposit application has no pending input or delivery",
                    record,
                )
            pending[key] = 0
        elif record.kind is TraceKind.SPIKE and pending[key] != 0:
            _fail(
                "same_time_deposit_causality",
                "node fired before its pending deposits were applied",
                record,
            )
    remaining = sum(pending.values())
    if remaining:
        _fail(
            "same_time_deposit_causality",
            f"{remaining} input or delivery records were never applied",
        )


def _check_deliveries(
    records: Sequence[TraceRecord],
    edges: Sequence[MixedEdge],
    result: MixedRunResult,
    t_end: float,
) -> None:
    plastic_edges = {state.edge for state in result.plasticity}
    actual: dict[int, list[TraceRecord]] = defaultdict(list)
    for record in records:
        if record.kind is not TraceKind.DELIVERY:
            continue
        assert record.subject is not None
        if not 0 <= record.subject < len(edges):
            _fail(
                "delivery_conservation",
                f"delivery references nonexistent edge {record.subject}",
                record,
            )
        edge = edges[record.subject]
        if record.node != edge.post or (
            record.subject not in plastic_edges
            and not _close(record.value, edge.weight)
        ):
            _fail(
                "delivery_conservation",
                "delivery target or weight disagrees with its edge",
                record,
            )
        actual[record.subject].append(record)

    expected: dict[int, list[float]] = defaultdict(list)
    outgoing: dict[int, list[tuple[int, MixedEdge]]] = defaultdict(list)
    for edge_index, edge in enumerate(edges):
        outgoing[edge.pre].append((edge_index, edge))
    for spike in result.spikes:
        for edge_index, edge in outgoing[spike.node]:
            delivery_time = spike.t + edge.delay
            if delivery_time <= t_end:
                expected[edge_index].append(delivery_time)

    for edge_index in set(actual) | set(expected):
        observed = sorted(record.t for record in actual[edge_index])
        wanted = sorted(expected[edge_index])
        if len(observed) != len(wanted) or any(
            not _close(left, right) for left, right in zip(observed, wanted)
        ):
            _fail(
                "delivery_conservation",
                f"edge {edge_index} expected deliveries {wanted}, observed {observed}",
            )


def _check_predictions(records: Sequence[TraceRecord], result: MixedRunResult) -> None:
    spikes = Counter(
        (record.t, record.node, record.generation)
        for record in records
        if record.kind is TraceKind.SPIKE
    )
    confirmed = Counter(
        (record.t, record.node, record.generation)
        for record in records
        if record.kind is TraceKind.PREDICTION_CONFIRMED
    )
    stale = Counter(
        (record.t, record.node, record.generation)
        for record in records
        if record.kind is TraceKind.STALE_PREDICTION
    )
    if any(count > spikes[key] for key, count in confirmed.items()):
        _fail(
            "prediction_generation_consistency",
            "a confirmed prediction lacks a same-generation spike",
        )
    if set(stale) & (set(confirmed) | set(spikes)):
        _fail(
            "prediction_generation_consistency",
            "a stale prediction was confirmed or fired at the same generation",
        )
    if sum(confirmed.values()) != result.stats.autonomous_spikes_confirmed:
        _fail(
            "prediction_generation_consistency",
            "confirmed-prediction trace count disagrees with runtime statistics",
        )


def _check_accounting(
    records: Sequence[TraceRecord],
    result: MixedRunResult,
    models: Sequence[ExecutableModel],
    *,
    expected_input_count: int | None,
    expected_drive_count: int | None,
) -> None:
    counts = Counter(record.kind for record in records)
    stats = result.stats
    comparisons = (
        (TraceKind.INPUT_SPIKE, stats.input_spikes_processed),
        (TraceKind.DELIVERY, stats.deliveries_processed),
        (TraceKind.DRIVE_UPDATE, stats.drive_updates_processed),
        (TraceKind.REFRACTORY_RELEASE, stats.refractory_releases_processed),
        (TraceKind.STALE_PREDICTION, stats.stale_predictions),
        (TraceKind.PREDICTION_CONFIRMED, stats.autonomous_spikes_confirmed),
        (TraceKind.SPIKE, stats.output_spikes),
    )
    for kind, expected in comparisons:
        if counts[kind] != expected:
            _fail(
                "runtime_accounting_from_trace",
                f"{kind.name} count {counts[kind]} disagrees with statistic {expected}",
            )
    popped = (
        sum(counts[kind] for kind, _ in comparisons[:6])
        + counts[TraceKind.MODULATION]
    )
    if popped != stats.events_popped:
        _fail(
            "runtime_accounting_from_trace",
            f"trace accounts for {popped} popped events, runtime reports {stats.events_popped}",
        )
    if counts[TraceKind.RESET] != counts[TraceKind.SPIKE]:
        _fail(
            "runtime_accounting_from_trace",
            "reset and spike trace counts differ",
        )
    expected_enters = sum(
        1
        for record in records
        if record.kind is TraceKind.SPIKE and models[record.node].refractory > 0.0
    )
    if counts[TraceKind.REFRACTORY_ENTER] != expected_enters:
        _fail(
            "runtime_accounting_from_trace",
            "refractory-entry count disagrees with refractory spikes",
        )
    if stats.deliveries_scheduled != stats.deliveries_processed:
        _fail(
            "runtime_accounting_from_trace",
            "not every in-horizon scheduled delivery was processed",
        )
    if expected_input_count is not None and counts[TraceKind.INPUT_SPIKE] != expected_input_count:
        _fail(
            "runtime_accounting_from_trace",
            "processed input count disagrees with the supplied in-horizon inputs",
        )
    if expected_drive_count is not None and counts[TraceKind.DRIVE_UPDATE] != expected_drive_count:
        _fail(
            "runtime_accounting_from_trace",
            "processed drive count disagrees with the supplied in-horizon updates",
        )
    input_subjects = [
        record.subject
        for record in records
        if record.kind is TraceKind.INPUT_SPIKE
    ]
    if len(input_subjects) != len(set(input_subjects)):
        _fail(
            "runtime_accounting_from_trace",
            "an input event was processed more than once",
        )


def _check_final_states(
    records: Sequence[TraceRecord],
    result: MixedRunResult,
    models: Sequence[ExecutableModel],
    t_end: float,
) -> None:
    final = [record for record in records if record.kind is TraceKind.FINAL_STATE]
    if len(final) != len(models) or [record.node for record in final] != list(
        range(len(models))
    ):
        _fail(
            "final_state_agreement",
            "final-state records are not exactly one per node in canonical order",
        )
    if len(result.states) != len(models):
        _fail(
            "final_state_agreement",
            "returned state count disagrees with the model count",
        )
    for record, state in zip(final, result.states):
        if (
            record.t != t_end
            or record.before != record.after
            or record.after != state.values
            or state.t_last != t_end
        ):
            _fail(
                "final_state_agreement",
                "final trace snapshot disagrees with the returned state",
                record,
            )


class _IncrementalTraceAuditor:
    """Single-pass scheduler audit with bounded temporal working state."""

    def __init__(
        self,
        result: MixedRunResult,
        models: Sequence[ExecutableModel],
        edges: Sequence[MixedEdge],
        t_end: float,
        *,
        polarities: Sequence[NeuronPolarity] | None,
        expected_input_count: int | None,
        expected_drive_count: int | None,
    ) -> None:
        if not models:
            raise ValueError("trace audit requires at least one model")
        if not math.isfinite(t_end) or t_end < 0.0:
            raise ValueError("trace audit t_end must be finite and nonnegative")
        if polarities is None:
            normalized_polarities = (NeuronPolarity.EXCITATORY,) * len(models)
        else:
            normalized_polarities = tuple(polarities)
            if len(normalized_polarities) != len(models):
                raise ValueError("trace audit requires one polarity per model")
            if any(
                not isinstance(polarity, NeuronPolarity)
                for polarity in normalized_polarities
            ):
                raise ValueError("invalid neuron polarity in trace audit")
        self.result = result
        self.models = tuple(models)
        self.edges = tuple(
            replace(
                edge,
                weight=(
                    edge.weight
                    * edge.deposit_scale
                    * normalized_polarities[edge.pre].sign
                ),
                deposit_scale=1.0,
            )
            for edge in edges
        )
        self.plastic_edges = {state.edge for state in result.plasticity}
        self.t_end = float(t_end)
        self.expected_input_count = expected_input_count
        self.expected_drive_count = expected_drive_count
        self.counts: Counter[TraceKind] = Counter()
        self.record_count = 0
        self.previous_t = -math.inf
        self.current_t: float | None = None
        self.final_started = False
        self.last_after_by_node: dict[int, tuple[float, ...]] = {}
        self.pending_deposits: Counter[int] = Counter()
        self.pending_confirmed: Counter[tuple[int, int]] = Counter()
        self.stale_at_time: set[tuple[int, int]] = set()
        self.spikes_at_time: set[tuple[int, int]] = set()
        self.expected_deliveries: Counter[int] = Counter()
        self.observed_deliveries: Counter[int] = Counter()
        self.future_deliveries: list[tuple[float, int]] = []
        self.clamped: dict[int, tuple[float, int]] = {}
        self.input_subjects: set[int] = set()
        self.previous: TraceRecord | None = None
        self.before_previous: TraceRecord | None = None
        self.result_spike_index = 0
        self.final_index = 0
        self.expected_refractory_enters = 0
        outgoing: dict[int, list[tuple[int, MixedEdge]]] = defaultdict(list)
        for edge_index, edge in enumerate(self.edges):
            outgoing[edge.pre].append((edge_index, edge))
        self.outgoing = dict(outgoing)

    def _finish_time(self) -> None:
        if self.current_t is None:
            return
        pending = sum(self.pending_deposits.values())
        if pending:
            _fail(
                "same_time_deposit_causality",
                f"{pending} input or delivery records were never applied",
            )
        if self.pending_confirmed:
            _fail(
                "prediction_generation_consistency",
                "a confirmed prediction lacks a same-generation spike",
            )
        if self.expected_deliveries != self.observed_deliveries:
            edge_indices = sorted(
                set(self.expected_deliveries) | set(self.observed_deliveries)
            )
            detail = ", ".join(
                f"edge {edge}: expected {self.expected_deliveries[edge]}, "
                f"observed {self.observed_deliveries[edge]}"
                for edge in edge_indices
                if self.expected_deliveries[edge] != self.observed_deliveries[edge]
            )
            _fail(
                "delivery_conservation",
                f"delivery counts disagree at t={self.current_t}: {detail}",
            )

    def _begin_time(self, t: float) -> None:
        self.current_t = t
        self.last_after_by_node.clear()
        self.pending_deposits.clear()
        self.pending_confirmed.clear()
        self.stale_at_time.clear()
        self.spikes_at_time.clear()
        self.expected_deliveries.clear()
        self.observed_deliveries.clear()
        while self.future_deliveries:
            delivery_time, edge_index = self.future_deliveries[0]
            if _close(delivery_time, t):
                heapq.heappop(self.future_deliveries)
                self.expected_deliveries[edge_index] += 1
            elif delivery_time < t:
                _fail(
                    "delivery_conservation",
                    f"edge {edge_index} delivery at {delivery_time} was not observed",
                )
            else:
                break

    def _check_immediate_pair(self, record: TraceRecord) -> None:
        previous = self.previous
        before_previous = self.before_previous
        if previous is not None and previous.kind is TraceKind.SPIKE:
            if (
                record.kind is not TraceKind.RESET
                or record.t != previous.t
                or record.node != previous.node
                or record.generation != previous.generation
            ):
                _fail(
                    "spike_reset_pairing",
                    "spike is not immediately followed by its atomic reset",
                    previous,
                )
        if (
            previous is not None
            and previous.kind is TraceKind.RESET
            and before_previous is not None
            and before_previous.kind is TraceKind.SPIKE
            and self.models[before_previous.node].refractory > 0.0
            and (
                record.kind is not TraceKind.REFRACTORY_ENTER
                or record.t != before_previous.t
                or record.node != before_previous.node
            )
        ):
            _fail(
                "refractory_state_machine",
                "reset is not immediately followed by refractory entry",
                before_previous,
            )

    def _check_record_contract(self, record: TraceRecord) -> None:
        if record.sequence != self.record_count:
            _fail(
                "trace_sequence_and_time",
                f"expected sequence {self.record_count}, got {record.sequence}",
                record,
            )
        if (
            not math.isfinite(record.t)
            or record.t < 0.0
            or record.t > self.t_end
            or record.t < self.previous_t
        ):
            _fail(
                "trace_sequence_and_time",
                f"invalid or nonchronological time {record.t!r}",
                record,
            )
        if self.current_t != record.t:
            self._finish_time()
            self._begin_time(record.t)
        self.previous_t = record.t
        if not 0 <= record.node < len(self.models):
            _fail(
                "trace_phase_and_payload_contract",
                f"node {record.node} is outside [0, {len(self.models)})",
                record,
            )
        if record.phase is not _PHASE_BY_KIND[record.kind]:
            _fail(
                "trace_phase_and_payload_contract",
                f"{record.kind.name} has phase {record.phase.name}",
                record,
            )
        if not math.isfinite(record.value):
            _fail(
                "trace_phase_and_payload_contract",
                "record value is not finite",
                record,
            )
        if (record.kind in _SUBJECT_KINDS) != (record.subject is not None):
            _fail(
                "trace_phase_and_payload_contract",
                f"{record.kind.name} has an invalid subject payload",
                record,
            )
        if self.final_started and record.kind is not TraceKind.FINAL_STATE:
            _fail(
                "trace_phase_and_payload_contract",
                "a non-final record follows final-state emission",
                record,
            )
        self.final_started = self.final_started or record.kind is TraceKind.FINAL_STATE

        expected_count = _state_count(self.models[record.node])
        if record.kind in _STATEFUL_KINDS:
            if (
                record.state_indices != tuple(range(expected_count))
                or len(record.before) != expected_count
                or len(record.after) != expected_count
            ):
                _fail(
                    "full_state_snapshot_contract",
                    f"{record.kind.name} does not contain all {expected_count} states",
                    record,
                )
            if any(not math.isfinite(value) for value in (*record.before, *record.after)):
                _fail(
                    "full_state_snapshot_contract",
                    "state snapshot contains a non-finite value",
                    record,
                )
            previous_after = self.last_after_by_node.get(record.node)
            if previous_after is not None and record.before != previous_after:
                _fail(
                    "same_time_state_continuity",
                    "before-state does not match the preceding same-time after-state",
                    record,
                )
            self.last_after_by_node[record.node] = record.after
        elif record.state_indices or record.before or record.after:
            _fail(
                "full_state_snapshot_contract",
                f"{record.kind.name} unexpectedly contains state",
                record,
            )

    def _record_delivery(self, record: TraceRecord) -> None:
        assert record.subject is not None
        if not 0 <= record.subject < len(self.edges):
            _fail(
                "delivery_conservation",
                f"delivery references nonexistent edge {record.subject}",
                record,
            )
        edge = self.edges[record.subject]
        if record.node != edge.post or (
            record.subject not in self.plastic_edges
            and not _close(record.value, edge.weight)
        ):
            _fail(
                "delivery_conservation",
                "delivery target or weight disagrees with its edge",
                record,
            )
        self.observed_deliveries[record.subject] += 1

    def _record_prediction(self, record: TraceRecord) -> None:
        key = (record.node, record.generation)
        if record.kind is TraceKind.STALE_PREDICTION:
            if key in self.pending_confirmed or key in self.spikes_at_time:
                _fail(
                    "prediction_generation_consistency",
                    "a stale prediction was confirmed or fired at the same generation",
                    record,
                )
            self.stale_at_time.add(key)
        elif record.kind is TraceKind.PREDICTION_CONFIRMED:
            if key in self.stale_at_time:
                _fail(
                    "prediction_generation_consistency",
                    "a stale prediction was confirmed or fired at the same generation",
                    record,
                )
            self.pending_confirmed[key] += 1

    def _record_spike(self, record: TraceRecord) -> None:
        model = self.models[record.node]
        if record.node in self.clamped:
            _fail(
                "refractory_state_machine",
                f"node {record.node} fired while refractory-clamped",
                record,
            )
        key = (record.node, record.generation)
        if key in self.stale_at_time:
            _fail(
                "prediction_generation_consistency",
                "a stale prediction was confirmed or fired at the same generation",
                record,
            )
        autonomous = self.pending_confirmed[key] > 0
        if autonomous:
            self.pending_confirmed[key] -= 1
            if self.pending_confirmed[key] == 0:
                del self.pending_confirmed[key]
        self.spikes_at_time.add(key)
        readout = record.before[model.readout_index]
        tolerance = _crossing_threshold_tolerance(model) if autonomous else 0.0
        if readout < model.threshold - tolerance:
            _fail(
                "spike_reset_pairing",
                "spike readout is below threshold",
                record,
            )
        if self.pending_deposits[record.node] != 0:
            _fail(
                "same_time_deposit_causality",
                "node fired before its pending deposits were applied",
                record,
            )
        if self.result_spike_index >= len(self.result.spikes):
            _fail(
                "spike_reset_pairing",
                "trace contains more spikes than the returned output",
                record,
            )
        expected = self.result.spikes[self.result_spike_index]
        if (record.t, record.node) != (expected.t, expected.node):
            _fail(
                "spike_reset_pairing",
                "spike record does not match the returned output spike",
                record,
            )
        self.result_spike_index += 1
        if model.refractory > 0.0:
            self.expected_refractory_enters += 1
        for edge_index, edge in self.outgoing.get(record.node, ()):
            delivery_time = record.t + edge.delay
            if delivery_time <= self.t_end:
                if delivery_time == record.t:
                    self.expected_deliveries[edge_index] += 1
                else:
                    heapq.heappush(
                        self.future_deliveries, (delivery_time, edge_index)
                    )

    def _record_reset(self, record: TraceRecord) -> None:
        previous = self.previous
        if previous is None or previous.kind is not TraceKind.SPIKE:
            _fail(
                "spike_reset_pairing",
                "reset is not paired with an immediately preceding spike",
                record,
            )
        model = self.models[record.node]
        if not _close(record.after[model.readout_index], model.reset):
            _fail(
                "spike_reset_pairing",
                "reset readout does not match the model reset value",
                record,
            )

    def _record_refractory(self, record: TraceRecord) -> None:
        model = self.models[record.node]
        if record.kind is TraceKind.REFRACTORY_ENTER:
            if model.refractory <= 0.0:
                _fail(
                    "refractory_state_machine",
                    "non-refractory model entered a clamp",
                    record,
                )
            if record.node in self.clamped:
                _fail(
                    "refractory_state_machine",
                    "node entered refractory while already clamped",
                    record,
                )
            expected_release = record.t + model.refractory
            if not _close(record.value, expected_release):
                _fail(
                    "refractory_state_machine",
                    f"release time {record.value} does not match {expected_release}",
                    record,
                )
            self.clamped[record.node] = (record.value, record.generation)
        else:
            active = self.clamped.get(record.node)
            if active is None:
                _fail(
                    "refractory_state_machine",
                    "release has no active refractory interval",
                    record,
                )
            release_time, generation = active
            if not _close(record.t, release_time) or record.generation != generation:
                _fail(
                    "refractory_state_machine",
                    "release time or generation does not match clamp entry",
                    record,
                )
            del self.clamped[record.node]

    def _record_final(self, record: TraceRecord) -> None:
        if self.final_index >= len(self.models) or record.node != self.final_index:
            _fail(
                "final_state_agreement",
                "final-state records are not exactly one per node in canonical order",
                record,
            )
        if len(self.result.states) != len(self.models):
            _fail(
                "final_state_agreement",
                "returned state count disagrees with the model count",
            )
        state = self.result.states[self.final_index]
        if (
            record.t != self.t_end
            or record.before != record.after
            or record.after != state.values
            or state.t_last != self.t_end
        ):
            _fail(
                "final_state_agreement",
                "final trace snapshot disagrees with the returned state",
                record,
            )
        self.final_index += 1

    def feed(self, record: TraceRecord) -> None:
        """Validate and consume one chronological trace record."""

        if not isinstance(record, TraceRecord):
            raise TypeError("trace records must be TraceRecord values")
        self._check_record_contract(record)
        self._check_immediate_pair(record)
        self.counts[record.kind] += 1

        if record.kind in (TraceKind.INPUT_SPIKE, TraceKind.DELIVERY):
            self.pending_deposits[record.node] += 1
        if record.kind is TraceKind.INPUT_SPIKE:
            assert record.subject is not None
            if record.subject in self.input_subjects:
                _fail(
                    "runtime_accounting_from_trace",
                    "an input event was processed more than once",
                    record,
                )
            self.input_subjects.add(record.subject)
        elif record.kind is TraceKind.DELIVERY:
            self._record_delivery(record)
        elif record.kind in (
            TraceKind.STALE_PREDICTION,
            TraceKind.PREDICTION_CONFIRMED,
        ):
            self._record_prediction(record)
        elif record.kind is TraceKind.DEPOSIT_APPLY:
            if self.pending_deposits[record.node] == 0:
                _fail(
                    "same_time_deposit_causality",
                    "deposit application has no pending input or delivery",
                    record,
                )
            self.pending_deposits[record.node] = 0
        elif record.kind is TraceKind.SPIKE:
            self._record_spike(record)
        elif record.kind is TraceKind.RESET:
            self._record_reset(record)
        elif record.kind in (
            TraceKind.REFRACTORY_ENTER,
            TraceKind.REFRACTORY_RELEASE,
        ):
            self._record_refractory(record)
        elif record.kind is TraceKind.FINAL_STATE:
            self._record_final(record)

        self.before_previous = self.previous
        self.previous = record
        self.record_count += 1

    def finish(self) -> TraceAuditReport:
        """Finalize accounting checks and return the audit report."""

        if self.record_count == 0:
            _fail("trace_sequence_and_time", "the causal trace is empty")
        self._finish_time()
        if self.previous is not None and self.previous.kind is TraceKind.SPIKE:
            _fail(
                "spike_reset_pairing",
                "spike has no reset record",
                self.previous,
            )
        if self.future_deliveries:
            delivery_time, edge_index = self.future_deliveries[0]
            _fail(
                "delivery_conservation",
                f"edge {edge_index} delivery at {delivery_time} was not observed",
            )
        if self.result_spike_index != len(self.result.spikes):
            _fail(
                "spike_reset_pairing",
                "spike records do not exactly match the returned output spikes",
            )
        if self.final_index != len(self.models):
            _fail(
                "final_state_agreement",
                "final-state records are not exactly one per node in canonical order",
            )
        stats = self.result.stats
        comparisons = (
            (TraceKind.INPUT_SPIKE, stats.input_spikes_processed),
            (TraceKind.DELIVERY, stats.deliveries_processed),
            (TraceKind.DRIVE_UPDATE, stats.drive_updates_processed),
            (TraceKind.REFRACTORY_RELEASE, stats.refractory_releases_processed),
            (TraceKind.STALE_PREDICTION, stats.stale_predictions),
            (TraceKind.PREDICTION_CONFIRMED, stats.autonomous_spikes_confirmed),
            (TraceKind.SPIKE, stats.output_spikes),
        )
        for kind, expected in comparisons:
            if self.counts[kind] != expected:
                _fail(
                    "runtime_accounting_from_trace",
                    f"{kind.name} count {self.counts[kind]} "
                    f"disagrees with statistic {expected}",
                )
        popped = (
            sum(self.counts[kind] for kind, _ in comparisons[:6])
            + self.counts[TraceKind.MODULATION]
        )
        if popped != stats.events_popped:
            _fail(
                "runtime_accounting_from_trace",
                f"trace accounts for {popped} popped events, "
                f"runtime reports {stats.events_popped}",
            )
        if self.counts[TraceKind.RESET] != self.counts[TraceKind.SPIKE]:
            _fail(
                "runtime_accounting_from_trace",
                "reset and spike trace counts differ",
            )
        if self.counts[TraceKind.REFRACTORY_ENTER] != self.expected_refractory_enters:
            _fail(
                "runtime_accounting_from_trace",
                "refractory-entry count disagrees with refractory spikes",
            )
        if stats.deliveries_scheduled != stats.deliveries_processed:
            _fail(
                "runtime_accounting_from_trace",
                "not every in-horizon scheduled delivery was processed",
            )
        if (
            self.expected_input_count is not None
            and self.counts[TraceKind.INPUT_SPIKE] != self.expected_input_count
        ):
            _fail(
                "runtime_accounting_from_trace",
                "processed input count disagrees with the supplied in-horizon inputs",
            )
        if (
            self.expected_drive_count is not None
            and self.counts[TraceKind.DRIVE_UPDATE] != self.expected_drive_count
        ):
            _fail(
                "runtime_accounting_from_trace",
                "processed drive count disagrees with the supplied in-horizon updates",
            )
        return TraceAuditReport(
            record_count=self.record_count,
            input_records=self.counts[TraceKind.INPUT_SPIKE],
            delivery_records=self.counts[TraceKind.DELIVERY],
            drive_records=self.counts[TraceKind.DRIVE_UPDATE],
            modulation_records=self.counts[TraceKind.MODULATION],
            spike_records=self.counts[TraceKind.SPIKE],
            stale_prediction_records=self.counts[TraceKind.STALE_PREDICTION],
            confirmed_prediction_records=self.counts[TraceKind.PREDICTION_CONFIRMED],
            final_state_records=self.counts[TraceKind.FINAL_STATE],
            checks=_CHECKS,
        )


def audit_causal_records(
    records: Iterable[TraceRecord],
    result: MixedRunResult,
    *,
    models: Sequence[ExecutableModel],
    edges: Sequence[MixedEdge],
    t_end: float,
    polarities: Sequence[NeuronPolarity] | None = None,
    expected_input_count: int | None = None,
    expected_drive_count: int | None = None,
) -> TraceAuditReport:
    """Audit a complete causal-record stream in one pass."""

    auditor = _IncrementalTraceAuditor(
        result,
        models,
        edges,
        t_end,
        polarities=polarities,
        expected_input_count=expected_input_count,
        expected_drive_count=expected_drive_count,
    )
    for record in records:
        auditor.feed(record)
    return auditor.finish()


def audit_causal_trace(
    result: MixedRunResult,
    *,
    models: Sequence[ExecutableModel],
    edges: Sequence[MixedEdge],
    t_end: float,
    polarities: Sequence[NeuronPolarity] | None = None,
    expected_input_count: int | None = None,
    expected_drive_count: int | None = None,
) -> TraceAuditReport:
    """Audit one complete, all-node, all-kind, full-state causal trace.

    The caller must record every :class:`~lacuna.ffi.TraceKind`, every node, and
    all state indices.  Filtered traces are useful for inspection but cannot
    establish global scheduler invariants and are rejected explicitly here.
    """

    return audit_causal_records(
        result.trace,
        result,
        models=models,
        edges=edges,
        t_end=t_end,
        polarities=polarities,
        expected_input_count=expected_input_count,
        expected_drive_count=expected_drive_count,
    )


def audit_persisted_trace(
    reader: "TraceArtifactReader",
    result: MixedRunResult | "GraphRunResult",
    *,
    resolved: "ResolvedGraph",
    t_end: float,
    expected_input_count: int | None = None,
    expected_drive_count: int | None = None,
) -> TraceAuditReport:
    """Audit a complete persisted trace without materializing its record stream."""

    from .graph import GraphRunResult, ResolvedGraph
    from .tracefile import TraceArtifactIdentityError, TraceArtifactReader

    if not isinstance(reader, TraceArtifactReader):
        raise TypeError("reader must be a TraceArtifactReader")
    if not isinstance(resolved, ResolvedGraph):
        raise TypeError("resolved must be a ResolvedGraph")
    if reader.closed:
        raise ValueError("trace artifact reader is closed")
    if not reader.complete:
        _fail(
            _PERSISTED_CHECK,
            "incremental audit requires a complete trace artifact",
        )
    core_result = result.core if isinstance(result, GraphRunResult) else result
    if not isinstance(core_result, MixedRunResult):
        raise TypeError("result must be a MixedRunResult or GraphRunResult")

    metadata = reader.metadata
    graph_hash = hashlib.sha256(
        resolved.graph.to_text().encode("utf-8")
    ).hexdigest()
    if metadata.graph_sha256 != graph_hash:
        raise TraceArtifactIdentityError(
            "trace artifact graph hash does not match the resolved graph"
        )
    if metadata.time_unit != resolved.graph.time_unit:
        raise TraceArtifactIdentityError(
            "trace artifact time unit does not match the resolved graph"
        )
    if tuple(item.node for item in metadata.nodes) != resolved.node_ids:
        raise TraceArtifactIdentityError(
            "trace artifact node identities do not match the resolved graph"
        )
    for item, model in zip(metadata.nodes, resolved.models):
        names = (
            (model.state_name,)
            if isinstance(model, ResolvedScalarLIF)
            else tuple(model.state_names)
        )
        if (
            item.state_names != names
            or item.model_hash != model.model_hash
            or item.resolution_key != model.resolution_key
        ):
            raise TraceArtifactIdentityError(
                f"trace artifact model identity does not match graph node {item.node}"
            )

    all_nodes = set(resolved.node_ids)
    if (
        metadata.recorded_nodes is not None
        and set(metadata.recorded_nodes) != all_nodes
    ):
        _fail(_PERSISTED_CHECK, "audit requires every graph node to be recorded")
    if set(metadata.recorded_kinds) != {kind.name for kind in TraceKind}:
        _fail(_PERSISTED_CHECK, "audit requires every causal trace kind")
    if not metadata.capture_state:
        _fail(_PERSISTED_CHECK, "audit requires state capture")
    if metadata.state_indices is not None:
        selected = tuple(metadata.state_indices)
        for model in resolved.models:
            if selected != tuple(range(_state_count(model))):
                _fail(
                    _PERSISTED_CHECK,
                    "audit requires every local state variable",
                )

    node_index = {
        public_node: index for index, public_node in enumerate(resolved.node_ids)
    }

    def normalized_records() -> Iterable[TraceRecord]:
        """Map persisted graph identifiers back to compiled node indices."""

        for record in reader.iter_records():
            try:
                internal_node = node_index[record.node]
            except KeyError:
                _fail(
                    _PERSISTED_CHECK,
                    f"record references unknown graph node {record.node}",
                    record,
                )
            yield replace(record, node=internal_node, state_names=())

    report = audit_causal_records(
        normalized_records(),
        core_result,
        models=resolved.models,
        edges=resolved.edges,
        t_end=t_end,
        polarities=resolved.polarities,
        expected_input_count=expected_input_count,
        expected_drive_count=expected_drive_count,
    )
    if report.record_count != reader.record_count:
        _fail(
            _PERSISTED_CHECK,
            f"audited {report.record_count} records but artifact indexes "
            f"{reader.record_count}",
        )
    return replace(report, checks=(_PERSISTED_CHECK,) + report.checks)
