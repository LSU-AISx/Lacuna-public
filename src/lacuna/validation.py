"""Generated whole-network validation corpus and scaling campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence

from .errors import CoreError
from .ffi import CoreEvaluator, RecordingConfig
from .graph import (
    DriveInput,
    Graph,
    GraphEdge,
    GraphModel,
    GraphNode,
    GraphSynapse,
    InputMode,
    InputPort,
    ResolvedGraph,
    SpikeInput,
)
from .tracefile import (
    TraceArtifactMetadata,
    TraceArtifactReader,
    TraceArtifactWriter,
)

VALIDATION_LIF = """
neuron ValidationLIF {
    params {
        tau_m : positive = 10.0
        v_rest = -65.0
        drive = 0.0
        v_th = -50.0
        v_reset = -65.0
    }
    state { v : membrane }
    dynamics { dv/dt = -(v - v_rest)/tau_m + drive/tau_m }
    threshold { v > v_th }
    reset { v <- v_reset }
    refractory { 2.0 }
}
"""

VALIDATION_ALPHA_LIF = """
neuron ValidationAlphaLIF {
    params {
        tau_m : positive = 10.0
        v_rest = -65.0
        drive = 0.0
        v_th = -50.0
        v_reset = -65.0
    }
    state {
        v : membrane
        i_exc : receptor
    }
    dynamics { dv/dt = -(v - v_rest)/tau_m + drive/tau_m + i_exc }
    threshold { v > v_th }
    reset { v <- v_reset }
    refractory { 2.0 }
}
"""

VALIDATION_ALPHA_SYNAPSE = """
synapse ValidationAlpha {
    params { tau_s : positive = 5.0 }
    state { s; z }
    dynamics {
        ds/dt = -s/tau_s + z
        dz/dt = -z/tau_s
    }
    on_spike { z <- z + w/tau_s^2 }
    output { current = s }
}
"""

VALIDATION_EXP_SYNAPSE = """
synapse ValidationExp {
    params { tau_s : positive = 5.0 }
    state { s }
    dynamics { ds/dt = -s/tau_s }
    on_spike { s <- s + w }
    output { current = s }
}
"""

VALIDATION_ADAPTIVE_LIF = """
neuron ValidationAdaptiveLIF {
    params {
        tau_m : positive = 10.0
        tau_w : positive = 50.0
        v_rest = -65.0
        drive = 25.0
        v_th = -50.0
        v_reset = -65.0
        beta = 2.0
    }
    state {
        v : membrane
        w : adaptation
    }
    dynamics {
        dv/dt = -(v - v_rest)/tau_m - w + drive/tau_m
        dw/dt = -w/tau_w
    }
    threshold { v > v_th }
    reset {
        v <- v_reset
        w <- w + beta
    }
    refractory { 2.0 }
}
"""


@dataclass(frozen=True)
class NetworkCase:
    """Deterministic graph workload and its expected accounting bounds."""

    name: str
    graph: Graph
    spike_inputs: tuple[SpikeInput, ...]
    drive_inputs: tuple[DriveInput, ...]
    t_end: float
    expected_min_spikes: int


@dataclass(frozen=True)
class CaseReport:
    """Correctness, replay, timing, and resource results for one case."""

    name: str
    graph_sha256: str
    nodes: int
    edges: int
    input_spikes: int
    drive_updates: int
    output_spikes: int
    events_popped: int
    deliveries: int
    stale_predictions: int
    peak_queue_occupancy: int
    queue_headroom_at_test_capacity: int
    max_same_time_cascade_depth: int
    median_seconds: float
    events_per_second: float
    deterministic_replays: int
    trace_records: int
    trace_audit_checks: tuple[str, ...]
    invariants_checked: tuple[str, ...]


def _base_graph(
    node_count: int,
    edges: Sequence[GraphEdge],
    input_ports: Sequence[InputPort],
    *,
    bindings: Sequence[dict[str, float]] | None = None,
) -> Graph:
    node_bindings = bindings or tuple({"drive": 0.0} for _ in range(node_count))
    return Graph(
        models=(GraphModel("lif", VALIDATION_LIF),),
        nodes=tuple(
            GraphNode(index, "lif", -65.0, node_bindings[index])
            for index in range(node_count)
        ),
        edges=tuple(edges),
        input_ports=tuple(input_ports),
    )


def chain_case(node_count: int) -> NetworkCase:
    """Build a feed-forward chain driven by one external spike."""

    edges = tuple(
        GraphEdge(index, index, index + 1, 20.0, 0.25)
        for index in range(node_count - 1)
    )
    graph = _base_graph(
        node_count,
        edges,
        (InputPort("start", 0, InputMode.SPIKE),),
    )
    return NetworkCase(
        f"feedforward_chain_{node_count}",
        graph,
        (SpikeInput(1.0, "start", 20.0),),
        (),
        2.0 + node_count * 0.25,
        node_count,
    )


def fanout_fanin_case(middle_count: int) -> NetworkCase:
    """Build equal-delay fan-out followed by convergent fan-in."""

    source = 0
    sink = middle_count + 1
    edges: list[GraphEdge] = []
    edge_id = 0
    for middle in range(1, sink):
        edges.append(GraphEdge(edge_id, source, middle, 20.0, 0.5))
        edge_id += 1
    contribution = 18.0 / middle_count
    for middle in range(1, sink):
        edges.append(GraphEdge(edge_id, middle, sink, contribution, 0.5))
        edge_id += 1
    graph = _base_graph(
        sink + 1,
        edges,
        (InputPort("start", source, InputMode.SPIKE),),
    )
    return NetworkCase(
        f"fanout_fanin_{middle_count}",
        graph,
        (SpikeInput(1.0, "start", 20.0),),
        (),
        5.0,
        sink + 1,
    )


def recurrent_ring_case(node_count: int) -> NetworkCase:
    """Build a recurrent ring with one outgoing edge per node."""

    edges = tuple(
        GraphEdge(index, index, (index + 1) % node_count, 20.0, 0.5)
        for index in range(node_count)
    )
    graph = _base_graph(
        node_count,
        edges,
        (InputPort("start", 0, InputMode.SPIKE),),
    )
    return NetworkCase(
        f"recurrent_ring_{node_count}",
        graph,
        (SpikeInput(1.0, "start", 20.0),),
        (),
        50.0,
        node_count * 2,
    )


def bipartite_burst_case(source_count: int, target_count: int) -> NetworkCase:
    """Build a dense bipartite projection driven by a source burst."""

    node_count = source_count + target_count
    edges: list[GraphEdge] = []
    edge_id = 0
    contribution = 18.0 / source_count
    for source in range(source_count):
        for target in range(source_count, node_count):
            edges.append(GraphEdge(edge_id, source, target, contribution, 0.5))
            edge_id += 1
    ports = tuple(InputPort(f"source_{index}", index, InputMode.SPIKE) for index in range(source_count))
    inputs = tuple(SpikeInput(1.0, f"source_{index}", 20.0) for index in range(source_count))
    return NetworkCase(
        f"bipartite_burst_{node_count}",
        _base_graph(node_count, edges, ports),
        inputs,
        (),
        5.0,
        node_count,
    )


def sparse_recurrent_case(node_count: int, out_degree: int, seed: int = 1729) -> NetworkCase:
    """Build a seeded sparse recurrent graph with fixed out-degree."""

    rng = random.Random(seed)
    edges: list[GraphEdge] = []
    edge_id = 0
    for source in range(node_count):
        candidates = [node for node in range(node_count) if node != source]
        for target in rng.sample(candidates, out_degree):
            delay = rng.choice((0.25, 0.5, 0.75, 1.0))
            edges.append(GraphEdge(edge_id, source, target, 8.0, delay))
            edge_id += 1
    seed_count = max(1, node_count // 4)
    ports = tuple(InputPort(f"seed_{index}", index, InputMode.SPIKE) for index in range(seed_count))
    inputs = tuple(SpikeInput(1.0, f"seed_{index}", 20.0) for index in range(seed_count))
    return NetworkCase(
        f"sparse_recurrent_{node_count}_d{out_degree}",
        _base_graph(node_count, edges, ports),
        inputs,
        (),
        30.0,
        seed_count,
    )


def driven_population_case(node_count: int) -> NetworkCase:
    """Build independent tonic neurons without recurrent edges."""

    bindings = tuple({"drive": 20.0} for _ in range(node_count))
    return NetworkCase(
        f"driven_population_{node_count}",
        _base_graph(node_count, (), (), bindings=bindings),
        (),
        (),
        50.0,
        node_count * 2,
    )


def drive_boundary_case(node_count: int) -> NetworkCase:
    """Build tonic neurons with drive replacements on event boundaries."""

    ports = tuple(
        InputPort(f"drive_{index}", index, InputMode.DRIVE, "drive")
        for index in range(node_count)
    )
    updates = tuple(
        [DriveInput(5.0, f"drive_{index}", 20.0) for index in range(node_count)]
        + [DriveInput(25.0, f"drive_{index}", 0.0) for index in range(node_count)]
    )
    return NetworkCase(
        f"drive_boundaries_{node_count}",
        _base_graph(node_count, (), ports),
        (),
        updates,
        40.0,
        node_count,
    )


def alpha_population_case(node_count: int) -> NetworkCase:
    """Build a population driven through folded alpha-current synapses."""

    ports = tuple(
        InputPort(f"alpha_{index}", index, InputMode.SPIKE)
        for index in range(node_count)
    )
    inputs = tuple(
        SpikeInput(1.0, f"alpha_{index}", 40.0)
        for index in range(node_count)
    )
    graph = Graph(
        models=(GraphModel("alpha_lif", VALIDATION_ALPHA_LIF),),
        nodes=tuple(
            GraphNode(
                index,
                "alpha_lif",
                (-65.0, 0.0, 0.0),
                {},
                synapse="alpha",
                receptor="i_exc",
                output="current",
            )
            for index in range(node_count)
        ),
        input_ports=ports,
        synapses=(GraphSynapse("alpha", VALIDATION_ALPHA_SYNAPSE),),
    )
    return NetworkCase(
        f"alpha_population_{node_count}", graph, inputs, (), 12.0, node_count
    )


def per_edge_shared_fanin_case(source_count: int) -> NetworkCase:
    """Build fan-in using identical per-edge exponential kernels."""

    target = source_count
    graph = Graph(
        models=(
            GraphModel("lif", VALIDATION_LIF),
            GraphModel("target", VALIDATION_ALPHA_LIF),
        ),
        nodes=tuple(
            GraphNode(index, "lif", -65.0, {"drive": 0.0})
            for index in range(source_count)
        )
        + (GraphNode(target, "target", -65.0, {}),),
        edges=tuple(
            GraphEdge(
                index,
                index,
                target,
                20.0 / source_count,
                0.25,
                synapse="exp",
                receptor="i_exc",
                output="current",
                synapse_bindings={"tau_s": 5.0},
            )
            for index in range(source_count)
        ),
        input_ports=tuple(
            InputPort(f"source_{index}", index, InputMode.SPIKE)
            for index in range(source_count)
        ),
        synapses=(GraphSynapse("exp", VALIDATION_EXP_SYNAPSE),),
    )
    return NetworkCase(
        f"per_edge_shared_fanin_{source_count}",
        graph,
        tuple(
            SpikeInput(1.0, f"source_{index}", 20.0)
            for index in range(source_count)
        ),
        (),
        5.0,
        source_count + 1,
    )


def per_edge_distinct_fanout_case(target_count: int) -> NetworkCase:
    """Build fan-out using heterogeneous per-edge exponential kernels."""

    source_count = 3
    node_count = source_count + target_count
    taus = (4.0, 5.0, 7.0)
    edges: list[GraphEdge] = []
    edge_id = 0
    for target in range(source_count, node_count):
        for source, tau in enumerate(taus):
            edges.append(
                GraphEdge(
                    edge_id,
                    source,
                    target,
                    7.0,
                    0.25,
                    synapse="exp",
                    receptor="i_exc",
                    output="current",
                    synapse_bindings={"tau_s": tau},
                )
            )
            edge_id += 1
    graph = Graph(
        models=(
            GraphModel("lif", VALIDATION_LIF),
            GraphModel("target", VALIDATION_ALPHA_LIF),
        ),
        nodes=tuple(
            GraphNode(index, "lif", -65.0, {"drive": 0.0})
            for index in range(source_count)
        )
        + tuple(
            GraphNode(index, "target", -65.0, {})
            for index in range(source_count, node_count)
        ),
        edges=tuple(edges),
        input_ports=tuple(
            InputPort(f"source_{index}", index, InputMode.SPIKE)
            for index in range(source_count)
        ),
        synapses=(GraphSynapse("exp", VALIDATION_EXP_SYNAPSE),),
    )
    return NetworkCase(
        f"per_edge_distinct_fanout_{target_count}",
        graph,
        tuple(
            SpikeInput(1.0, f"source_{index}", 20.0)
            for index in range(source_count)
        ),
        (),
        5.0,
        node_count,
    )


def per_edge_alpha_fanout_case(
    target_count: int, *, equal_membrane_rate: bool
) -> NetworkCase:
    """Build fan-out using per-edge alpha kernels with optional equal rates."""

    source_count = 1 if equal_membrane_rate else 2
    node_count = source_count + target_count
    kernels = ((10.0, 80.0),) if equal_membrane_rate else (
        (5.0, 20.0),
        (10.0, 40.0),
    )
    edges: list[GraphEdge] = []
    edge_id = 0
    for target in range(source_count, node_count):
        for source, (tau, weight) in enumerate(kernels):
            edges.append(
                GraphEdge(
                    edge_id,
                    source,
                    target,
                    weight,
                    0.25,
                    synapse="alpha",
                    receptor="i_exc",
                    output="current",
                    synapse_bindings={"tau_s": tau},
                )
            )
            edge_id += 1
    graph = Graph(
        models=(
            GraphModel("lif", VALIDATION_LIF),
            GraphModel("target", VALIDATION_ALPHA_LIF),
        ),
        nodes=tuple(
            GraphNode(index, "lif", -65.0, {"drive": 0.0})
            for index in range(source_count)
        )
        + tuple(
            GraphNode(index, "target", -65.0, {})
            for index in range(source_count, node_count)
        ),
        edges=tuple(edges),
        input_ports=tuple(
            InputPort(f"source_{index}", index, InputMode.SPIKE)
            for index in range(source_count)
        ),
        synapses=(GraphSynapse("alpha", VALIDATION_ALPHA_SYNAPSE),),
    )
    regime = "equal" if equal_membrane_rate else "distinct"
    return NetworkCase(
        f"per_edge_alpha_{regime}_fanout_{target_count}",
        graph,
        tuple(
            SpikeInput(1.0, f"source_{index}", 20.0)
            for index in range(source_count)
        ),
        (),
        30.0,
        node_count,
    )


def adaptive_population_case(node_count: int) -> NetworkCase:
    """Build independent adaptive LIF neurons under constant drive."""

    graph = Graph(
        models=(GraphModel("adaptive_lif", VALIDATION_ADAPTIVE_LIF),),
        nodes=tuple(
            GraphNode(index, "adaptive_lif", (-65.0, 0.0), {})
            for index in range(node_count)
        ),
    )
    return NetworkCase(
        f"adaptive_population_{node_count}",
        graph,
        (),
        (),
        140.0,
        node_count * 3,
    )


def mixed_chain_case(node_count: int) -> NetworkCase:
    """Build a chain that mixes scalar and adaptive LIF nodes."""

    nodes = tuple(
        GraphNode(index, "lif", -65.0, {"drive": 0.0})
        if index % 2 == 0
        else GraphNode(
            index,
            "alpha_lif",
            (-65.0, 0.0, 0.0),
            {},
            synapse="alpha",
            receptor="i_exc",
            output="current",
        )
        for index in range(node_count)
    )
    edges = tuple(
        GraphEdge(
            index,
            index,
            index + 1,
            40.0 if (index + 1) % 2 else 20.0,
            0.25,
        )
        for index in range(node_count - 1)
    )
    graph = Graph(
        models=(
            GraphModel("alpha_lif", VALIDATION_ALPHA_LIF),
            GraphModel("lif", VALIDATION_LIF),
        ),
        nodes=nodes,
        edges=edges,
        input_ports=(InputPort("start", 0, InputMode.SPIKE),),
        synapses=(GraphSynapse("alpha", VALIDATION_ALPHA_SYNAPSE),),
    )
    return NetworkCase(
        f"mixed_chain_{node_count}",
        graph,
        (SpikeInput(1.0, "start", 20.0),),
        (),
        2.0 + node_count * 5.0,
        node_count,
    )


def alpha_drive_boundary_case(node_count: int) -> NetworkCase:
    """Build folded-alpha nodes with simultaneous drive updates."""

    ports = tuple(
        InputPort(f"drive_{index}", index, InputMode.DRIVE, "drive")
        for index in range(node_count)
    )
    graph = Graph(
        models=(GraphModel("alpha_lif", VALIDATION_ALPHA_LIF),),
        nodes=tuple(
            GraphNode(
                index,
                "alpha_lif",
                -65.0,
                {},
                synapse="alpha",
                receptor="i_exc",
                output="current",
            )
            for index in range(node_count)
        ),
        input_ports=ports,
        synapses=(GraphSynapse("alpha", VALIDATION_ALPHA_SYNAPSE),),
    )
    updates = tuple(
        DriveInput(5.0, f"drive_{index}", 30.0) for index in range(node_count)
    )
    return NetworkCase(
        f"alpha_drive_boundaries_{node_count}", graph, (), updates, 12.0, node_count
    )


def validation_corpus() -> tuple[NetworkCase, ...]:
    """Return the deterministic correctness workload corpus."""

    return (
        chain_case(32),
        fanout_fanin_case(32),
        recurrent_ring_case(32),
        bipartite_burst_case(32, 96),
        sparse_recurrent_case(128, 4),
        driven_population_case(64),
        drive_boundary_case(64),
        alpha_population_case(64),
        per_edge_shared_fanin_case(64),
        per_edge_distinct_fanout_case(96),
        per_edge_alpha_fanout_case(64, equal_membrane_rate=True),
        per_edge_alpha_fanout_case(64, equal_membrane_rate=False),
        adaptive_population_case(64),
        mixed_chain_case(16),
        alpha_drive_boundary_case(32),
    )


def scaling_corpus() -> tuple[NetworkCase, ...]:
    """Return larger workloads used for scaling measurements."""

    cases: list[NetworkCase] = []
    for size in (16, 64, 256, 1024):
        cases.append(chain_case(size))
        cases.append(driven_population_case(size))
    for sources, targets in ((8, 24), (16, 48), (32, 96), (64, 192), (128, 384)):
        cases.append(bipartite_burst_case(sources, targets))
    cases.append(sparse_recurrent_case(1024, 4))
    for size in (16, 64, 256, 1024):
        cases.append(alpha_population_case(size))
        cases.append(adaptive_population_case(size))
        cases.append(per_edge_shared_fanin_case(size))
    for size in (16, 64, 256):
        cases.append(per_edge_distinct_fanout_case(size))
        cases.append(per_edge_alpha_fanout_case(size, equal_membrane_rate=False))
    for size in (16, 64, 256):
        cases.append(per_edge_alpha_fanout_case(size, equal_membrane_rate=True))
    for size in (16, 64):
        cases.append(mixed_chain_case(size))
    return tuple(cases)


def _queue_capacity(case: NetworkCase) -> int:
    return max(
        4096,
        len(case.graph.edges) * 2
        + len(case.graph.nodes) * 8
        + len(case.spike_inputs)
        + len(case.drive_inputs),
    )


def _run(
    runner,
    case: NetworkCase,
    queue_capacity: int,
    *,
    recording: RecordingConfig | None = None,
):
    return runner.run(
        spike_inputs=case.spike_inputs,
        drive_inputs=case.drive_inputs,
        t_end=case.t_end,
        queue_capacity=queue_capacity,
        output_capacity=250_000,
        same_time_cascade_limit=4096,
        recording=recording,
    ).core


def _assert_invariants(case: NetworkCase, resolved: ResolvedGraph, result) -> tuple[str, ...]:
    spikes = result.spikes
    if len(spikes) < case.expected_min_spikes:
        raise AssertionError(
            f"{case.name}: expected at least {case.expected_min_spikes} spikes, got {len(spikes)}"
        )
    if any(left.t > right.t for left, right in zip(spikes, spikes[1:])):
        raise AssertionError(f"{case.name}: spike times are not chronological")
    if result.stats.output_spikes != len(spikes):
        raise AssertionError(f"{case.name}: output spike accounting mismatch")
    expected_inputs = sum(event.t <= case.t_end for event in case.spike_inputs)
    expected_drives = sum(event.t <= case.t_end for event in case.drive_inputs)
    if result.stats.input_spikes_processed != expected_inputs:
        raise AssertionError(f"{case.name}: input event accounting mismatch")
    if result.stats.drive_updates_processed != expected_drives:
        raise AssertionError(f"{case.name}: drive event accounting mismatch")

    expected_deliveries = 0
    for spike in spikes:
        for edge in resolved.edges:
            if edge.pre == spike.node and spike.t + edge.delay <= case.t_end:
                expected_deliveries += 1
    if result.stats.deliveries_scheduled != expected_deliveries:
        raise AssertionError(f"{case.name}: scheduled delivery accounting mismatch")
    if result.stats.deliveries_processed != result.stats.deliveries_scheduled:
        raise AssertionError(f"{case.name}: not every in-horizon delivery was processed")

    last_spike: dict[int, float] = {}
    for spike in spikes:
        previous = last_spike.get(spike.node)
        refractory = resolved.models[spike.node].refractory
        if previous is not None and spike.t - previous < refractory:
            raise AssertionError(f"{case.name}: node {spike.node} violated refractory spacing")
        last_spike[spike.node] = spike.t
    for state in result.states:
        values = state.values if hasattr(state, "values") else (state.value,)
        if any(not math.isfinite(value) for value in values) or state.t_last != case.t_end:
            raise AssertionError(f"{case.name}: invalid final state")
    return (
        "minimum_activity",
        "chronological_spikes",
        "output_accounting",
        "input_accounting",
        "drive_accounting",
        "delivery_conservation",
        "refractory_spacing",
        "finite_final_state",
    )


def evaluate_case(
    core: CoreEvaluator,
    case: NetworkCase,
    *,
    deterministic_replays: int = 3,
    timed_repetitions: int = 5,
    verify_queue_boundary: bool = True,
    verify_causal_trace: bool = True,
) -> CaseReport:
    """Run correctness checks, deterministic replays, and timed repetitions."""

    resolved = case.graph.resolve()
    capacity = _queue_capacity(case)
    with resolved.compile(core) as runner:
        baseline = _run(runner, case, capacity)
        invariants = _assert_invariants(case, resolved, baseline)
        trace_records = 0
        trace_checks: tuple[str, ...] = ()
        if verify_causal_trace:
            recording = RecordingConfig(capacity=0)
            metadata = TraceArtifactMetadata.from_resolved_graph(
                resolved, recording
            )
            with TemporaryDirectory(prefix="lacuna-trace-audit-") as directory:
                artifact_path = Path(directory) / "causal.lctrace"
                with TraceArtifactWriter(
                    artifact_path, metadata, chunk_records=1024
                ) as writer:
                    traced = _run(
                        runner,
                        case,
                        capacity,
                        recording=replace(recording, consumer=writer),
                    )
                    writer.set_summary(
                        {"t_end": case.t_end, "run_stats": asdict(traced.stats)}
                    )
                if traced != baseline:
                    raise AssertionError(
                        f"{case.name}: enabling a causal trace changed the network result"
                    )
                with TraceArtifactReader(artifact_path) as reader:
                    audit = reader.audit(
                        traced,
                        resolved,
                        t_end=case.t_end,
                        expected_input_count=sum(
                            event.t <= case.t_end for event in case.spike_inputs
                        ),
                        expected_drive_count=sum(
                            event.t <= case.t_end for event in case.drive_inputs
                        ),
                    )
            trace_records = audit.record_count
            trace_checks = audit.checks
        for _ in range(deterministic_replays - 1):
            replay = _run(runner, case, capacity)
            if replay != baseline:
                raise AssertionError(f"{case.name}: deterministic replay mismatch")

        peak = baseline.stats.peak_queue_occupancy
        if verify_queue_boundary:
            exact = _run(runner, case, max(peak, 1))
            if exact != baseline:
                raise AssertionError(f"{case.name}: exact peak-capacity replay mismatch")
            if peak > 1:
                try:
                    _run(runner, case, peak - 1)
                except CoreError as exc:
                    if "queue capacity" not in str(exc):
                        raise
                else:
                    raise AssertionError(
                        f"{case.name}: capacity below observed peak did not fail"
                    )

        durations: list[float] = []
        for _ in range(timed_repetitions):
            started = time.perf_counter()
            _run(runner, case, capacity)
            durations.append(time.perf_counter() - started)
    median_seconds = statistics.median(durations)
    events_per_second = (
        baseline.stats.events_popped / median_seconds if median_seconds > 0.0 else math.inf
    )
    graph_text = case.graph.to_text().encode("utf-8")
    return CaseReport(
        name=case.name,
        graph_sha256=hashlib.sha256(graph_text).hexdigest(),
        nodes=len(case.graph.nodes),
        edges=len(case.graph.edges),
        input_spikes=len(case.spike_inputs),
        drive_updates=len(case.drive_inputs),
        output_spikes=len(baseline.spikes),
        events_popped=baseline.stats.events_popped,
        deliveries=baseline.stats.deliveries_processed,
        stale_predictions=baseline.stats.stale_predictions,
        peak_queue_occupancy=peak,
        queue_headroom_at_test_capacity=capacity - peak,
        max_same_time_cascade_depth=baseline.stats.max_same_time_cascade_depth,
        median_seconds=median_seconds,
        events_per_second=events_per_second,
        deterministic_replays=deterministic_replays,
        trace_records=trace_records,
        trace_audit_checks=trace_checks,
        invariants_checked=invariants,
    )


def run_campaign(core: CoreEvaluator) -> dict[str, object]:
    """Evaluate the full validation corpus with one C evaluator."""

    validation_reports = [evaluate_case(core, case) for case in validation_corpus()]
    scaling_reports = [
        evaluate_case(
            core,
            case,
            deterministic_replays=2,
            timed_repetitions=3,
            verify_queue_boundary=False,
        )
        for case in scaling_corpus()
    ]
    return {
        "schema": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "validation_cases": [asdict(report) for report in validation_reports],
        "scaling_cases": [asdict(report) for report in scaling_reports],
        "summary": {
            "validation_case_count": len(validation_reports),
            "scaling_case_count": len(scaling_reports),
            "total_validation_spikes": sum(item.output_spikes for item in validation_reports),
            "total_validation_events": sum(item.events_popped for item in validation_reports),
            "total_validation_trace_records": sum(
                item.trace_records for item in validation_reports
            ),
            "total_scaling_trace_records": sum(
                item.trace_records for item in scaling_reports
            ),
            "largest_node_count": max(item.nodes for item in scaling_reports),
            "largest_edge_count": max(item.edges for item in scaling_reports),
            "all_invariants_passed": True,
            "deterministic_replay_passed": True,
            "queue_boundary_checks_passed": True,
            "causal_trace_audits_passed": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the validation campaign command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="path to the built Lacuna C library")
    parser.add_argument("--output", required=True, help="path for the JSON report")
    arguments = parser.parse_args(argv)
    report = run_campaign(CoreEvaluator(arguments.library))
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
