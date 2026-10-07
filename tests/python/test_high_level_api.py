from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from lacuna import (
    AdEx,
    AlphaCurrent,
    AtTimes,
    DriveSeries,
    Engine,
    Every,
    ExponentialCurrent,
    FixedInDegree,
    FixedOutDegree,
    FixedProbability,
    LIF,
    LocallyConnected,
    NativeEventEncoder,
    Network,
    NetworkBuilder,
    NeuronPolarity,
    Normal,
    OneToOne,
    OutputPort,
    RateDecoder,
    RecordingPlan,
    RunOptions,
    ScalarPresentation,
    SpikeRecording,
    SpikeTrain,
    StateRecording,
    TraceRecording,
    Uniform,
    coefficient_of_variation,
    firing_rate,
    interspike_intervals,
    population_rate,
    spike_count,
)
from lacuna.codec import TTFSEncoder
from lacuna.errors import ResolutionError
from lacuna.ffi import CoreEvaluator
from lacuna.tracefile import read_trace_artifact


def test_builder_expands_populations_patterns_and_filtered_synapses() -> None:
    builder = NetworkBuilder("mixed")
    source = builder.population(
        "source",
        3,
        LIF(name="source_lif"),
        parameters={"drive": (16.0, 18.0, 20.0)},
    )
    target = builder.population(
        "target",
        2,
        LIF(name="target_lif", synaptic_input=True),
    )
    edges = builder.connect(
        source,
        target,
        synapse=AlphaCurrent(5.0, name="alpha"),
        pattern=FixedProbability(1.0, seed=8),
        weight=Normal(2.0, 0.1, seed=9),
        delay=Uniform(0.5, 1.5, seed=10),
    )
    network = builder.build()
    report = network.validate()

    assert len(edges) == 6
    assert [node.bindings["drive"] for node in network.graph.nodes[:3]] == [
        16.0,
        18.0,
        20.0,
    ]
    assert report.node_count == 5
    assert report.edge_count == 6
    assert report.dispatch_counts == {"CLOSED_FORM": 3, "ROOT_FIND": 2}
    assert report.synapse_tier_counts == {"DELTA": 3, "FOLDED_SHARED": 2}
    assert all(edge.synapse == "alpha" for edge in network.graph.edges)
    assert all(edge.receptor == "i_syn" for edge in network.graph.edges)


def test_one_to_one_and_exponential_current_are_explicit() -> None:
    builder = NetworkBuilder()
    pre = builder.population("pre", 2, LIF(name="pre_lif"))
    post = builder.population(
        "post", 2, LIF(name="post_lif", synaptic_input=True)
    )
    builder.connect(
        pre,
        post,
        pattern=OneToOne(),
        synapse=ExponentialCurrent(3.0),
        weight=(2.0, 3.0),
    )
    network = builder.build()
    assert [(edge.pre, edge.post, edge.weight) for edge in network.graph.edges] == [
        (0, 2, 2.0),
        (1, 3, 3.0),
    ]


def test_locally_connected_supports_lines_rings_and_aligned_populations() -> None:
    line_builder = NetworkBuilder("line")
    line = line_builder.population("line", 5, LIF())
    line_edges = line_builder.connect(
        line,
        line,
        pattern=LocallyConnected(radius=1),
    )
    line_network = line_builder.build()
    assert len(line_edges) == 8
    assert [(edge.pre, edge.post) for edge in line_network.graph.edges] == [
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
    ]

    ring_builder = NetworkBuilder("ring")
    ring = ring_builder.population("ring", 5, LIF())
    ring_builder.connect(
        ring,
        ring,
        pattern=LocallyConnected(radius=1, wrap=True),
    )
    ring_network = ring_builder.build()
    assert len(ring_network.graph.edges) == 10
    assert {(edge.pre, edge.post) for edge in ring_network.graph.edges} >= {
        (0, 4),
        (4, 0),
    }

    aligned_builder = NetworkBuilder("aligned")
    pre = aligned_builder.population("pre", 3, LIF(name="pre"))
    post = aligned_builder.population("post", 3, LIF(name="post"))
    aligned_builder.connect(
        pre,
        post,
        pattern=LocallyConnected(radius=0),
    )
    aligned = aligned_builder.build()
    assert [(edge.pre, edge.post) for edge in aligned.graph.edges] == [
        (0, 3),
        (1, 4),
        (2, 5),
    ]


def test_fixed_degree_patterns_enforce_exact_recurrent_degrees() -> None:
    out_builder = NetworkBuilder("out-degree")
    population = out_builder.population("p", 6, LIF())
    out_builder.connect(
        population,
        population,
        pattern=FixedOutDegree(2, seed=10, exclude_self=True),
    )
    out_network = out_builder.build()
    assert {
        node: sum(edge.pre == node for edge in out_network.graph.edges)
        for node in population.node_ids
    } == {node: 2 for node in population.node_ids}

    in_builder = NetworkBuilder("in-degree")
    population = in_builder.population("p", 6, LIF())
    in_builder.connect(
        population,
        population,
        pattern=FixedInDegree(3, seed=11, exclude_self=True),
    )
    in_network = in_builder.build()
    assert {
        node: sum(edge.post == node for edge in in_network.graph.edges)
        for node in population.node_ids
    } == {node: 3 for node in population.node_ids}


def test_reservoir_constructor_builds_and_persists_recurrent_topology(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    builder = NetworkBuilder("reservoir-network")
    reservoir = builder.reservoir(
        "liquid",
        8,
        LIF(name="reservoir_lif"),
        connectivity=LocallyConnected(radius=2, wrap=True),
        weight=Normal(0.5, 0.1, seed=22),
        delay=1.0,
    )
    stimulus = builder.neuron("stimulus", LIF(name="stimulus_lif"))
    builder.connect(stimulus, reservoir, pattern=FixedOutDegree(3, seed=4))
    input_port = builder.input("stimulus-in", stimulus)
    builder.output("stimulus-out", stimulus)
    network = builder.build()

    assert len(reservoir) == 8
    assert len(reservoir.recurrent_edge_ids) == 32
    assert network.node_ids(reservoir) == reservoir.node_ids
    assert network.reservoir("liquid").recurrent_edge_ids == reservoir.recurrent_edge_ids
    assert sum(
        node.polarity is NeuronPolarity.INHIBITORY
        for node in network.graph.nodes[:8]
    ) == 2
    assert all(edge.weight >= 0.0 for edge in network.graph.edges)
    resolved_recurrent = network.graph.resolve().effective_edges[:32]
    assert all(
        edge.weight < 0.0
        if network.graph.nodes[edge.pre].polarity is NeuronPolarity.INHIBITORY
        else edge.weight > 0.0
        for edge in resolved_recurrent
    )

    path = tmp_path / "reservoir.json"
    network.save(path)
    restored = Network.load(path)
    assert restored.reservoir("liquid").node_ids == tuple(range(8))
    assert len(restored.reservoir("liquid").recurrent_edge_ids) == 32

    with Engine(core._lib._name).compile(restored) as simulation:
        result = simulation.run(
            3.0,
            inputs={input_port: SpikeTrain((1.0,), (20.0,))},
        )
    assert result.spikes.select("stimulus").times == (1.0,)


def test_builder_assigns_intrinsic_polarity_and_rejects_signed_weights() -> None:
    builder = NetworkBuilder("dale")
    neurons = builder.population(
        "neurons",
        2,
        LIF(),
        polarity=("excitatory", "inhibitory"),
    )
    builder.connect(neurons, neurons, pattern=OneToOne(), weight=(2.0, 3.0))
    network = builder.build()
    assert [node.polarity for node in network.graph.nodes] == [
        NeuronPolarity.EXCITATORY,
        NeuronPolarity.INHIBITORY,
    ]
    assert [edge.weight for edge in network.graph.resolve().effective_edges] == [
        2.0,
        -3.0,
    ]

    invalid = NetworkBuilder("invalid-dale")
    source = invalid.neuron(
        "source", LIF(name="source"), polarity=NeuronPolarity.INHIBITORY
    )
    target = invalid.neuron("target", LIF(name="target"))
    with pytest.raises(ResolutionError, match="nonnegative magnitudes"):
        invalid.connect(source, target, weight=-1.0)


def test_inhibitory_polarity_signs_filtered_synapse_deposits(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("filtered-dale")
    source = builder.neuron(
        "source",
        LIF(name="source"),
        polarity=NeuronPolarity.INHIBITORY,
    )
    target = builder.neuron(
        "target",
        LIF(name="target", synaptic_input=True),
    )
    builder.connect(source, target, synapse=AlphaCurrent(5.0), weight=20.0)
    port = builder.input("stimulus", source)
    network = builder.build()

    assert network.graph.edges[0].weight == 20.0
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            2.0,
            inputs={port: SpikeTrain((1.0,), 20.0)},
        )
    target_state = result.final_states[target.id].values
    assert target_state[1] < 0.0
    assert target_state[2] < 0.0


def test_unconnected_filtered_lif_nodes_lower_to_cheaper_delta_variant() -> None:
    builder = NetworkBuilder()
    pre = builder.population("pre", 1, LIF(name="pre_lif"))
    post = builder.population(
        "post", 2, LIF(name="post_lif", synaptic_input=True)
    )
    builder.connect(
        pre,
        post,
        pattern=FixedProbability(0.0, seed=1),
        synapse=AlphaCurrent(5.0),
    )
    network = builder.build()
    report = network.validate()
    assert report.edge_count == 0
    assert report.state_count == 3
    assert report.dispatch_counts == {"REACTIVE": 3}
    assert all(
        node.model == "post_lif__delta_only"
        for node in network.graph.nodes[1:]
    )


def test_each_port_can_have_an_independent_codec() -> None:
    builder = NetworkBuilder()
    population = builder.population("input", 2, LIF())
    ports = builder.inputs(
        "sensor",
        population,
        encoder=(
            NativeEventEncoder(),
            TTFSEncoder(min_latency=1.0, max_latency=5.0, amplitude=20.0),
        ),
    )
    outputs = builder.outputs(
        "readout",
        population,
        decoder=(RateDecoder(), RateDecoder()),
    )
    network = builder.build()
    assert ports.port_ids == ("sensor[0]", "sensor[1]")
    assert outputs.port_ids == ("readout[0]", "readout[1]")
    assert network.graph.input_ports[0].encoder != network.graph.input_ports[1].encoder


def test_network_json_is_deterministic_hashed_and_retains_authoring_metadata(
    tmp_path,
) -> None:
    builder = NetworkBuilder("persisted", metadata={"purpose": "test"})
    population = builder.population("neurons", 2, LIF())
    builder.output("first", population[0])
    network = builder.build()
    path = tmp_path / "network.json"
    second_path = tmp_path / "network-copy.json"

    network.save(path)
    restored = Network.load(path)
    restored.save(second_path)

    assert path.read_bytes() == second_path.read_bytes()
    assert restored.semantic_sha256 == network.semantic_sha256
    assert restored.population("neurons").node_ids == (0, 1)
    assert restored.neuron("neurons[1]").id == 1
    assert restored.metadata == {"purpose": "test"}

    tampered = json.loads(path.read_text())
    tampered["authoring"]["name"] = "changed"
    with pytest.raises(ResolutionError, match="metadata hash mismatch"):
        Network.from_text(json.dumps(tampered))


def test_compiled_network_runs_named_inputs_records_exact_state_and_analyzes(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("run")
    population = builder.population("neurons", 1, LIF())
    port = builder.input("stimulus", population[0])
    builder.output("spikes", population[0])
    network = builder.build()
    recording = RecordingPlan(
        states=(
            StateRecording(population, Every(0.5), variables=("v",)),
            StateRecording(population, AtTimes((1.0,)), variables=("v",)),
        )
    )

    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            3.0,
            inputs={port: SpikeTrain((1.0,), (20.0,))},
            recording=recording,
        )

    assert result.spikes.times == (1.0,)
    assert result.spikes.nodes == (0,)
    assert result.port_spikes[0].port == "spikes"
    assert len(result.states.samples) == 8
    at_spike = [sample for sample in result.states.samples if sample.t == 1.0]
    assert all(sample.values == (-65.0,) for sample in at_spike)
    assert result.final_states[0].values == (-65.0,)
    assert spike_count(result.spikes) == {0: 1}
    assert firing_rate(result.spikes, t_start=0.0, t_end=2.0) == {0: 0.5}
    assert interspike_intervals(result.spikes) == {0: ()}
    assert coefficient_of_variation(result.spikes) == {0: None}
    rate = population_rate(
        result.spikes,
        t_start=0.0,
        t_end=2.0,
        bin_width=1.0,
        nodes=(0,),
    )
    assert rate.times == (0.0, 1.0)
    assert rate.values == (0.0, 1.0)


def test_scalar_presentations_and_drive_series_use_named_ports(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder()
    population = builder.population("n", 1, LIF())
    encoded = builder.input(
        "encoded",
        population[0],
        encoder=TTFSEncoder(1.0, 5.0, amplitude=20.0),
    )
    network = builder.build()
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            6.0,
            inputs={encoded: ScalarPresentation(0.0, 6.0, 1.0)},
        )
    assert result.spikes.times == (1.0,)

    drive_builder = NetworkBuilder()
    driven_population = drive_builder.population("n", 1, LIF())
    drive = drive_builder.inputs(
        "drive",
        driven_population[0],
        parameter="drive",
    )[0]
    driven_network = drive_builder.build()
    with Engine(core._lib._name).compile(driven_network) as simulation:
        driven = simulation.run(
            40.0,
            inputs={drive: DriveSeries((0.0,), (20.0,))},
        )
    assert driven.spikes.times[0] == pytest.approx(
        20.0 * math.log(4.0), abs=2e-12
    )


def test_high_level_trace_streams_to_binary_artifact(
    core: CoreEvaluator, tmp_path
) -> None:
    builder = NetworkBuilder("trace")
    population = builder.population("n", 1, LIF())
    port = builder.input("stimulus", population[0])
    network = builder.build()
    path = tmp_path / "run.lctrace"

    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            2.0,
            inputs={port: SpikeTrain((1.0,), 20.0)},
            recording=RecordingPlan(
                trace=TraceRecording(
                    targets=population,
                    kinds=("INPUT_SPIKE", "SPIKE", "RESET", "FINAL_STATE"),
                    variables=("v",),
                    path=path,
                    chunk_records=2,
                )
            ),
        )

    artifact = read_trace_artifact(path)
    assert result.trace == ()
    assert result.trace_path == path
    assert [record.kind.name for record in artifact.records] == [
        "INPUT_SPIKE",
        "SPIKE",
        "RESET",
        "FINAL_STATE",
    ]
    assert artifact.summary["run_stats"]["output_spikes"] == 1


def test_recording_can_disable_raw_spike_retention_while_decoding(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder()
    population = builder.population("n", 1, LIF())
    port = builder.input("stimulus", population[0])
    builder.output("rate", population[0], decoder=RateDecoder())
    network = builder.build()
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            2.0,
            inputs={port: SpikeTrain((1.0,), 20.0)},
            recording=RecordingPlan(spikes=None),
        )
    assert result.spikes.events == ()
    assert result.decoded[0].count == 1


def test_recording_can_count_spikes_without_retaining_payloads(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder()
    population = builder.population("n", 1, LIF())
    port = builder.input("stimulus", population[0])
    network = builder.build()
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            2.0,
            inputs={port: SpikeTrain((1.0,), 20.0)},
            recording=RecordingPlan(spikes=None),
            options=RunOptions(return_final_state=False),
        )
    assert result.spikes.events == ()
    assert result.stats.output_spikes == 1
    assert result.final_states == ()
    assert result.raw.core.kernel_seconds >= 0.0


def test_high_level_incremental_run_preserves_boundary_inputs_and_recording(
    core: CoreEvaluator, tmp_path
) -> None:
    builder = NetworkBuilder("incremental")
    population = builder.population("n", 1, LIF())
    port = builder.input("stimulus", population[0])
    network = builder.build()
    path = tmp_path / "incremental.lctrace"
    plan = RecordingPlan(
        states=(StateRecording(population, Every(1.0), ("v",)),),
        trace=TraceRecording(targets=population, path=path, chunk_records=2),
    )

    with Engine(core._lib._name).compile(network) as simulation:
        with simulation.start_run(3.0, recording=plan) as run:
            first = run.advance(1.0)
            second = run.advance(
                2.0, inputs={port: SpikeTrain((1.0,), 20.0)}
            )
            final = run.finish()

    assert first.spikes.events == ()
    assert second.spikes.times == (1.0,)
    assert final.final_states[0].values == (-65.0,)
    assert [sample.t for sample in first.states] == [0.0]
    assert [sample.t for sample in second.states] == [1.0]
    assert [sample.t for sample in final.states] == [2.0, 3.0]
    artifact = read_trace_artifact(path)
    assert artifact.complete
    assert artifact.summary["run_stats"]["output_spikes"] == 1


def test_engine_uses_shared_compiled_scheduler_for_scalar_network(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("automatic-fast-path")
    population = builder.population("n", 2, LIF(drive=20.0))
    builder.connect(population[0], population[1], weight=2.0, delay=1.0)

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        assert simulation.preferred_execution_path == "compiled_sparse"
        assert simulation.last_execution_path is None
        result = simulation.run(40.0)
        assert simulation.last_execution_path == "compiled_sparse"

    assert result.spikes.events
    assert result.weights == (2.0,)


def test_immediate_incremental_finish_uses_shared_compiled_scheduler(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("deferred-fast-path")
    population = builder.population("n", 1, LIF())
    port = builder.input("stimulus", population[0])

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        with simulation.start_run(2.0) as run:
            result = run.finish(inputs={port: SpikeTrain((1.0,), 20.0)})
        assert simulation.last_execution_path == "compiled_sparse"

    assert result.spikes.times == (1.0,)


def test_state_recording_uses_the_same_compiled_scheduler(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("recording-fallback")
    population = builder.population("n", 1, LIF(drive=20.0))
    plan = RecordingPlan(
        states=(StateRecording(population, Every(1.0), ("v",)),)
    )

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        assert simulation.preferred_execution_path == "compiled_sparse"
        result = simulation.run(2.0, recording=plan)
        assert simulation.last_execution_path == "compiled_sparse"

    assert [sample.t for sample in result.states] == [0.0, 1.0, 2.0]


def test_actual_incremental_advance_uses_the_same_compiled_scheduler(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("incremental-fallback")
    builder.population("n", 1, LIF(drive=20.0))

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        with simulation.start_run(2.0) as run:
            run.advance(1.0)
            result = run.finish()
        assert simulation.last_execution_path == "compiled_sparse"

    assert result.final_states[0].t == 2.0


def test_high_level_compiled_network_exports_the_c_graph_image(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    builder = NetworkBuilder("deployment-image")
    builder.population("n", 1, LIF(drive=20.0))
    destination = tmp_path / "deployment.lcg"

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        image = simulation.compiled_graph_image()
        assert simulation.save_compiled_graph_image(destination) == destination

    assert destination.read_bytes() == image
    with core.load_compiled_graph_file(destination) as loaded:
        result = loaded.run((-65.0,), t_end=2.0)
    assert result.states[0].t_last == 2.0


def test_builder_rejects_cross_network_handles_and_mutation_after_build() -> None:
    first = NetworkBuilder("first")
    population = first.population("p", 1, LIF())
    first.build()
    with pytest.raises(RuntimeError, match="sealed"):
        first.neuron("late", LIF(name="late"))

    second = NetworkBuilder("second")
    other = second.population("q", 1, LIF(name="other"))
    with pytest.raises(ResolutionError, match="another builder"):
        second.connect(population, other)


def test_learned_network_preserves_unsorted_noncontiguous_edge_ids(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("sparse-edge-ids")
    source = builder.neuron("source", LIF())
    targets = builder.population("targets", 2, LIF())
    builder.connect(source, targets, weight=(2.0, 3.0))
    network = builder.build()
    edges = tuple(
        replace(edge, id=100 + 3 * edge.id)
        for edge in reversed(network.graph.edges)
    )
    network = replace(network, graph=replace(network.graph, edges=edges))

    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(1.0)
    snapshot = result.learned_network()

    assert result.weights == (2.0, 3.0)
    assert [(edge.id, edge.weight) for edge in snapshot.graph.edges] == [
        (103, 3.0),
        (100, 2.0),
    ]
    assert network.graph.edges == edges
    snapshot.validate()
