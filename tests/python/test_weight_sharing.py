import math
from dataclasses import replace

import pytest

from lacuna import (
    Convolution2D,
    Engine,
    LIF,
    ModulatedSTDP,
    ModulationSeries,
    Network,
    NetworkBuilder,
    PairSTDP,
    SpikeTrain,
    TripletSTDP,
)
from lacuna.errors import ResolutionError
from lacuna.ffi import CoreEvaluator


def _two_copy_projection(rule, *, weight=0.5):
    builder = NetworkBuilder("two-copy-kernel")
    pre = builder.population("pre", 2, LIF())
    post = builder.population("post", 2, LIF())
    edges = builder.connect(
        pre,
        post,
        pattern=Convolution2D((1, 2, 1), 1, 1),
        weight=(weight,),
        plasticity=rule,
    )
    pre_inputs = builder.inputs("pre_input", pre)
    post_inputs = builder.inputs("post_input", post)
    return builder, edges, pre_inputs, post_inputs


def test_convolution_expands_nhwc_kernel_and_reuses_coefficients() -> None:
    pattern = Convolution2D(
        input_shape=(4, 4, 1),
        output_channels=2,
        kernel_size=2,
        stride=2,
    )
    assert pattern.output_shape == (2, 2, 2)
    assert pattern.kernel_parameter_count == 8

    builder = NetworkBuilder("convolution-layout")
    source = builder.population("source", 16, LIF())
    target = builder.population("target", 8, LIF())
    edges = builder.connect(
        source,
        target,
        pattern=pattern,
        weight=tuple(range(1, 9)),
    )
    network = builder.build()
    graph = network.graph

    assert len(edges) == 32
    assert len({edge.weight_group for edge in graph.edges}) == 8
    for group in range(8):
        members = [edge for edge in graph.edges if edge.weight_group == group]
        assert len(members) == 4
        assert {edge.weight for edge in members} == {float(group + 1)}
    report = network.validate()
    assert report.edge_count == 32
    assert report.weight_parameter_count == 8
    assert report.shared_weight_group_count == 8


def test_static_shared_convolution_compiles_without_learning_state(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("static-convolution")
    source = builder.population("source", 16, LIF())
    target = builder.population("target", 8, LIF())
    builder.connect(
        source,
        target,
        pattern=Convolution2D(
            input_shape=(4, 4, 1),
            output_channels=2,
            kernel_size=2,
            stride=2,
        ),
        weight=tuple(range(1, 9)),
    )

    with Engine(core._lib._name).compile(builder.build()):
        pass


def test_pair_stdp_shares_weight_but_keeps_edge_local_traces(
    core: CoreEvaluator,
) -> None:
    rule = PairSTDP(
        tau_pre=10.0,
        tau_post=10.0,
        a_plus=1.0,
        a_minus=0.0,
        learning_rate=1.0,
        bounds=(0.0, 1.0),
    )
    builder, edges, pre_inputs, post_inputs = _two_copy_projection(rule)
    network = builder.build()
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            4.0,
            inputs={
                pre_inputs[0]: SpikeTrain((1.0,), 20.0),
                post_inputs[0]: SpikeTrain((2.0,), 20.0),
            },
        )

    expected = 0.5 + 0.5 * 0.5 * math.exp(-0.1)
    assert result.weights[edges[0]] == pytest.approx(expected, abs=1e-14)
    assert result.weights[edges[1]] == pytest.approx(expected, abs=1e-14)
    assert result.plasticity[0].pre_fast == pytest.approx(math.exp(-0.1))
    assert result.plasticity[1].pre_fast == 0.0
    assert result.plasticity[1].post_fast == 0.0


def test_triplet_stdp_updates_one_shared_kernel_weight(core: CoreEvaluator) -> None:
    rule = TripletSTDP(
        tau_plus=10.0,
        tau_minus=10.0,
        tau_x=20.0,
        tau_y=20.0,
        a2_plus=0.1,
        a2_minus=0.0,
        a3_plus=0.1,
        a3_minus=0.0,
        learning_rate=1.0,
        bounds=(0.0, 1.0),
    )
    builder, edges, pre_inputs, post_inputs = _two_copy_projection(rule)
    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            5.0,
            inputs={
                pre_inputs[0]: SpikeTrain((1.0, 3.0), 20.0),
                post_inputs[0]: SpikeTrain((2.0, 4.0), 20.0),
            },
        )

    assert result.weights[edges[0]] > 0.5
    assert result.weights[edges[0]] == result.weights[edges[1]]
    assert result.plasticity[1].pre_fast == 0.0


def test_modulated_stdp_shares_weight_with_local_eligibility(
    core: CoreEvaluator,
) -> None:
    rule = ModulatedSTDP(
        tau_pre=10.0,
        tau_post=10.0,
        tau_eligibility_plus=20.0,
        tau_eligibility_minus=20.0,
        learning_rate=1.0,
        bounds=(0.0, 1.0),
    )
    builder, edges, pre_inputs, post_inputs = _two_copy_projection(rule)
    reward = builder.modulator("reward", targets=edges)
    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                pre_inputs[0]: SpikeTrain((1.0,), 20.0),
                post_inputs[0]: SpikeTrain((2.0,), 20.0),
                reward: ModulationSeries((4.0,), (1.0,)),
            },
        )

    assert result.weights[edges[0]] > 0.5
    assert result.weights[edges[0]] == result.weights[edges[1]]
    assert result.plasticity[0].eligibility_plus == 0.0
    assert result.plasticity[1].pre_fast == 0.0


def test_shared_weight_accepts_neuron_local_modulators_and_averages_copies(
    core: CoreEvaluator,
) -> None:
    rule = ModulatedSTDP(
        tau_pre=10.0,
        tau_post=10.0,
        tau_eligibility_plus=20.0,
        tau_eligibility_minus=20.0,
        learning_rate=1.0,
        bounds=(0.0, 1.0),
    )
    builder, edges, pre_inputs, post_inputs = _two_copy_projection(rule)
    first_reward = builder.modulator("first_reward", targets=(edges[0],))
    builder.modulator("second_reward", targets=(edges[1],))
    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                pre_inputs[0]: SpikeTrain((1.0,), 20.0),
                post_inputs[0]: SpikeTrain((2.0,), 20.0),
                first_reward: ModulationSeries((4.0,), (1.0,)),
            },
        )

    expected = 0.5 + 0.5 * math.exp(-0.2)
    assert result.weights[edges[0]] == pytest.approx(expected, abs=1.0e-14)
    assert result.weights[edges[1]] == pytest.approx(expected, abs=1.0e-14)
    assert result.plasticity[0].eligibility_plus == 0.0
    assert result.plasticity[1].eligibility_plus == 0.0


def test_simultaneous_local_modulation_clamps_shared_weight_after_reduction(
    core: CoreEvaluator,
) -> None:
    rule = ModulatedSTDP(
        tau_pre=10.0,
        tau_post=10.0,
        tau_eligibility_plus=20.0,
        tau_eligibility_minus=20.0,
        learning_rate=1.0,
        bounds=(0.0, 1.0),
    )
    builder, edges, pre_inputs, post_inputs = _two_copy_projection(
        rule, weight=0.95
    )
    positive = builder.modulator("positive", targets=(edges[0],))
    negative = builder.modulator("negative", targets=(edges[1],))
    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                pre_inputs[0]: SpikeTrain((1.0,), 20.0),
                pre_inputs[1]: SpikeTrain((1.0,), 20.0),
                post_inputs[0]: SpikeTrain((2.0,), 20.0),
                post_inputs[1]: SpikeTrain((2.0,), 20.0),
                positive: ModulationSeries((4.0,), (1.0,)),
                negative: ModulationSeries((4.0,), (-1.0,)),
            },
        )

    assert result.weights[edges[0]] == pytest.approx(0.95, abs=1.0e-14)
    assert result.weights[edges[1]] == pytest.approx(0.95, abs=1.0e-14)


def test_shared_groups_round_trip_and_learned_snapshot(
    core: CoreEvaluator, tmp_path
) -> None:
    builder, edges, pre_inputs, post_inputs = _two_copy_projection(PairSTDP())
    network = builder.build()
    restored = Network.from_text(network.to_text())
    assert restored.to_text() == network.to_text()
    assert [edge.weight_group for edge in restored.graph.edges] == [0, 0]

    with Engine(core._lib._name).compile(restored) as simulation:
        result = simulation.run(
            4.0,
            inputs={
                pre_inputs[0]: SpikeTrain((1.0,), 20.0),
                post_inputs[0]: SpikeTrain((2.0,), 20.0),
            },
        )
    destination = tmp_path / "shared-learned.json"
    snapshot = result.save_learned_network(destination)
    loaded = Network.load(destination)
    assert snapshot.graph.edges[edges[0]].weight == result.weights[edges[0]]
    assert snapshot.graph.edges[edges[1]].weight == result.weights[edges[1]]
    assert [edge.weight_group for edge in loaded.graph.edges] == [0, 0]


def test_shared_group_rejects_divergent_member_definition() -> None:
    builder, _, _, _ = _two_copy_projection(PairSTDP())
    network = builder.build()
    invalid = replace(
        network.graph,
        edges=(network.graph.edges[0], replace(network.graph.edges[1], weight=0.6)),
    )
    with pytest.raises(ResolutionError, match="weight group 0"):
        invalid.resolve()
