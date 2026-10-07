import math

import pytest

from lacuna import (
    AdEx,
    AlphaCurrent,
    AdaptiveLIF,
    Engine,
    LIF,
    ModulatedSTDP,
    ModulationInput,
    ModulationSeries,
    Network,
    NetworkBuilder,
    PairSTDP,
    RecordingPlan,
    SoftExcursionModulated,
    SpikeTrain,
    SpikeInput,
    TripletSTDP,
    VoltageModulatedSTDP,
    TraceKind,
    TraceRecording,
    audit_causal_trace,
)
from lacuna.errors import CapabilityError, ResolutionError
from lacuna.ffi import CoreEvaluator


def _pair_network(*, delay: float = 0.0):
    builder = NetworkBuilder("pair")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edge = builder.connect(
        pre,
        post,
        weight=0.5,
        delay=delay,
        plasticity=PairSTDP(),
    )[0]
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)
    return builder.build(), edge, pre_input, post_input


def test_pair_stdp_uses_presynaptic_arrival_time_and_exact_trace_decay(
    core: CoreEvaluator,
) -> None:
    network, edge, pre_input, post_input = _pair_network(delay=2.0)
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            8.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                post_input: SpikeTrain((5.0,), 20.0),
            },
        )

    # The pre trace starts at synaptic arrival t=3, not source firing t=1.
    pre_trace = math.exp(-(5.0 - 3.0) / 5.0)
    expected = 0.5 + 0.05 * 0.6 * (1.0 - 0.5) * pre_trace
    assert result.weights[edge] == pytest.approx(expected, abs=1e-14)
    assert result.plasticity[0].pre_fast == pytest.approx(pre_trace, abs=1e-14)
    assert result.plasticity[0].post_fast == 1.0


def test_pair_stdp_depresses_post_before_pre_pair(core: CoreEvaluator) -> None:
    network, edge, pre_input, post_input = _pair_network()
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                post_input: SpikeTrain((1.0,), 20.0),
                pre_input: SpikeTrain((3.0,), 20.0),
            },
        )

    post_trace = math.exp(-2.0 / 5.0)
    expected = 0.5 - 0.05 * 0.3 * 0.5 * post_trace
    assert result.weights[edge] == pytest.approx(expected, abs=1e-14)


def test_triplet_rule_uses_slow_trace_from_prior_post_spike(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("triplet")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    rule = TripletSTDP(
        tau_plus=10.0,
        tau_minus=10.0,
        tau_x=20.0,
        tau_y=20.0,
        a2_plus=0.01,
        a2_minus=0.0,
        a3_plus=0.02,
        a3_minus=0.0,
        learning_rate=1.0,
    )
    edge = builder.connect(pre, post, weight=0.5, plasticity=rule)[0]
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)
    network = builder.build()
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                pre_input: SpikeTrain((1.0, 3.0), 20.0),
                post_input: SpikeTrain((2.0, 4.0), 20.0),
            },
        )

    first_pre_at_post = math.exp(-1.0 / 10.0)
    second_pre_at_post = first_pre_at_post * math.exp(-2.0 / 10.0) + math.exp(
        -1.0 / 10.0
    )
    prior_post_slow = math.exp(-(4.0 - 2.0) / 20.0)
    expected = (
        0.5
        + first_pre_at_post * 0.01
        + second_pre_at_post * (0.01 + 0.02 * prior_post_slow)
    )
    assert result.weights[edge] == pytest.approx(expected, abs=1e-14)
    assert result.plasticity[0].kind == "TRIPLET_STDP"


def test_mixed_static_pair_triplet_and_modulated_edges(core: CoreEvaluator) -> None:
    builder = NetworkBuilder("mixed-plasticity")
    pre = builder.neuron("pre", LIF())
    posts = builder.population("post", 4, LIF())
    static = builder.connect(pre, posts[0], weight=0.5)[0]
    pair = builder.connect(pre, posts[1], weight=0.5, plasticity=PairSTDP())[0]
    triplet = builder.connect(
        pre, posts[2], weight=0.5, plasticity=TripletSTDP.hippocampus()
    )[0]
    modulated = builder.connect(
        pre, posts[3], weight=0.5, plasticity=ModulatedSTDP()
    )[0]
    reward = builder.modulator("reward", targets=(modulated,))
    pre_input = builder.input("pre_input", pre)
    post_inputs = builder.inputs("post_input", posts)
    network = builder.build()

    inputs = {pre_input: SpikeTrain((1.0,), 20.0)}
    inputs.update(
        {
            port: SpikeTrain((2.0,), 20.0)
            for port in post_inputs
        }
    )
    inputs[reward] = ModulationSeries((4.0,), (1.0,))
    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(6.0, inputs=inputs)

    assert result.weights[static] == 0.5
    assert result.weights[pair] > 0.5
    assert result.weights[triplet] > 0.5
    assert result.weights[modulated] > 0.5
    assert tuple(state.edge for state in result.plasticity) == (
        pair,
        triplet,
        modulated,
    )
    assert tuple(state.kind for state in result.plasticity) == (
        "PAIR_STDP",
        "TRIPLET_STDP",
        "MODULATED_STDP",
    )


def test_plasticity_is_orthogonal_to_filtered_synapse_execution(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("plastic-alpha")
    pre = builder.neuron("pre", LIF(name="pre_lif"))
    post = builder.neuron("post", LIF(name="post_lif", synaptic_input=True))
    edge = builder.connect(
        pre,
        post,
        synapse=AlphaCurrent(5.0),
        weight=0.5,
        plasticity=PairSTDP(),
    )[0]
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)
    network = builder.build()

    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                post_input: SpikeTrain((2.0,), 20.0),
            },
        )

    assert result.weights[edge] > 0.5
    assert result.plasticity[0].kind == "PAIR_STDP"
    assert len(result.final_states[post.id].values) == 3


def test_modulators_are_edge_scoped_and_can_be_streamed_incrementally(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("scoped-reward")
    pre = builder.neuron("pre", LIF())
    posts = builder.population("post", 2, LIF())
    first = builder.connect(
        pre, posts[0], weight=0.5, plasticity=ModulatedSTDP()
    )[0]
    second = builder.connect(
        pre, posts[1], weight=0.5, plasticity=ModulatedSTDP()
    )[0]
    reward = builder.modulator("reward", targets=(first,))
    builder.modulator("other_reward", targets=(second,))
    pre_input = builder.input("pre_input", pre)
    post_inputs = builder.inputs("post_input", posts)
    network = builder.build()

    with Engine(core._lib._name).compile(network) as simulation:
        with simulation.start_run(6.0) as run:
            run.advance(
                3.0,
                inputs={
                    pre_input: SpikeTrain((1.0,), 20.0),
                    post_inputs[0]: SpikeTrain((2.0,), 20.0),
                    post_inputs[1]: SpikeTrain((2.0,), 20.0),
                },
            )
            result = run.finish(
                inputs={reward: ModulationSeries((4.0,), (1.0,))}
            )

    assert result.weights[first] > 0.5
    assert result.weights[second] == 0.5
    assert result.plasticity[0].eligibility_plus == 0.0
    assert result.plasticity[1].eligibility_plus > 0.0


def test_voltage_modulation_supplies_eligibility_without_postsynaptic_spike(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("voltage-eligibility-without-post-spike")
    pre = builder.neuron("pre", LIF())
    spike_post = builder.neuron("spike_post", LIF())
    voltage_post = builder.neuron("voltage_post", LIF())
    spike_edge = builder.connect(
        pre,
        spike_post,
        weight=0.5,
        plasticity=ModulatedSTDP(
            tau_pre=20.0,
            tau_post=20.0,
            tau_eligibility_plus=80.0,
            tau_eligibility_minus=80.0,
            learning_rate=0.1,
            bounds=(0.0, 1.0),
            consume_on_modulation=False,
        ),
    )[0]
    voltage_edge = builder.connect(
        pre,
        voltage_post,
        weight=0.5,
        plasticity=VoltageModulatedSTDP(
            spike_scale=0.0,
            voltage_scale=1.0,
            surrogate_threshold=-60.0,
            surrogate_slope=3.0,
            learning_rate=0.1,
            bounds=(0.0, 1.0),
            consume_on_modulation=False,
        ),
    )[0]
    builder.modulator("spike_reward", targets=(spike_edge,))
    voltage_reward = builder.modulator("voltage_reward", targets=(voltage_edge,))
    pre_input = builder.input("pre_input", pre)
    network = builder.build()

    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            3.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                voltage_reward: ModulationSeries((2.0,), (1.0,)),
            },
        )

    assert result.weights[spike_edge] == 0.5
    assert result.weights[voltage_edge] > 0.5
    voltage_state = next(
        state for state in result.plasticity if state.edge == voltage_edge
    )
    assert voltage_state.voltage_eligibility > 0.0


def test_voltage_eligibility_reads_exact_target_state_at_delayed_delivery(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("voltage-eligibility-current-target-state")
    pre = builder.neuron("pre", LIF())
    resting_post = builder.neuron("resting_post", LIF())
    depolarized_post = builder.neuron("depolarized_post", LIF())
    rule = VoltageModulatedSTDP(
        spike_scale=0.0,
        voltage_scale=1.0,
        surrogate_threshold=-65.0,
        surrogate_slope=1.0,
        learning_rate=0.1,
        bounds=(0.0, 1.0),
        consume_on_modulation=False,
    )
    resting_edge = builder.connect(
        pre, resting_post, weight=0.1, delay=1.0, plasticity=rule
    )[0]
    depolarized_edge = builder.connect(
        pre, depolarized_post, weight=0.1, delay=1.0, plasticity=rule
    )[0]
    resting_reward = builder.modulator("resting_reward", targets=(resting_edge,))
    depolarized_reward = builder.modulator(
        "depolarized_reward", targets=(depolarized_edge,)
    )
    pre_input = builder.input("pre_input", pre)
    depolarizing_input = builder.input("depolarizing_input", depolarized_post)

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            4.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                depolarizing_input: SpikeTrain((0.5,), 2.0),
                resting_reward: ModulationSeries((3.0,), (1.0,)),
                depolarized_reward: ModulationSeries((3.0,), (1.0,)),
            },
        )

    assert result.weights[depolarized_edge] > result.weights[resting_edge]


def test_soft_excursion_learning_needs_no_postsynaptic_spike(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("soft-excursion-without-post-spike")
    pre = builder.neuron("pre", LIF(name="soft_pre_lif"))
    post = builder.neuron(
        "post", LIF(name="soft_post_lif", v_threshold=-60.0)
    )
    edge = builder.connect(
        pre,
        post,
        weight=0.1,
        plasticity=SoftExcursionModulated(
            threshold=-60.0,
            proximity_width=5.0,
            proximity_slope=0.5,
            excursion_smoothing=0.01,
            soft_scale=1.0,
            learning_rate=0.1,
            bounds=(0.0, 1.0),
            consume_on_modulation=False,
        ),
    )[0]
    reward = builder.modulator("reward", targets=(edge,))
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            4.0,
            inputs={
                pre_input: SpikeTrain((0.25, 2.0), 20.0),
                post_input: SpikeTrain((1.5,), 4.0),
                reward: ModulationSeries((3.0,), (1.0,)),
            },
        )

    assert all(spike.node != post.id for spike in result.spikes)
    assert result.weights[edge] > 0.1
    state = next(item for item in result.plasticity if item.edge == edge)
    assert state.voltage_eligibility > 0.0


@pytest.mark.parametrize("modulation", (0.4, -0.4))
def test_zero_voltage_scale_matches_current_modulated_stdp(
    core: CoreEvaluator,
    modulation: float,
) -> None:
    builder = NetworkBuilder("voltage-zero-scale-equivalence")
    pre = builder.neuron("pre", LIF())
    posts = builder.population("post", 2, LIF())
    common = dict(
        tau_pre=20.0,
        tau_post=20.0,
        tau_eligibility_plus=80.0,
        tau_eligibility_minus=80.0,
        learning_rate=0.01,
        bounds=(0.0, 1.0),
        consume_on_modulation=False,
    )
    current_edge = builder.connect(
        pre, posts[0], weight=0.5, plasticity=ModulatedSTDP(**common)
    )[0]
    voltage_edge = builder.connect(
        pre,
        posts[1],
        weight=0.5,
        plasticity=VoltageModulatedSTDP(
            **common,
            spike_scale=1.0,
            voltage_scale=0.0,
        ),
    )[0]
    current_reward = builder.modulator("current_reward", targets=(current_edge,))
    voltage_reward = builder.modulator("voltage_reward", targets=(voltage_edge,))
    pre_input = builder.input("pre_input", pre)
    post_inputs = builder.inputs("post_input", posts)

    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            5.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                post_inputs[0]: SpikeTrain((2.0,), 20.0),
                post_inputs[1]: SpikeTrain((2.0,), 20.0),
                current_reward: ModulationSeries((4.0,), (modulation,)),
                voltage_reward: ModulationSeries((4.0,), (modulation,)),
            },
        )

    assert result.weights[voltage_edge] == pytest.approx(
        result.weights[current_edge], abs=1.0e-15
    )


def test_voltage_modulated_rule_round_trips_through_network_schema() -> None:
    builder = NetworkBuilder("persisted-voltage-learning")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edge = builder.connect(
        pre,
        post,
        weight=0.5,
        plasticity=VoltageModulatedSTDP(
            surrogate_threshold=-59.0,
            surrogate_slope=2.5,
            spike_scale=0.25,
            voltage_scale=1.75,
        ),
    )[0]
    builder.modulator("reward", targets=(edge,))
    network = builder.build()

    restored = Network.from_text(network.to_text())

    assert restored.to_text() == network.to_text()
    assert restored.graph.edges[edge].plasticity == (
        network.graph.edges[edge].plasticity
    )


def test_soft_excursion_rule_round_trips_through_network_schema() -> None:
    builder = NetworkBuilder("persisted-soft-excursion-learning")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edge = builder.connect(
        pre,
        post,
        weight=0.5,
        plasticity=SoftExcursionModulated(
            threshold=-59.0,
            proximity_width=4.5,
            proximity_slope=0.6,
            excursion_smoothing=0.02,
            fixed_baseline=-66.0,
            adaptive_baseline=False,
            use_proximity=False,
            soft_scale=1.75,
        ),
    )[0]
    builder.modulator("reward", targets=(edge,))
    network = builder.build()

    restored = Network.from_text(network.to_text())

    assert restored.to_text() == network.to_text()
    assert restored.graph.edges[edge].plasticity == (
        network.graph.edges[edge].plasticity
    )


def test_episode_reset_clears_state_and_traces_but_preserves_weight(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("episode-reset")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edge = builder.connect(
        pre,
        post,
        weight=0.5,
        plasticity=ModulatedSTDP(consume_on_modulation=False),
    )[0]
    reward = builder.modulator("reward", targets=(edge,))
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)
    network = builder.build()

    with network.graph.resolve().compile(core) as compiled:
        with compiled.create_incremental_run(t_end=8.0) as run:
            run.advance_until(
                3.0,
                spike_inputs=(
                    SpikeInput(1.0, pre_input, 20.0),
                    SpikeInput(2.0, post_input, 20.0),
                ),
            )
            learned = run.advance_until(
                5.0,
                modulation_inputs=(ModulationInput(4.0, reward, 1.0),),
            )
            assert learned.core.weights[edge] > 0.5
            assert learned.core.plasticity[0].eligibility_plus > 0.0

            run.reset_episode()
            reset = run.advance_until(6.0)

    assert reset.core.weights[edge] == pytest.approx(learned.core.weights[edge])
    assert reset.core.plasticity[0].pre_fast == 0.0
    assert reset.core.plasticity[0].post_fast == 0.0
    assert reset.core.plasticity[0].eligibility_plus == 0.0
    assert reset.core.plasticity[0].eligibility_minus == 0.0
    assert all(state.values[0] == pytest.approx(-65.0) for state in reset.core.states)


def test_plasticity_and_modulator_schema_round_trip_and_learned_snapshot(
    core: CoreEvaluator, tmp_path
) -> None:
    builder = NetworkBuilder("persisted-learning")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edges = builder.connect(
        pre, post, weight=0.5, plasticity=ModulatedSTDP()
    )
    reward = builder.modulator("reward", targets=edges)
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)
    network = builder.build()
    restored = Network.from_text(network.to_text())
    assert restored.to_text() == network.to_text()

    with Engine(core._lib._name).compile(restored) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                post_input: SpikeTrain((2.0,), 20.0),
                reward: ModulationSeries((4.0,), (1.0,)),
            },
        )
    destination = tmp_path / "learned.json"
    snapshot = result.save_learned_network(destination)
    assert snapshot.graph.edges[0].weight == result.weights[0]
    assert Network.load(destination).to_text() == snapshot.to_text()


def test_modulation_is_present_in_complete_causal_trace(core: CoreEvaluator) -> None:
    builder = NetworkBuilder("traced-learning")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edges = builder.connect(
        pre, post, weight=0.5, plasticity=ModulatedSTDP()
    )
    reward = builder.modulator("reward", targets=edges)
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)
    network = builder.build()
    resolved = network.graph.resolve()

    with Engine(core._lib._name).compile(network) as simulation:
        result = simulation.run(
            6.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                post_input: SpikeTrain((2.0,), 20.0),
                reward: ModulationSeries((4.0,), 1.0),
            },
            recording=RecordingPlan(trace=TraceRecording(capacity=128)),
        )

    modulation = [
        record for record in result.trace if record.kind is TraceKind.MODULATION
    ]
    assert len(modulation) == 1
    assert modulation[0].subject == 0
    assert modulation[0].value == 1.0
    report = audit_causal_trace(
        result.raw.core,
        models=resolved.models,
        edges=resolved.edges,
        polarities=resolved.polarities,
        t_end=6.0,
        expected_input_count=2,
    )
    assert report.modulation_records == 1


def test_modulated_edge_requires_exactly_one_modulator() -> None:
    builder = NetworkBuilder("missing-modulator")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    builder.connect(pre, post, weight=0.5, plasticity=ModulatedSTDP())
    with pytest.raises(ResolutionError, match="not targeted by a modulator"):
        builder.build()
