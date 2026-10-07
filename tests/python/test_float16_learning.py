"""Native half learning, checkpoint provenance, and exact image replay."""

from dataclasses import replace
from pathlib import Path

import pytest

from lacuna import (
    Engine, LIF, MixedInputSpike, ModulatedSTDP, ModulationEvent, ModulationInput,
    Network, NetworkBuilder, PairSTDP, PrecisionProfile, RecordingConfig,
    SpikeInput, TripletSTDP,
)
from lacuna.errors import PrecisionResolutionError, ResolutionError
from lacuna.expr import ExprNode, ExprOp
from lacuna.learning import resolve_learning
import lacuna.target_lowering as target_lowering


@pytest.fixture(scope="module")
def half_engine():
    if not list(Path("build-float16").glob("liblacuna_half_host.*")):
        pytest.skip("native half runtime is not built")
    return Engine(precision=PrecisionProfile.FLOAT16)


def _mixed_learning_network():
    rules = (
        None,
        PairSTDP(tau_pre=8.0, tau_post=8.0, a_plus=0.5, a_minus=0.25,
                 learning_rate=0.125),
        TripletSTDP(tau_plus=8.0, tau_minus=16.0, tau_x=32.0, tau_y=16.0,
                    a2_plus=0.0625, a2_minus=0.015625,
                    a3_plus=0.0625, a3_minus=0.015625),
        ModulatedSTDP(tau_pre=8.0, tau_post=8.0,
                      tau_eligibility_plus=32.0, tau_eligibility_minus=32.0,
                      learning_rate=0.125, bounds=(0.0, 1.0)),
        ModulatedSTDP(tau_pre=8.0, tau_post=8.0,
                      tau_eligibility_plus=32.0, tau_eligibility_minus=32.0,
                      learning_rate=0.125, bounds=(0.0, 1.0),
                      consume_on_modulation=False),
    )
    builder = NetworkBuilder("half-learning", metadata={"purpose": "native smoke"})
    pre = builder.neuron("pre", LIF(name="pre"))
    pre_input = builder.input("pre", pre)
    spike_inputs = [SpikeInput(t, pre_input, 20.0) for t in (1.0, 5.0)]
    modulation_inputs = []
    for index, rule in enumerate(rules):
        name = f"post-{index}"
        post = builder.neuron(name, LIF(name=name, v_rest=-54.0, v_reset=-54.0))
        edges = builder.connect(pre, post, weight=0.5, delay=0.5, plasticity=rule)
        post_input = builder.input(name, post)
        spike_inputs.extend(SpikeInput(t, post_input, 20.0) for t in (2.5, 6.5))
        if isinstance(rule, ModulatedSTDP):
            reward = builder.modulator(f"reward-{index}", targets=edges)
            modulation_inputs.append(ModulationInput(7.5, reward, 1.0))
    return builder.build(), tuple(spike_inputs), tuple(modulation_inputs)


def _native_inputs():
    spikes = tuple(MixedInputSpike(t, node, 20.0)
                   for node in range(6)
                   for t in ((1.0, 5.0) if node == 0 else (2.5, 6.5)))
    rewards = tuple(ModulationEvent(7.5, index, 1.0) for index in range(2))
    return spikes, rewards


def test_half_mixed_learning_trains_saves_reloads_and_replays_images(half_engine, tmp_path):
    network, inputs, rewards = _mixed_learning_network()
    original = network.to_text()
    with half_engine.compile(network) as compiled:
        result = compiled.run(8.0, spike_inputs=inputs, modulation_inputs=rewards)
        assert result.precision is PrecisionProfile.FLOAT16
        assert result.weights[0] == 0.5
        assert all(weight != 0.5 for weight in result.weights[1:])
        assert all(0.0 < weight < 1.0 for weight in result.weights)
        assert all(half_engine.precision.round_real(weight) == weight
                   for weight in result.weights)
        assert len(result.raw.core.plasticity) == 4
        assert result.raw.core.spikes
        assert compiled.run(8.0, spike_inputs=inputs, modulation_inputs=rewards) == result
    assert network.to_text() == original

    checkpoint = tmp_path / "learned-half.json"
    learned = result.save_learned_network(checkpoint)
    reloaded = Network.load(checkpoint)
    assert reloaded.to_text() == learned.to_text()
    assert reloaded.metadata == {
        "purpose": "native smoke", "lacuna_precision": PrecisionProfile.FLOAT16.to_record(),
    }
    assert tuple(edge.weight for edge in reloaded.graph.edges) == result.weights
    with pytest.raises(ResolutionError, match="does not match engine precision"):
        Engine().compile(reloaded)

    native_inputs, native_rewards = _native_inputs()
    with half_engine.compile(reloaded) as resumed:
        assert resumed.run(0.0).weights == result.weights
        scheduler = resumed.compiled._ensure_compiled_scheduler()
        initial = resumed.compiled.resolved.initial_values
        arguments = dict(inputs=native_inputs, modulations=native_rewards, t_end=8.0)
        expected = scheduler.run(initial, recording=RecordingConfig(capacity=1024), **arguments)
        assert expected.trace
        assert all(after != before for before, after in
                   zip(result.weights[1:], expected.weights[1:]))
        observed = []
        replay = scheduler.run(initial, recording=RecordingConfig(consumer=observed.append),
                               **arguments)
        assert replace(replay, trace=tuple(observed)) == expected
        image = scheduler.to_bytes()
        with half_engine.core.load_compiled_graph_image(
            image, execution_plan=resumed.execution_plan,
        ) as loaded:
            assert loaded.to_bytes() == image
            assert loaded.run(initial, t_end=0.0).weights == result.weights
            restored = loaded.run(initial, recording=RecordingConfig(capacity=1024), **arguments)
            assert restored == expected


def test_half_episode_reset_preserves_learned_weights_and_clears_traces(half_engine):
    network, inputs, rewards = _mixed_learning_network()
    with half_engine.compile(network) as compiled:
        with compiled.compiled.create_incremental_run(t_end=12.0) as run:
            trained = run.advance_until(8.0, spike_inputs=inputs, modulation_inputs=rewards)
            assert all(weight != 0.5 for weight in trained.core.weights[1:])
            assert any(state.pre_fast != 0.0 for state in trained.core.plasticity)
            run.reset_episode()
            reset = run.advance_until(9.0)
            assert reset.core.weights == trained.core.weights
            for state in reset.core.plasticity:
                assert (state.pre_fast, state.post_fast, state.pre_slow, state.post_slow,
                        state.eligibility_plus, state.eligibility_minus) == (0.0,) * 6
            run.finish()


@pytest.mark.parametrize("dynamic", (False, True), ids=("unsupported-fixed", "state-dependent"))
def test_half_learning_rejects_unsupported_powers_before_native_compile(
    half_engine, monkeypatch, dynamic,
):
    learning = resolve_learning(PairSTDP())
    event = learning.program.events[0]
    dag = event.expressions
    exponent = ExprNode(ExprOp.VAR, binding=0) if dynamic else ExprNode(ExprOp.CONST, value=0.25)
    index = len(dag.nodes)
    modified = replace(dag, nodes=(*dag.nodes, ExprNode(ExprOp.CONST, value=1.0), exponent,
                                   ExprNode(ExprOp.POW, lhs=index, rhs=index + 1)),
                       roots={**dag.roots, "unsupported_power": index + 2})
    learning = replace(learning, program=replace(
        learning.program, events=(replace(event, expressions=modified), *learning.program.events[1:]),
    ))
    monkeypatch.setattr(target_lowering, "resolve_learning", lambda rule: learning)
    monkeypatch.setattr(half_engine.core, "compile_execution_plan",
                        lambda *args, **kwargs: pytest.fail("invalid learning allocated native graph"))
    builder = NetworkBuilder("invalid-half-learning-power")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    builder.connect(pre, post, weight=0.5, plasticity=PairSTDP())
    message = "parameter-only fixed exponent" if dynamic else "unsupported fixed exponent"
    with pytest.raises(PrecisionResolutionError, match=message):
        half_engine.compile(builder.build())
