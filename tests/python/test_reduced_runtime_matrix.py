"""Native reduced-profile execution, learning, reset, callback, and image parity."""

from dataclasses import replace
from pathlib import Path

import pytest

from lacuna import (
    CoreEvaluator, DecoderBinding, Engine, LIF, MixedDriveUpdate, MixedInputSpike, ModulatedSTDP, ModulationEvent,
    ModulationInput, Network, NetworkBuilder, PairSTDP, PrecisionProfile,
    RecordingConfig, SoftExcursionModulated, SpikeInput, StateInspectionRequest,
    TTFSDecoder, TripletSTDP, VoltageModulatedSTDP,
)
from lacuna.errors import CoreError, PrecisionResolutionError

from .test_execution_plan_matrix import PLAN_CASES, _matrix_inputs


@pytest.fixture(params=(PrecisionProfile.FLOAT32, PrecisionProfile.FLOAT32_TIME64),
                ids=lambda profile: profile.value)
def reduced_engine(request):
    profile = request.param
    libraries = sorted(Path("build-" + profile.value).glob("liblacuna_core.*"))
    if not libraries:
        pytest.skip(f"native {profile.value} build is not available")
    return Engine(libraries[0], precision=profile)


@pytest.mark.parametrize("name,factory", PLAN_CASES, ids=[name for name, _ in PLAN_CASES])
def test_reduced_model_matrix_replays_through_callbacks_and_compiled_images(
    reduced_engine, name, factory,
):
    network = Network(factory(), name=name)
    original = network.to_text()
    with reduced_engine.compile(network) as high_level:
        resolved = high_level.compiled.resolved
        plan = high_level.execution_plan
        compiled = high_level.compiled._ensure_compiled_scheduler()
        inputs = _matrix_inputs(resolved)
        modulations = tuple(
            ModulationEvent(1.5, index, 0.75) for index in range(len(plan.modulators))
        )
        expected = compiled.run(
            resolved.initial_values, inputs=inputs, modulations=modulations,
            t_end=5.0, recording=RecordingConfig(capacity=1024),
        )
        observed = []
        replay = compiled.run(
            resolved.initial_values, inputs=inputs, modulations=modulations,
            t_end=5.0, recording=RecordingConfig(consumer=observed.append),
        )
        assert replace(replay, trace=tuple(observed)) == expected
        image = compiled.to_bytes()
        with reduced_engine.core.load_compiled_graph_image(image, execution_plan=plan) as loaded:
            assert loaded.to_bytes() == image
            restored = loaded.run(
                resolved.initial_values, inputs=inputs, modulations=modulations,
                t_end=5.0, recording=RecordingConfig(capacity=1024),
            )
        assert restored == expected
        with pytest.raises(CoreError):
            CoreEvaluator().load_compiled_graph_image(image)
    assert network.to_text() == original


_LEARNING_RULES = (
    ("pair", PairSTDP()),
    ("triplet", TripletSTDP()),
    ("modulated", ModulatedSTDP(consume_on_modulation=False)),
    ("voltage", VoltageModulatedSTDP(consume_on_modulation=False)),
    ("soft-excursion", SoftExcursionModulated(consume_on_modulation=False)),
)


@pytest.mark.parametrize("name,rule", _LEARNING_RULES, ids=[name for name, _ in _LEARNING_RULES])
def test_reduced_learning_changes_native_weights_and_episode_reset_preserves_them(
    reduced_engine, name, rule,
):
    builder = NetworkBuilder(name)
    pre = builder.neuron("pre", LIF(name="pre"))
    post = builder.neuron("post", LIF(name="post", v_rest=-54.0, v_reset=-54.0))
    edges = builder.connect(pre, post, weight=0.5, delay=0.5, plasticity=rule)
    pre_input = builder.input("pre", pre)
    post_input = builder.input("post", post)
    reward = None
    if not isinstance(rule, (PairSTDP, TripletSTDP)):
        reward = builder.modulator("reward", targets=edges)
    network = builder.build()
    with reduced_engine.compile(network) as compiled:
        with compiled.compiled.create_incremental_run(t_end=12.0) as run:
            first = run.advance_until(
                8.0,
                spike_inputs=(
                    SpikeInput(1.0, pre_input, 20.0),
                    SpikeInput(2.5, post_input, 20.0),
                    SpikeInput(5.0, pre_input, 20.0),
                    SpikeInput(6.5, post_input, 20.0),
                ),
                modulation_inputs=(
                    () if reward is None else (ModulationInput(7.5, reward, 1.0),)
                ),
            )
            assert first.core.weights[0] != 0.5
            assert first.core.plasticity
            assert first.core.weights[0] == reduced_engine.precision.round_real(first.core.weights[0])
            run.reset_episode()
            reset = run.advance_until(9.0)
            assert reset.core.weights == first.core.weights
            for state in reset.core.plasticity:
                assert (state.pre_fast, state.post_fast, state.pre_slow, state.post_slow,
                        state.eligibility_plus, state.eligibility_minus) == (0.0,) * 6
            run.finish()


def test_reduced_validation_failure_does_not_queue_modulations(reduced_engine, monkeypatch):
    builder = NetworkBuilder("validation-order")
    pre = builder.neuron("pre", LIF(name="pre"))
    post = builder.neuron("post", LIF(name="post"))
    edges = builder.connect(pre, post, weight=0.5, plasticity=ModulatedSTDP())
    builder.modulator("reward", targets=edges)
    with reduced_engine.compile(builder.build()) as compiled:
        scheduler = compiled.compiled._ensure_compiled_scheduler()
        initial = compiled.compiled.resolved.initial_values
        with scheduler.create_run(initial) as run:
            original = reduced_engine.core._lib.lc_mixed_run_schedule_modulations
            monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_schedule_modulations",
                                lambda *args: pytest.fail("validation mutated native state"))
            with pytest.raises(PrecisionResolutionError):
                run.execute(
                    inputs=(MixedInputSpike(1.0, 0, 1e-50),),
                    modulations=(ModulationEvent(2.0, 0, 1.0),), t_end=3.0,
                )
            monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_schedule_modulations", original)
            assert run.execute(t_end=3.0).weights == (0.5,)


def test_reduced_runtime_rejects_foreign_precision_decoder_handles(reduced_engine, monkeypatch):
    builder = NetworkBuilder("decoder-profile")
    builder.neuron("node", LIF())
    with reduced_engine.compile(builder.build()) as compiled:
        scheduler = compiled.compiled._ensure_compiled_scheduler()
        initial = compiled.compiled.resolved.initial_values
        with CoreEvaluator().compile_decoders((DecoderBinding(0, TTFSDecoder()),),
                                             node_count=1) as decoders:
            with decoders.create_run(t_start=0.0, t_end=3.0) as decoder_run:
                with scheduler.create_run(initial) as run:
                    with pytest.raises(CoreError, match="decoder precision"):
                        run.execute(t_end=3.0, decoder_run=decoder_run)
                monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_create",
                                    lambda *args: pytest.fail("foreign handle allocated a run"))
                with pytest.raises(CoreError, match="decoder precision"):
                    scheduler.create_incremental_run(initial, t_end=3.0,
                                                     decoder_run=decoder_run)


def test_reduced_invalid_horizon_fails_before_native_allocation(reduced_engine, monkeypatch):
    builder = NetworkBuilder("horizon-validation")
    builder.neuron("node", LIF())
    with reduced_engine.compile(builder.build()) as compiled:
        scheduler = compiled.compiled._ensure_compiled_scheduler()
        initial = compiled.compiled.resolved.initial_values
        monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_create",
                            lambda *args: pytest.fail("invalid horizon allocated a run"))
        horizon = 1e39 if reduced_engine.precision is PrecisionProfile.FLOAT32 else float("inf")
        with pytest.raises((PrecisionResolutionError, ValueError)):
            scheduler.create_incremental_run(initial, t_end=horizon)


def test_reduced_incremental_validation_does_not_queue_modulations(reduced_engine, monkeypatch):
    builder = NetworkBuilder("incremental-validation")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edges = builder.connect(pre, post, weight=0.5, plasticity=ModulatedSTDP())
    builder.modulator("reward", targets=edges)
    with reduced_engine.compile(builder.build()) as compiled:
        scheduler = compiled.compiled._ensure_compiled_scheduler()
        initial = compiled.compiled.resolved.initial_values
        with scheduler.create_incremental_run(initial, t_end=3.0) as run:
            original = reduced_engine.core._lib.lc_mixed_run_schedule_modulations
            monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_schedule_modulations",
                                lambda *args: pytest.fail("validation mutated native state"))
            with pytest.raises(PrecisionResolutionError):
                run.advance_until(
                    2.0, inputs=(MixedInputSpike(1.0, 0, 1e-50),),
                    modulations=(ModulationEvent(1.5, 0, 1.0),),
                )
            monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_schedule_modulations", original)
            assert run.finish().weights == (0.5,)


@pytest.mark.parametrize("kind", ("event", "drive", "modulation", "inspection"))
def test_incremental_boundaries_are_validated_after_target_rounding(
    reduced_engine, monkeypatch, kind,
):
    builder = NetworkBuilder("rounded-boundary")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edges = builder.connect(pre, post, weight=0.5, plasticity=ModulatedSTDP())
    builder.modulator("reward", targets=edges)
    event_time = 0.99999999 if reduced_engine.precision is PrecisionProfile.FLOAT32 else 1.0
    submissions = {
        "event": {"inputs": (MixedInputSpike(event_time, 0, 1.0),)},
        "drive": {"drive_updates": (MixedDriveUpdate(event_time, 0, 1.0),)},
        "modulation": {"modulations": (ModulationEvent(event_time, 0, 1.0),)},
        "inspection": {"inspections": (StateInspectionRequest(event_time, 0, (0,)),)},
    }
    with reduced_engine.compile(builder.build()) as compiled:
        scheduler = compiled.compiled._ensure_compiled_scheduler()
        initial = compiled.compiled.resolved.initial_values
        with scheduler.create_incremental_run(initial, t_end=2.0) as run:
            original = reduced_engine.core._lib.lc_mixed_run_schedule_modulations
            monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_schedule_modulations",
                                lambda *args: pytest.fail("invalid boundary queued a modulation"))
            arguments = {"modulations": (ModulationEvent(0.5, 0, 1.0),), **submissions[kind]}
            with pytest.raises(ValueError, match="outside"):
                run.advance_until(1.0, **arguments)
            assert not run._failed
            assert run.frontier == 0.0
            monkeypatch.setattr(reduced_engine.core._lib, "lc_mixed_run_schedule_modulations", original)
            assert run.advance_until(1.0).weights == (0.5,)
            assert run.finish().weights == (0.5,)
