from __future__ import annotations

import math
from dataclasses import replace

import pytest

from lacuna import (
    AdaptiveEscapeLIF,
    CoreEvaluator,
    DriveInput,
    EscapeLIF,
    Graph,
    InputMode,
    InputPort,
    LIF,
    OutputPort,
    SpikeInput,
    parse_neuron,
)
from lacuna.errors import CapabilityError, DSLParseError
from lacuna.execution_plan import CrossingMethod
from lacuna.resolver import resolve_escape_lif
from lacuna.resolver import resolve_adaptive_escape_lif


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _exponential_draw(seed: int, node: int, draw: int) -> float:
    bits = _splitmix64(
        seed
        ^ ((node * 0xD1B54A32D192ED03) & ((1 << 64) - 1))
        ^ ((draw * 0x94D049BB133111EB) & ((1 << 64) - 1))
    )
    uniform = ((bits >> 11) + 0.5) * 2.0**-53
    return -math.log(uniform)


def _single_node(model, *, with_input: bool = False):
    return Graph(
        models=(model.model,),
        nodes=(model.node(0),),
        input_ports=(InputPort("in", 0, InputMode.SPIKE),) if with_input else (),
        output_ports=(OutputPort("out", 0),),
    ).resolve()


def test_escape_dsl_requires_hazard_or_threshold_but_not_both():
    source = EscapeLIF().source
    assert parse_neuron(source).hazard is not None
    with pytest.raises(DSLParseError, match="exactly one threshold or hazard"):
        parse_neuron(source.replace(
            "hazard { rate = escape_rate*exp((v-v_escape)/delta_v) }",
            "hazard { rate = escape_rate*exp((v-v_escape)/delta_v) }\n"
            "threshold { v > -40.0 }",
        ))


def test_escape_resolver_recognizes_equivalent_exponential_voltage_hazard():
    model = EscapeLIF(escape_rate=0.25, v_escape=-60.0, delta_v=4.0)
    resolved = resolve_escape_lif(model.authored_model)
    assert resolved.hazard is not None
    assert resolved.hazard.voltage_gain == pytest.approx(0.25)
    expected_log_scale = math.log(0.25) + 15.0
    assert resolved.hazard.log_scale == pytest.approx(expected_log_scale)


def test_adaptive_escape_exposes_equation_derived_hazard_trajectory():
    resolved = resolve_adaptive_escape_lif(AdaptiveEscapeLIF().authored_model)
    assert resolved.hazard is not None
    assert resolved.hazard.trajectory_limit_root == "trajectory_limit"
    assert resolved.hazard.trajectory_coefficient_roots == (
        "trajectory_coefficient_one",
        "trajectory_coefficient_two",
    )
    assert resolved.hazard.trajectory_rate_roots == (
        "trajectory_rate_one",
        "trajectory_rate_two",
    )


def test_nearly_equal_adaptive_rates_retain_stable_hazard_evaluator(
    core: CoreEvaluator,
):
    model = AdaptiveEscapeLIF(tau_m=20.0, tau_adaptation=20.00001)
    resolved = resolve_adaptive_escape_lif(model.authored_model)
    assert resolved.hazard is not None
    assert resolved.hazard.trajectory_limit_root is None
    assert resolved.hazard.trajectory_coefficient_roots == ()
    assert resolved.hazard.trajectory_rate_roots == ()
    result = _single_node(model).run(core, t_end=50.0, stochastic_seed=17)
    assert all(math.isfinite(spike.t) for spike in result.core.spikes)


def test_modal_hazard_specialization_matches_expression_fallback(
    core: CoreEvaluator,
):
    model = AdaptiveEscapeLIF(
        drive=25.0,
        escape_rate=0.05,
        adaptation_increment=2.0,
        refractory=1.0,
    )
    optimized = _single_node(model, with_input=True)
    resolved_model = optimized.models[0]
    assert resolved_model.hazard is not None
    fallback_hazard = replace(
        resolved_model.hazard,
        trajectory_limit_root=None,
        trajectory_coefficient_roots=(),
        trajectory_rate_roots=(),
    )
    fallback = replace(
        optimized,
        models=(replace(resolved_model, hazard=fallback_hazard),),
    )
    inputs = (
        SpikeInput(10.0, "in", 1.5),
        SpikeInput(31.25, "in", -0.75),
        SpikeInput(58.0, "in", 2.0),
    )
    optimized_result = optimized.run(
        core, spike_inputs=inputs, t_end=120.0, stochastic_seed=888
    )
    fallback_result = fallback.run(
        core, spike_inputs=inputs, t_end=120.0, stochastic_seed=888
    )
    assert [spike.t for spike in optimized_result.core.spikes] == pytest.approx(
        [spike.t for spike in fallback_result.core.spikes],
        rel=0.0,
        abs=2e-10,
    )
    assert optimized_result.core.states[0].values == pytest.approx(
        fallback_result.core.states[0].values, rel=0.0, abs=2e-10
    )
    assert optimized_result.core.states[0].t_last == pytest.approx(
        fallback_result.core.states[0].t_last, rel=0.0, abs=2e-10
    )


def test_modal_hazard_specialization_uses_runtime_drive_bindings(
    core: CoreEvaluator,
):
    model = AdaptiveEscapeLIF(
        drive=5.0,
        escape_rate=0.05,
        v_escape=-60.0,
        delta_v=5.0,
        adaptation_increment=2.0,
        refractory=1.0,
    )
    optimized = Graph(
        models=(model.model,),
        nodes=(model.node(0),),
        input_ports=(InputPort("drive", 0, InputMode.DRIVE, "drive"),),
        output_ports=(OutputPort("out", 0),),
    ).resolve()
    resolved_model = optimized.models[0]
    assert resolved_model.hazard is not None
    fallback = replace(
        optimized,
        models=(
            replace(
                resolved_model,
                hazard=replace(
                    resolved_model.hazard,
                    trajectory_limit_root=None,
                    trajectory_coefficient_roots=(),
                    trajectory_rate_roots=(),
                ),
            ),
        ),
    )
    updates = (
        DriveInput(20.0, "drive", 30.0),
        DriveInput(70.0, "drive", 8.0),
    )
    optimized_result = optimized.run(
        core, drive_inputs=updates, t_end=120.0, stochastic_seed=123
    )
    fallback_result = fallback.run(
        core, drive_inputs=updates, t_end=120.0, stochastic_seed=123
    )
    assert [spike.t for spike in optimized_result.core.spikes] == pytest.approx(
        [spike.t for spike in fallback_result.core.spikes],
        rel=0.0,
        abs=2e-10,
    )


def test_escape_resolver_rejects_non_affine_log_hazard():
    authored = parse_neuron(
        EscapeLIF().source.replace(
            "escape_rate*exp((v-v_escape)/delta_v)",
            "escape_rate*exp(v*v/delta_v)",
        )
    )
    with pytest.raises(CapabilityError, match="affine"):
        resolve_escape_lif(authored)


def test_constant_voltage_escape_matches_counter_draws_exactly(core: CoreEvaluator):
    seed = 417
    rate = 0.1
    refractory = 2.0
    model = EscapeLIF(
        drive=0.0,
        escape_rate=rate,
        v_escape=-65.0,
        delta_v=2.0,
        refractory=refractory,
    )
    resolved = _single_node(model)
    result = resolved.run(core, t_end=200.0, stochastic_seed=seed)
    expected = []
    cursor = 0.0
    for draw in range(len(result.core.spikes)):
        if draw:
            cursor += refractory
        cursor += _exponential_draw(seed, 0, draw) / rate
        expected.append(cursor)
    assert [spike.t for spike in result.core.spikes] == pytest.approx(
        expected, rel=0.0, abs=2e-10
    )


def test_time_varying_escape_matches_independent_exponential_series(core: CoreEvaluator):
    seed = 83
    model = EscapeLIF(
        tau_m=20.0,
        v_rest=-65.0,
        drive=20.0,
        escape_rate=0.05,
        v_escape=-55.0,
        delta_v=10.0,
        refractory=0.0,
    )
    resolved = _single_node(model)
    actual = resolved.run(core, t_end=100.0, stochastic_seed=seed).core.spikes[0].t
    decay = -1.0 / model.tau_m
    asymptote = model.v_rest + model.resistance * model.drive
    z = (model.v_rest - asymptote) / model.delta_v
    scale = model.escape_rate * math.exp(
        (asymptote - model.v_escape) / model.delta_v
    )
    target = _exponential_draw(seed, 0, 0)

    def cumulative(duration: float) -> float:
        total = duration
        power_over_factorial = 1.0
        for order in range(1, 120):
            power_over_factorial *= z / order
            term = power_over_factorial * math.expm1(
                decay * order * duration
            ) / (decay * order)
            total += term
            if abs(term) < 1e-16 * max(1.0, abs(total)):
                break
        return scale * total

    lower, upper = 0.0, 100.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if cumulative(midpoint) >= target:
            upper = midpoint
        else:
            lower = midpoint
    assert actual == pytest.approx(0.5 * (lower + upper), rel=0.0, abs=2e-9)


@pytest.mark.parametrize("model_type", [EscapeLIF, AdaptiveEscapeLIF])
def test_rescheduling_preserves_same_hazard_sample(core: CoreEvaluator, model_type):
    model = model_type(drive=20.0, escape_rate=0.02, refractory=1.0)
    resolved = _single_node(model, with_input=True)
    uninterrupted = resolved.run(core, t_end=100.0, stochastic_seed=123)
    interrupted = resolved.run(
        core,
        spike_inputs=(SpikeInput(10.0, "in", 0.0), SpikeInput(50.0, "in", 0.0)),
        t_end=100.0,
        stochastic_seed=123,
    )
    assert [spike.t for spike in interrupted.core.spikes] == pytest.approx(
        [spike.t for spike in uninterrupted.core.spikes],
        rel=0.0,
        abs=2e-9,
    )


def test_escape_seed_reproducibility_and_plan_capability(core: CoreEvaluator):
    resolved = _single_node(EscapeLIF(drive=20.0, escape_rate=0.02))
    assert resolved.execution_plan().programs[0].crossing is CrossingMethod.INTEGRATED_HAZARD
    first = resolved.run(core, t_end=100.0, stochastic_seed=5)
    repeated = resolved.run(core, t_end=100.0, stochastic_seed=5)
    different = resolved.run(core, t_end=100.0, stochastic_seed=6)
    assert first.core.spikes == repeated.core.spikes
    assert first.core.spikes != different.core.spikes


def test_escape_graph_round_trip_and_incremental_execution(core: CoreEvaluator):
    model = AdaptiveEscapeLIF(drive=20.0, escape_rate=0.02, refractory=1.0)
    authored = Graph(
        models=(model.model,),
        nodes=(model.node(0),),
        output_ports=(OutputPort("out", 0),),
    )
    resolved = Graph.from_text(authored.to_text()).resolve()
    one_shot = resolved.run(core, t_end=100.0, stochastic_seed=123)
    with resolved.compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=100.0, stochastic_seed=123
        ) as run:
            parts = (
                run.advance_until(10.0),
                run.advance_until(40.0),
                run.finish(),
            )
    incremental_spikes = tuple(
        spike for part in parts for spike in part.core.spikes
    )
    assert incremental_spikes == one_shot.core.spikes


def test_high_adaptation_reduces_escape_firing(core: CoreEvaluator):
    common = dict(
        drive=25.0,
        escape_rate=0.05,
        v_escape=-55.0,
        delta_v=4.0,
        tau_adaptation=100.0,
        refractory=1.0,
    )
    weak = _single_node(AdaptiveEscapeLIF(**common, adaptation_increment=0.0))
    strong = _single_node(AdaptiveEscapeLIF(**common, adaptation_increment=20.0))
    weak_result = weak.run(core, t_end=500.0, stochastic_seed=10)
    strong_result = strong.run(core, t_end=500.0, stochastic_seed=10)
    assert len(strong_result.core.spikes) < len(weak_result.core.spikes)
    assert strong_result.core.states[0].values[1] > 0.0


def test_deterministic_and_stochastic_neurons_mix_in_one_graph(core: CoreEvaluator):
    deterministic = LIF(name="det", drive=20.0)
    escape = EscapeLIF(name="escape", drive=20.0, escape_rate=0.02)
    adaptive = AdaptiveEscapeLIF(name="adaptive", drive=20.0, escape_rate=0.02)
    graph = Graph(
        models=(deterministic.model, escape.model, adaptive.model),
        nodes=(deterministic.node(0), escape.node(1), adaptive.node(2)),
        output_ports=(OutputPort("d", 0), OutputPort("e", 1), OutputPort("a", 2)),
    ).resolve()
    result = graph.run(core, t_end=100.0, stochastic_seed=91)
    fired = {spike.node for spike in result.core.spikes}
    assert fired == {0, 1, 2}
