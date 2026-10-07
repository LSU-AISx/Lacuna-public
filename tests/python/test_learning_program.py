from __future__ import annotations

import math
from dataclasses import replace

import pytest
import sympy

from lacuna import (
    DepositKind,
    Engine,
    ExprOp,
    LearningEvent,
    LearningEventProgram,
    LearningProgram,
    LIF,
    MixedInputSpike,
    ModulatedSTDP,
    NetworkBuilder,
    PairSTDP,
    SoftExcursionModulated,
    SpikeTrain,
    TripletSTDP,
    VoltageModulatedSTDP,
    resolve_learning,
)
from lacuna.expr import lower_expressions
from lacuna.ffi import CoreEvaluator


def _event(rule, kind: LearningEvent):
    resolved = resolve_learning(rule)
    return resolved, next(
        event for event in resolved.program.events if event.event is kind
    )


def _parameters(resolved) -> dict[str, float]:
    return dict(
        zip(resolved.program.parameter_names, resolved.parameter_values)
    )


def _variables(resolved, **values: float) -> dict[str, float]:
    result = {name: 0.0 for name in resolved.program.variable_names}
    result.update(values)
    return result


def test_numeric_rule_values_do_not_change_structural_program_identity() -> None:
    first = resolve_learning(PairSTDP())
    second = resolve_learning(
        PairSTDP(
            tau_pre=17.0,
            tau_post=29.0,
            a_plus=0.12,
            a_minus=0.07,
            learning_rate=0.003,
            bounds=(0.2, 4.0),
        )
    )

    assert first.program.key == second.program.key
    assert first.parameter_values != second.parameter_values
    assert first.weight_bounds != second.weight_bounds


def test_structurally_different_learning_equations_have_different_hashes() -> None:
    pair = resolve_learning(PairSTDP()).program
    triplet = resolve_learning(TripletSTDP()).program
    retained = resolve_learning(
        ModulatedSTDP(consume_on_modulation=False)
    ).program
    consumed = resolve_learning(
        ModulatedSTDP(consume_on_modulation=True)
    ).program
    voltage = resolve_learning(VoltageModulatedSTDP()).program
    soft = resolve_learning(SoftExcursionModulated()).program

    assert len(
        {
            pair.key,
            triplet.key,
            retained.key,
            consumed.key,
            voltage.key,
            soft.key,
        }
    ) == 6
    assert tuple(event.event for event in pair.events) == (
        LearningEvent.PRE_SPIKE,
        LearningEvent.POST_SPIKE,
    )
    assert tuple(event.event for event in consumed.events) == (
        LearningEvent.PRE_SPIKE,
        LearningEvent.POST_SPIKE,
        LearningEvent.MODULATION_POSITIVE,
        LearningEvent.MODULATION_NEGATIVE,
    )
    assert "post_readout" in voltage.variable_names
    assert tuple(trace.name for trace in voltage.traces)[-1] == (
        "voltage_eligibility"
    )
    assert tuple(event.event for event in soft.events) == (
        LearningEvent.PRE_SPIKE,
        LearningEvent.MODULATION_POSITIVE,
        LearningEvent.MODULATION_NEGATIVE,
    )
    assert tuple(trace.name for trace in soft.traces) == (
        "pre_fast",
        "voltage_sum",
        "voltage_mass",
        "soft_eligibility",
    )
def test_soft_excursion_equation_prefers_upward_near_threshold_voltage(
    core: CoreEvaluator,
) -> None:
    rule = SoftExcursionModulated(
        threshold=-60.0,
        proximity_width=5.0,
        proximity_slope=0.5,
        excursion_smoothing=0.01,
    )
    resolved, event = _event(rule, LearningEvent.PRE_SPIKE)
    common = dict(
        pre_fast=1.0,
        voltage_sum=-65.0,
        voltage_mass=1.0,
        soft_eligibility=0.0,
    )
    upward = core.evaluate_expr(
        event.expressions,
        parameters=_parameters(resolved),
        variables=_variables(resolved, post_readout=-61.0, **common),
    )
    downward = core.evaluate_expr(
        event.expressions,
        parameters=_parameters(resolved),
        variables=_variables(resolved, post_readout=-66.0, **common),
    )

    assert upward["trace_3"] > 0.0
    assert upward["trace_3"] > 1_000.0 * downward["trace_3"]


def test_soft_component_ablation_compiles_only_retained_state() -> None:
    full = resolve_learning(SoftExcursionModulated()).program
    fixed = resolve_learning(
        SoftExcursionModulated(adaptive_baseline=False)
    ).program
    proximity_only = resolve_learning(
        SoftExcursionModulated(use_upward_excursion=False)
    ).program
    excursion_only = resolve_learning(
        SoftExcursionModulated(use_proximity=False)
    ).program

    assert len({full.key, fixed.key, proximity_only.key, excursion_only.key}) == 4
    assert tuple(trace.name for trace in fixed.traces) == (
        "pre_fast",
        "soft_eligibility",
    )
    assert "fixed_baseline" in fixed.parameter_names
    assert "tau_voltage_baseline" not in fixed.parameter_names
    assert fixed.events[1].advance_traces == (1,)
    assert fixed.events[2].advance_traces == (1,)
    assert tuple(trace.name for trace in proximity_only.traces) == (
        "pre_fast",
        "soft_eligibility",
    )
    assert "tau_voltage_baseline" not in proximity_only.parameter_names
    assert "excursion_smoothing" not in proximity_only.parameter_names
    assert tuple(trace.name for trace in excursion_only.traces) == (
        "pre_fast",
        "voltage_sum",
        "voltage_mass",
        "soft_eligibility",
    )
    assert "threshold" not in excursion_only.parameter_names
    assert "proximity_slope" not in excursion_only.parameter_names


def test_pair_event_program_matches_trusted_c_learning_execution(
    core: CoreEvaluator,
) -> None:
    rule = PairSTDP(
        tau_pre=5.0,
        tau_post=7.0,
        a_plus=0.6,
        a_minus=0.3,
        learning_rate=0.05,
    )
    resolved, event = _event(rule, LearningEvent.POST_SPIKE)
    pre_at_post = math.exp(-1.0 / rule.tau_pre)
    evaluated = core.evaluate_expr(
        event.expressions,
        parameters=_parameters(resolved),
        variables=_variables(
            resolved,
            weight=0.5,
            learning_scale=1.0,
            pre_fast=pre_at_post,
        ),
    )

    builder = NetworkBuilder("declarative-pair-check")
    pre = builder.neuron("pre", LIF(name="pre_lif"))
    post = builder.neuron("post", LIF(name="post_lif"))
    edge = builder.connect(pre, post, weight=0.5, plasticity=rule)[0]
    pre_input = builder.input("pre_input", pre)
    post_input = builder.input("post_input", post)
    with Engine(core._lib._name).compile(builder.build()) as simulation:
        result = simulation.run(
            3.0,
            inputs={
                pre_input: SpikeTrain((1.0,), 20.0),
                post_input: SpikeTrain((2.0,), 20.0),
            },
        )

    assert evaluated[event.weight_root] == pytest.approx(result.weights[edge])
    post_root = dict(event.trace_roots)[1]
    assert evaluated[post_root] == 1.0


def test_modulation_event_program_encodes_sign_and_consumption_in_equations(
    core: CoreEvaluator,
) -> None:
    rule = ModulatedSTDP(
        positive_plus=1.2,
        positive_minus=-0.4,
        negative_plus=-0.8,
        negative_minus=0.3,
        learning_rate=0.1,
        consume_on_modulation=True,
    )
    resolved, positive = _event(rule, LearningEvent.MODULATION_POSITIVE)
    evaluated = core.evaluate_expr(
        positive.expressions,
        parameters=_parameters(resolved),
        variables=_variables(
            resolved,
            weight=0.4,
            modulation=0.5,
            learning_scale=1.0,
            eligibility_plus=0.7,
            eligibility_minus=0.2,
        ),
    )
    expected = 0.4 + 0.1 * 0.5 * (1.2 * 0.7 - 0.4 * 0.2)

    assert evaluated[positive.weight_root] == pytest.approx(expected)
    assert {
        index: evaluated[root] for index, root in positive.trace_roots
    } == {2: 0.0, 3: 0.0}


def test_plan_reuses_learning_program_across_different_pair_parameters() -> None:
    builder = NetworkBuilder("learning-program-reuse")
    pre = builder.neuron("pre", LIF(name="pre_lif"))
    posts = builder.population("post", 2, LIF(name="post_lif"))
    builder.connect(pre, posts[0], weight=0.5, plasticity=PairSTDP())
    builder.connect(
        pre,
        posts[1],
        weight=0.6,
        plasticity=PairSTDP(
            tau_pre=11.0,
            tau_post=13.0,
            learning_rate=0.002,
        ),
    )

    plan = builder.build().graph.resolve().execution_plan()

    assert len(plan.learning_programs) == 1
    assert tuple(edge.learning_program for edge in plan.connections) == (0, 0)
    assert plan.connections[0].learning_parameter_values != (
        plan.connections[1].learning_parameter_values
    )
    assert len(plan.connection_batches) == 1
    assert plan.connection_batches[0].connections == (0, 1)


def test_default_c_runtime_executes_an_unrecognized_learning_equation(
    core: CoreEvaluator,
) -> None:
    """The runtime must interpret structure, not dispatch on a model name."""

    builder = NetworkBuilder("custom-learning-equation")
    pre = builder.neuron("pre", LIF(name="pre_lif"))
    post = builder.neuron("post", LIF(name="post_lif"))
    builder.connect(pre, post, weight=0.5, plasticity=PairSTDP())
    resolved = builder.build().graph.resolve()
    plan = resolved.execution_plan()

    eta = sympy.Symbol("eta")
    weight, modulation, scale, post_readout = sympy.symbols(
        "weight modulation learning_scale post_readout"
    )
    expressions = lower_expressions(
        {"weight": weight + eta * scale},
        parameters=("eta",),
        variables=("weight", "modulation", "learning_scale", "post_readout"),
    )
    custom = LearningProgram(
        key="custom-pre-increment",
        parameter_names=("eta",),
        variable_names=(
            "weight",
            "modulation",
            "learning_scale",
            "post_readout",
        ),
        traces=(),
        events=(
            LearningEventProgram(
                event=LearningEvent.PRE_SPIKE,
                advance_traces=(),
                expressions=expressions,
                weight_root="weight",
                trace_roots=(),
            ),
        ),
    )
    connection = replace(
        plan.connections[0],
        learning_parameter_values=(0.1,),
    )
    custom_plan = replace(
        plan,
        learning_programs=(custom,),
        connections=(connection,),
    )

    with core.compile_execution_plan(custom_plan) as compiled:
        result = compiled.run(
            resolved.initial_values,
            inputs=(MixedInputSpike(1.0, pre.id, 20.0, DepositKind.STATE_ADD, 0),),
            t_end=2.0,
        )

    assert result.weights == pytest.approx((0.6,))


def test_learning_event_dependency_masks_are_derived_from_equations(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("learning-dependency-mask")
    pre = builder.neuron("pre", LIF())
    post = builder.neuron("post", LIF())
    edge = builder.connect(
        pre, post, weight=0.5, plasticity=ModulatedSTDP()
    )[0]
    builder.modulator("reward", targets=(edge,))
    plan = builder.build().graph.resolve().execution_plan()

    programs, _, _, _ = core._pack_learning_plan(plan)
    pre_event = programs[0].events[0]
    positive_modulation = programs[0].events[2]

    # Trace construction does not read weight, modulation, or learning scale.
    assert pre_event.variable_mask & 0b111 == 0
    # The reward update reads all three plus its eligibility traces.
    assert positive_modulation.variable_mask & 0b111 == 0b111
