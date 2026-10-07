"""Declarative edge-learning programs derived from event equations.

The standard plasticity classes are authoring conveniences.  This module lowers
them into model-neutral trace and event programs whose structural identity is
independent of rule names and numeric parameter values.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import sympy

from .expr import ExprDAG, lower_expressions
from .plasticity import (
    ModulatedSTDP,
    PairSTDP,
    PlasticityRule,
    SoftExcursionModulated,
    TripletSTDP,
    VoltageModulatedSTDP,
)


class LearningEvent(str, Enum):
    """Event kinds that may execute an edge-learning equation map."""

    PRE_SPIKE = "PRE_SPIKE"
    POST_SPIKE = "POST_SPIKE"
    OBSERVATION = "OBSERVATION"
    MODULATION_POSITIVE = "MODULATION_POSITIVE"
    MODULATION_NEGATIVE = "MODULATION_NEGATIVE"


@dataclass(frozen=True)
class LearningTrace:
    """One lazily decayed scalar trace."""

    name: str
    tau_parameter: int


@dataclass(frozen=True)
class LearningEventProgram:
    """State and normalized-weight map applied at one event kind."""

    event: LearningEvent
    advance_traces: tuple[int, ...]
    expressions: ExprDAG
    weight_root: str | None
    trace_roots: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class LearningObserverProgram:
    """Neuron-local observation equations shared by incoming plastic edges.

    The C runtime advances ``slow_voltage`` continuously and the two activity
    traces lazily.  This DAG applies an observation event and returns the local
    gain supplied to the edge-local ``OBSERVATION`` program.
    """

    variable_names: tuple[str, ...]
    expressions: ExprDAG
    voltage_tau_parameter: int
    fast_activity_tau_parameter: int
    slow_activity_tau_parameter: int
    band_width_parameter: int
    fast_activity_root: str
    slow_activity_root: str
    gain_root: str


@dataclass(frozen=True)
class LearningProgram:
    """Reusable structural program for one family of edge-local learning."""

    key: str
    parameter_names: tuple[str, ...]
    variable_names: tuple[str, ...]
    traces: tuple[LearningTrace, ...]
    events: tuple[LearningEventProgram, ...]
    observer: LearningObserverProgram | None = None
    clamp_normalized_weight: bool = True


@dataclass(frozen=True)
class ResolvedLearning:
    """One program plus the numeric values bound by an authored edge rule."""

    program: LearningProgram
    parameter_values: tuple[float, ...]
    weight_bounds: tuple[float, float]


def _event_program(
    event: LearningEvent,
    *,
    parameter_names: tuple[str, ...],
    variable_names: tuple[str, ...],
    advance_traces: tuple[int, ...],
    weight: sympy.Expr | None = None,
    traces: dict[int, sympy.Expr] | None = None,
) -> LearningEventProgram:
    expressions: dict[str, sympy.Expr] = {}
    weight_root = None
    if weight is not None:
        weight_root = "weight"
        expressions[weight_root] = weight
    trace_roots = []
    for index, expression in sorted((traces or {}).items()):
        root = f"trace_{index}"
        expressions[root] = expression
        trace_roots.append((index, root))
    return LearningEventProgram(
        event=event,
        advance_traces=advance_traces,
        expressions=lower_expressions(
            expressions,
            parameters=parameter_names,
            variables=variable_names,
        ),
        weight_root=weight_root,
        trace_roots=tuple(trace_roots),
    )


def _dag_shape(dag: ExprDAG) -> dict[str, object]:
    return {
        "parameter_count": len(dag.parameters),
        "variable_count": len(dag.variables),
        "nodes": [
            [int(node.op), node.lhs, node.rhs, node.binding, float(node.value).hex()]
            for node in dag.nodes
        ],
    }


def _structural_key(
    traces: tuple[LearningTrace, ...],
    parameter_count: int,
    variable_count: int,
    events: tuple[LearningEventProgram, ...],
    observer: LearningObserverProgram | None,
) -> str:
    document = {
        "parameter_count": parameter_count,
        "variable_count": variable_count,
        "trace_tau_parameters": [trace.tau_parameter for trace in traces],
        "events": [
            {
                "event": event.event.value,
                "advance": list(event.advance_traces),
                "dag": _dag_shape(event.expressions),
                "weight": (
                    None
                    if event.weight_root is None
                    else event.expressions.roots[event.weight_root]
                ),
                "traces": [
                    [index, event.expressions.roots[root]]
                    for index, root in event.trace_roots
                ],
            }
            for event in events
        ],
        "observer": None
        if observer is None
        else {
            "variables": list(observer.variable_names),
            "dag": _dag_shape(observer.expressions),
            "voltage_tau_parameter": observer.voltage_tau_parameter,
            "fast_activity_tau_parameter": observer.fast_activity_tau_parameter,
            "slow_activity_tau_parameter": observer.slow_activity_tau_parameter,
            "band_width_parameter": observer.band_width_parameter,
            "fast_activity_root": observer.expressions.roots[
                observer.fast_activity_root
            ],
            "slow_activity_root": observer.expressions.roots[
                observer.slow_activity_root
            ],
            "gain_root": observer.expressions.roots[observer.gain_root],
        },
        "clamp_normalized_weight": True,
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _program(
    *,
    parameter_names: tuple[str, ...],
    trace_definitions: tuple[tuple[str, str], ...],
    build_events,
    build_observer=None,
    use_input_acceptance: bool = False,
) -> LearningProgram:
    trace_names = tuple(name for name, _ in trace_definitions)
    base_variable_names = (
        "weight",
        "modulation",
        "learning_scale",
        "post_readout",
    )
    if build_observer is not None:
        base_variable_names += ("observation_gain", "event_amplitude")
    if use_input_acceptance:
        base_variable_names += ("input_accepted",)
    variable_names = base_variable_names + trace_names
    parameters = dict(zip(parameter_names, sympy.symbols(parameter_names)))
    variables = dict(zip(variable_names, sympy.symbols(variable_names)))
    traces = tuple(
        LearningTrace(name, parameter_names.index(tau))
        for name, tau in trace_definitions
    )
    events = tuple(build_events(parameters, variables, parameter_names, variable_names))
    observer = (
        None
        if build_observer is None
        else build_observer(parameters, parameter_names)
    )
    key = _structural_key(
        traces, len(parameter_names), len(variable_names), events, observer
    )
    return LearningProgram(
        key=key,
        parameter_names=parameter_names,
        variable_names=variable_names,
        traces=traces,
        events=events,
        observer=observer,
    )


def _pair_program() -> LearningProgram:
    names = ("tau_pre", "tau_post", "a_plus", "a_minus", "learning_rate")

    def events(p, v, parameter_names, variable_names):
        """Build pair-STDP maps for pre and postsynaptic spikes."""

        weight = v["weight"]
        scale = v["learning_scale"]
        pre = v["pre_fast"]
        post = v["post_fast"]
        yield _event_program(
            LearningEvent.PRE_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1),
            weight=weight - p["learning_rate"] * scale * p["a_minus"] * weight * post,
            traces={0: pre + 1},
        )
        yield _event_program(
            LearningEvent.POST_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1),
            weight=weight
            + p["learning_rate"] * scale * p["a_plus"] * (1 - weight) * pre,
            traces={1: post + 1},
        )

    return _program(
        parameter_names=names,
        trace_definitions=(("pre_fast", "tau_pre"), ("post_fast", "tau_post")),
        build_events=events,
    )


def _triplet_program() -> LearningProgram:
    names = (
        "tau_plus",
        "tau_minus",
        "tau_x",
        "tau_y",
        "a2_plus",
        "a2_minus",
        "a3_plus",
        "a3_minus",
        "learning_rate",
    )

    def events(p, v, parameter_names, variable_names):
        """Build triplet-STDP maps with fast and slow traces."""

        weight = v["weight"]
        scale = v["learning_scale"]
        pre_fast = v["pre_fast"]
        post_fast = v["post_fast"]
        pre_slow = v["pre_slow"]
        post_slow = v["post_slow"]
        yield _event_program(
            LearningEvent.PRE_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1, 2, 3),
            weight=weight
            - p["learning_rate"]
            * scale
            * post_fast
            * (p["a2_minus"] + p["a3_minus"] * pre_slow),
            traces={0: pre_fast + 1, 2: pre_slow + 1},
        )
        yield _event_program(
            LearningEvent.POST_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1, 2, 3),
            weight=weight
            + p["learning_rate"]
            * scale
            * pre_fast
            * (p["a2_plus"] + p["a3_plus"] * post_slow),
            traces={1: post_fast + 1, 3: post_slow + 1},
        )

    return _program(
        parameter_names=names,
        trace_definitions=(
            ("pre_fast", "tau_plus"),
            ("post_fast", "tau_minus"),
            ("pre_slow", "tau_x"),
            ("post_slow", "tau_y"),
        ),
        build_events=events,
    )


def _modulated_program(*, consume: bool) -> LearningProgram:
    names = (
        "tau_pre",
        "tau_post",
        "tau_eligibility_plus",
        "tau_eligibility_minus",
        "positive_plus",
        "positive_minus",
        "negative_plus",
        "negative_minus",
        "learning_rate",
    )

    def events(p, v, parameter_names, variable_names):
        """Build split eligibility and modulation maps."""

        weight = v["weight"]
        modulation = v["modulation"]
        scale = v["learning_scale"]
        pre = v["pre_fast"]
        post = v["post_fast"]
        plus = v["eligibility_plus"]
        minus = v["eligibility_minus"]
        yield _event_program(
            LearningEvent.PRE_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1, 3),
            traces={0: pre + 1, 3: minus + post},
        )
        yield _event_program(
            LearningEvent.POST_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1, 2),
            traces={1: post + 1, 2: plus + pre},
        )
        reset = {2: sympy.Integer(0), 3: sympy.Integer(0)} if consume else {}
        for event, plus_name, minus_name, amplitude in (
            (
                LearningEvent.MODULATION_POSITIVE,
                "positive_plus",
                "positive_minus",
                modulation,
            ),
            (
                LearningEvent.MODULATION_NEGATIVE,
                "negative_plus",
                "negative_minus",
                -modulation,
            ),
        ):
            yield _event_program(
                event,
                parameter_names=parameter_names,
                variable_names=variable_names,
                advance_traces=(2, 3),
                weight=weight
                + p["learning_rate"]
                * scale
                * amplitude
                * (p[plus_name] * plus + p[minus_name] * minus),
                traces=reset,
            )

    return _program(
        parameter_names=names,
        trace_definitions=(
            ("pre_fast", "tau_pre"),
            ("post_fast", "tau_post"),
            ("eligibility_plus", "tau_eligibility_plus"),
            ("eligibility_minus", "tau_eligibility_minus"),
        ),
        build_events=events,
    )


_PAIR_PROGRAM = _pair_program()
_TRIPLET_PROGRAM = _triplet_program()
_MODULATED_PROGRAMS = {
    False: _modulated_program(consume=False),
    True: _modulated_program(consume=True),
}


def _voltage_modulated_program(*, consume: bool) -> LearningProgram:
    names = (
        "tau_pre",
        "tau_post",
        "tau_eligibility_plus",
        "tau_eligibility_minus",
        "tau_voltage_eligibility",
        "surrogate_threshold",
        "surrogate_slope",
        "spike_scale",
        "voltage_scale",
        "learning_rate",
    )

    def events(p, v, parameter_names, variable_names):
        """Build voltage-sensitive eligibility and modulation maps."""

        weight = v["weight"]
        modulation = v["modulation"]
        scale = v["learning_scale"]
        post_readout = v["post_readout"]
        pre = v["pre_fast"]
        post = v["post_fast"]
        plus = v["eligibility_plus"]
        minus = v["eligibility_minus"]
        voltage = v["voltage_eligibility"]
        slope = p["surrogate_slope"]
        # Stable derivative of softplus((v - threshold) / slope) with
        # respect to v: sigmoid((v - threshold) / slope) / slope.
        voltage_gain = (
            1
            + sympy.tanh(
                (post_readout - p["surrogate_threshold"]) / (2 * slope)
            )
        ) / (2 * slope)
        yield _event_program(
            LearningEvent.PRE_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1, 2, 3, 4),
            traces={
                0: pre + 1,
                3: minus + post,
                4: voltage + (pre + 1) * voltage_gain,
            },
        )
        yield _event_program(
            LearningEvent.POST_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=(0, 1, 2),
            traces={1: post + 1, 2: plus + pre},
        )
        reset = (
            {2: sympy.Integer(0), 3: sympy.Integer(0), 4: sympy.Integer(0)}
            if consume
            else {}
        )
        eligibility = (
            p["spike_scale"] * (plus - minus)
            + p["voltage_scale"] * voltage
        )
        for event in (
            LearningEvent.MODULATION_POSITIVE,
            LearningEvent.MODULATION_NEGATIVE,
        ):
            yield _event_program(
                event,
                parameter_names=parameter_names,
                variable_names=variable_names,
                advance_traces=(2, 3, 4),
                weight=(
                    weight
                    + p["learning_rate"]
                    * scale
                    * modulation
                    * eligibility
                ),
                traces=reset,
            )

    return _program(
        parameter_names=names,
        trace_definitions=(
            ("pre_fast", "tau_pre"),
            ("post_fast", "tau_post"),
            ("eligibility_plus", "tau_eligibility_plus"),
            ("eligibility_minus", "tau_eligibility_minus"),
            ("voltage_eligibility", "tau_voltage_eligibility"),
        ),
        build_events=events,
    )


_VOLTAGE_MODULATED_PROGRAMS = {
    False: _voltage_modulated_program(consume=False),
    True: _voltage_modulated_program(consume=True),
}


def _soft_excursion_program(
    *,
    consume: bool,
    adaptive_baseline: bool,
    use_upward_excursion: bool,
    use_proximity: bool,
) -> LearningProgram:
    names = ["tau_pre", "tau_eligibility"]
    if use_upward_excursion:
        names.extend(("proximity_width", "excursion_smoothing"))
        if adaptive_baseline:
            names.extend(("tau_voltage_baseline", "baseline_epsilon"))
        else:
            names.append("fixed_baseline")
    if use_proximity:
        names.extend(("threshold", "proximity_width", "proximity_slope"))
    names.extend(("soft_scale", "learning_rate"))
    # Preserve first occurrence when a parameter, such as proximity_width, is
    # used by more than one retained component.
    parameter_names = tuple(dict.fromkeys(names))

    trace_definitions = [("pre_fast", "tau_pre")]
    if use_upward_excursion and adaptive_baseline:
        trace_definitions.extend(
            (
                ("voltage_sum", "tau_voltage_baseline"),
                ("voltage_mass", "tau_voltage_baseline"),
            )
        )
    trace_definitions.append(("soft_eligibility", "tau_eligibility"))
    trace_definitions = tuple(trace_definitions)

    def events(p, v, parameter_names, variable_names):
        """Build graded excursion eligibility and modulation maps."""

        weight = v["weight"]
        modulation = v["modulation"]
        scale = v["learning_scale"]
        post_readout = v["post_readout"]
        pre = v["pre_fast"]
        eligibility = v["soft_eligibility"]

        upward = sympy.Integer(1)
        if use_upward_excursion:
            if adaptive_baseline:
                voltage_sum = v["voltage_sum"]
                voltage_mass = v["voltage_mass"]
                baseline = voltage_sum / (voltage_mass + p["baseline_epsilon"])
            else:
                baseline = p["fixed_baseline"]
            excursion = post_readout - baseline
            # Smooth positive part of the upward voltage excursion.  This is a
            # graded local event, not a declared spike surrogate derivative.
            upward = (
                excursion
                + (excursion**2 + p["excursion_smoothing"] ** 2)
                ** sympy.Rational(1, 2)
            ) / (2 * p["proximity_width"])

        proximity = sympy.Integer(1)
        if use_proximity:
            # Graded activation of the band beginning proximity_width below
            # threshold.  Only the voltage local to the postsynaptic endpoint
            # is read.
            proximity = (
                1
                + sympy.tanh(
                    (
                        post_readout
                        - (p["threshold"] - p["proximity_width"])
                    )
                    / (2 * p["proximity_slope"])
                )
            ) / 2
        drive = (pre + 1) * proximity * upward

        trace_updates = {0: pre + 1}
        if use_upward_excursion and adaptive_baseline:
            trace_updates.update(
                {
                    1: voltage_sum + post_readout,
                    2: voltage_mass + 1,
                }
            )
        eligibility_index = len(trace_definitions) - 1
        trace_updates[eligibility_index] = eligibility + drive
        yield _event_program(
            LearningEvent.PRE_SPIKE,
            parameter_names=parameter_names,
            variable_names=variable_names,
            advance_traces=tuple(range(len(trace_definitions))),
            traces=trace_updates,
        )
        reset = {eligibility_index: sympy.Integer(0)} if consume else {}
        for event in (
            LearningEvent.MODULATION_POSITIVE,
            LearningEvent.MODULATION_NEGATIVE,
        ):
            yield _event_program(
                event,
                parameter_names=parameter_names,
                variable_names=variable_names,
                advance_traces=(eligibility_index,),
                weight=(
                    weight
                    + p["learning_rate"]
                    * scale
                    * modulation
                    * p["soft_scale"]
                    * eligibility
                ),
                traces=reset,
            )

    return _program(
        parameter_names=parameter_names,
        trace_definitions=trace_definitions,
        build_events=events,
    )


_SOFT_EXCURSION_PROGRAMS = {
    (consume, adaptive_baseline, use_upward_excursion, use_proximity): (
        _soft_excursion_program(
            consume=consume,
            adaptive_baseline=adaptive_baseline,
            use_upward_excursion=use_upward_excursion,
            use_proximity=use_proximity,
        )
    )
    for consume in (False, True)
    for adaptive_baseline in (False, True)
    for use_upward_excursion in (False, True)
    for use_proximity in (False, True)
}


def resolve_learning(rule: PlasticityRule) -> ResolvedLearning:
    """Lower one standard authoring rule into a structural event program."""

    if isinstance(rule, PairSTDP):
        program = _PAIR_PROGRAM
    elif isinstance(rule, TripletSTDP):
        program = _TRIPLET_PROGRAM
    elif isinstance(rule, ModulatedSTDP):
        program = _MODULATED_PROGRAMS[rule.consume_on_modulation]
    elif isinstance(rule, VoltageModulatedSTDP):
        program = _VOLTAGE_MODULATED_PROGRAMS[rule.consume_on_modulation]
    elif isinstance(rule, SoftExcursionModulated):
        program = _SOFT_EXCURSION_PROGRAMS[
            (
                rule.consume_on_modulation,
                rule.adaptive_baseline,
                rule.use_upward_excursion,
                rule.use_proximity,
            )
        ]
    else:  # pragma: no cover - graph validation rejects unsupported rules
        raise TypeError("unsupported plasticity rule")
    return ResolvedLearning(
        program=program,
        parameter_values=tuple(float(getattr(rule, name)) for name in program.parameter_names),
        weight_bounds=tuple(float(value) for value in rule.bounds),
    )


__all__ = [
    "LearningEvent",
    "LearningEventProgram",
    "LearningObserverProgram",
    "LearningProgram",
    "LearningTrace",
    "ResolvedLearning",
    "resolve_learning",
]
