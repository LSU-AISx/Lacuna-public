from __future__ import annotations

from dataclasses import replace

import pytest

from lacuna import (
    parse_neuron,
    parse_synapse,
    resolve_adaptive_lif,
    resolve_folded_alpha_lif,
    resolve_scalar_lif,
    resolve_stepped_neuron,
)
from lacuna.errors import CapabilityError, PrecisionResolutionError, ResolutionError
from lacuna.expr import ExprDAG, ExprNode, ExprOp
from lacuna.precision import PrecisionProfile
from lacuna.precision_checks import validate_precision_dag, validate_resolved_precision

from .test_adaptive import ADAPTIVE_LIF
from .test_alpha import ALPHA_LIF, ALPHA_SYNAPSE
from .test_dsl_resolver import LIF
from .test_per_edge import _graph
from .test_stepped import ADEX, QIF


def _audit(model, profile=PrecisionProfile.FLOAT32):
    observed = {}

    def record(value, path, is_time):
        round_value = profile.round_time if is_time else profile.round_real
        try:
            rounded = round_value(value, name=path)
        except (ValueError, TypeError) as exc:
            raise PrecisionResolutionError(str(exc)) from exc
        observed[path] = (rounded, is_time)
        return rounded

    result = validate_resolved_precision(model, profile, context="node[3]", record=record)
    assert result is None
    return observed


def _lif():
    return resolve_scalar_lif(parse_neuron(LIF))


def _alpha(tau_m=10.0, tau_s=5.0):
    return resolve_folded_alpha_lif(
        parse_neuron(ALPHA_LIF), parse_synapse(ALPHA_SYNAPSE),
        receptor="i_exc", output="current",
        neuron_bindings={"tau_m": tau_m}, synapse_bindings={"tau_s": tau_s},
    )


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
@pytest.mark.parametrize("model", [
    pytest.param(_lif, id="scalar"),
    pytest.param(lambda: resolve_adaptive_lif(parse_neuron(ADAPTIVE_LIF)), id="adaptive"),
    pytest.param(_alpha, id="alpha-distinct"),
    pytest.param(lambda: _alpha(tau_s=10.0), id="alpha-equal"),
    pytest.param(lambda: _graph(taus=(5.0, 10.0), weights=(1.0, 2.0)).resolve().models[1], id="per-edge"),
    pytest.param(lambda: resolve_stepped_neuron(parse_neuron(QIF)), id="qif"),
    pytest.param(lambda: resolve_stepped_neuron(parse_neuron(ADEX)), id="adex"),
])
def test_standard_models_have_representable_derived_constants(model, profile):
    resolved = model()
    original = repr(resolved)
    values = _audit(resolved, profile)
    assert "node[3].threshold" in values
    assert values["node[3].refractory"][1]
    assert repr(resolved) == original


@pytest.mark.parametrize("family", ("adaptive", "alpha", "per-edge"))
def test_distinct_representable_taus_can_have_collapsed_target_rates(family):
    tau_one = 24.00001335144043
    tau_two = 24.000015258789062
    profile = PrecisionProfile.FLOAT32
    assert profile.round_real(tau_one) == tau_one
    assert profile.round_real(tau_two) == tau_two
    assert -1.0 / tau_one != -1.0 / tau_two
    assert profile.round_real(-1.0 / tau_one) == profile.round_real(-1.0 / tau_two)
    if family == "adaptive":
        model = resolve_adaptive_lif(
            parse_neuron(ADAPTIVE_LIF), {"tau_m": tau_one, "tau_w": tau_two}
        )
    elif family == "alpha":
        model = _alpha(tau_m=tau_one, tau_s=tau_two)
    else:
        model = _graph(taus=(tau_one, tau_two), weights=(1.0, 2.0)).resolve().models[1]
    with pytest.raises(PrecisionResolutionError, match="distinct decay rates collapse"):
        _audit(model)
    _audit(model, PrecisionProfile.FLOAT64)


def test_derived_rate_underflow_does_not_allow_stepped_fallback():
    model = replace(_lif(), a=-1e-48)
    with pytest.raises(PrecisionResolutionError, match="underflows") as caught:
        _audit(model)
    assert isinstance(caught.value, ResolutionError)
    assert not isinstance(caught.value, CapabilityError)


def test_threshold_reset_separation_is_checked_after_rounding():
    model = replace(_lif(), reset=1.0, threshold=1.0 + 2.0**-25)
    with pytest.raises(PrecisionResolutionError, match="reset must remain strictly below"):
        _audit(model)


def test_affine_asymptote_overflow_is_rejected():
    model = replace(_lif(), a=-1e-30, b=1e30)
    with pytest.raises(PrecisionResolutionError, match="asymptote.*overflows"):
        _audit(model)


def test_time_and_model_metadata_have_distinct_roles():
    model = resolve_stepped_neuron(parse_neuron(QIF))
    observed = _audit(model, PrecisionProfile.FLOAT32_TIME64)
    assert observed["node[3].numerical.initial_step"] == (1e-3, True)
    assert observed["node[3].numerical.event_tolerance"] == (1e-9, True)
    assert observed["node[3].numerical.relative_tolerance"][1] is False
    assert observed["node[3].numerical.absolute_tolerance"][1] is False
    adaptive = _audit(resolve_adaptive_lif(parse_neuron(ADAPTIVE_LIF)))
    assert adaptive["node[3].root_hint.fastest_time_constant"][1]
    assert adaptive["node[3].bindings.tau_w"][1] is False


def test_integer_authored_numerical_metadata_is_also_recorded():
    model = resolve_stepped_neuron(parse_neuron(QIF))
    model = replace(model, numerical=replace(model.numerical, maximum_step=1))
    observed = _audit(model)
    assert observed["node[3].numerical.maximum_step"] == (1.0, True)
    assert "node[3].numerical.maximum_steps" not in observed


def _with_dag(nodes, parameters=(), variables=("Delta", "x0"), bindings=None):
    base = _lif()
    return replace(
        base,
        propagation_dag=ExprDAG(tuple(nodes), parameters, variables, {"probe": len(nodes) - 1}),
        bindings={**base.bindings, **(bindings or {})},
    )


def test_parameter_denominator_collapse_is_found_without_state_evaluation():
    model = _with_dag(
        [ExprNode(ExprOp.PARAM, binding=0), ExprNode(ExprOp.PARAM, binding=1),
         ExprNode(ExprOp.SUB, lhs=0, rhs=1), ExprNode(ExprOp.VAR, binding=1),
         ExprNode(ExprOp.DIV, lhs=3, rhs=2)],
        parameters=("p", "q"), bindings={"p": 100000001.0, "q": 100000000.0},
    )
    with pytest.raises(PrecisionResolutionError, match="denominator becomes zero"):
        _audit(model)
    _audit(model, PrecisionProfile.FLOAT64)


def test_static_intermediate_overflow_is_rejected():
    model = _with_dag([
        ExprNode(ExprOp.CONST, value=1e30), ExprNode(ExprOp.CONST, value=1e30),
        ExprNode(ExprOp.MUL, lhs=0, rhs=1),
    ])
    with pytest.raises(PrecisionResolutionError, match="algebraic_constant.*overflows"):
        _audit(model)


@pytest.mark.parametrize("op", (ExprOp.EXP, ExprOp.SIN, ExprOp.COS, ExprOp.TANH))
def test_audit_does_not_evaluate_transcendental_functions(op):
    model = _with_dag([ExprNode(ExprOp.CONST, value=1e30), ExprNode(op, lhs=0)])
    observed = _audit(model)
    assert not any("nodes[1].algebraic_constant" in key for key in observed)


def test_time_dependency_remains_unbound_in_preflight():
    model = _with_dag([ExprNode(ExprOp.VAR, binding=0)], variables=("Time", "x0"))
    _audit(model, PrecisionProfile.FLOAT32_TIME64)
    _audit(model, PrecisionProfile.FLOAT32)


def test_unused_absolute_time_slot_does_not_imply_dependency():
    model = _with_dag([ExprNode(ExprOp.VAR, binding=1)], variables=("Time", "x0"))
    _audit(model, PrecisionProfile.FLOAT32_TIME64)


def test_hazard_sentinel_has_target_representation():
    from lacuna.ir import HazardKind, ResolvedHazard

    model = replace(_lif(), hazard=ResolvedHazard(HazardKind.EXPONENTIAL_VOLTAGE, 0.0, 1.0))
    observed = _audit(model)
    assert any(value == float.fromhex("0x1.fffffep127") for value, _ in observed.values())


@pytest.mark.parametrize("op,left,right", [
    (ExprOp.LOG, 0.0, 0.0),
    (ExprOp.POW, 0.0, -1.0),
    (ExprOp.POW, -1.0, 0.5),
])
def test_known_parameter_only_function_domain_is_rejected(op, left, right):
    model = _with_dag([
        ExprNode(ExprOp.CONST, value=left), ExprNode(ExprOp.CONST, value=right),
        ExprNode(op, lhs=0, rhs=1),
    ])
    with pytest.raises(PrecisionResolutionError, match="target.*domain|nonpositive"):
        _audit(model)


@pytest.mark.parametrize("rule_name", ("pair", "triplet", "modulated"))
def test_learning_dag_audit_does_not_execute_weight_or_trace_updates(rule_name):
    from lacuna.learning import resolve_learning
    from lacuna.plasticity import ModulatedSTDP, PairSTDP, TripletSTDP

    rule = {"pair": PairSTDP, "triplet": TripletSTDP, "modulated": ModulatedSTDP}[rule_name]()
    lowered = resolve_learning(rule)
    bindings = dict(zip(lowered.program.parameter_names, lowered.parameter_values))
    records = {}

    def record(value, path, is_time):
        assert is_time is False
        result = PrecisionProfile.FLOAT32.round_real(value, name=path)
        records[path] = result
        return result

    for event in lowered.program.events:
        result = validate_precision_dag(
            event.expressions, bindings, "float32", context=event.event.value, record=record
        )
        assert result is None
    assert any("bindings" in path for path in records)
    assert not any("weight" in path or "trace" in path for path in records)
