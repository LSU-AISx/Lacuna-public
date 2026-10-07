"""Symbolic resolver for the scalar stable-LIF milestone."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import sympy

from .errors import CapabilityError, ResolutionError
from .resolution_cache import _cached_resolution
from .expr import lower_expressions, phi1, phi1_derivative
from .ir import (
    DispatchForm,
    ExpPolyRootHint,
    HazardKind,
    NeuronModel,
    MultiExpRootHint,
    NumericalConfig,
    ParameterDefinition,
    ParameterDomain,
    ReactiveMode,
    ResolvedAdaptiveLIF,
    ResolvedAlphaLIF,
    ResolvedPerEdgeLIF,
    ResolvedReactiveIF,
    ResolvedHazard,
    ResolvedScalarLIF,
    ResolvedSteppedNeuron,
    RootFindHint,
    ScalarLogRootHint,
    StateRole,
    SynapseModel,
    SynapseTier,
    TwoExpRootHint,
    ThresholdDefinition,
)

_TOKENS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SAFE_EXPRESSION = re.compile(r"^[A-Za-z0-9_+*/().,^\-\s]+$")
_DECIMAL_LITERAL = re.compile(r"(?<![A-Za-z_])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_SAFE_INTRINSICS = {
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tanh": sympy.tanh,
}


@dataclass(frozen=True)
class PerEdgeSynapseInstance:
    """One graph edge requesting a stateful current-based synapse mapping."""

    edge_id: int
    synapse: SynapseModel
    receptor: str
    output: str
    bindings: Mapping[str, float]
    initial: float | tuple[float, ...] = 0.0


def _hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _expression(text: str, symbols: Mapping[str, sympy.Symbol], context: str) -> sympy.Expr:
    if "__" in text or not _SAFE_EXPRESSION.fullmatch(text):
        raise ResolutionError(f"unsafe or unsupported syntax in {context}: '{text}'")
    identifiers = set(_TOKENS.findall(_DECIMAL_LITERAL.sub("", text)))
    unknown = identifiers.difference(set(symbols).union(_SAFE_INTRINSICS))
    if unknown:
        raise ResolutionError(f"unknown symbol(s) in {context}: {', '.join(sorted(unknown))}")
    try:
        locals_map = dict(_SAFE_INTRINSICS)
        locals_map.update(symbols)
        return sympy.sympify(text, locals=locals_map)
    except (TypeError, ValueError, SyntaxError) as exc:
        raise ResolutionError(f"could not parse {context}: '{text}'") from exc


def _number(expr: sympy.Expr, bindings: Mapping[sympy.Symbol, float], context: str) -> float:
    value = expr.subs(bindings)
    if value.free_symbols:
        names = ", ".join(sorted(str(item) for item in value.free_symbols))
        raise ResolutionError(f"unbound symbol(s) in {context}: {names}")
    if value.is_real is False:
        raise ResolutionError(f"{context} must be real")
    result = float(sympy.N(value, 17))
    if not math.isfinite(result):
        raise ResolutionError(f"{context} must be finite")
    return result


def _parameter_values(
    definitions: tuple[ParameterDefinition, ...],
    bindings: Mapping[str, float] | None,
    context: str,
) -> dict[str, float]:
    by_name = {definition.name: definition for definition in definitions}
    values = {name: float(value) for name, value in (bindings or {}).items()}
    unknown = set(values).difference(by_name)
    if unknown:
        raise ResolutionError(
            f"unknown {context} parameter binding(s): {', '.join(sorted(unknown))}"
        )
    for name, definition in by_name.items():
        value = values.setdefault(name, definition.default)
        if not math.isfinite(value):
            raise ResolutionError(f"{context} parameter '{name}' must be finite")
        if definition.domain is ParameterDomain.POSITIVE and value <= 0.0:
            raise ResolutionError(f"{context} parameter '{name}' must be positive")
    return values


@_cached_resolution
def resolve_scalar_lif(
    model: NeuronModel, bindings: Mapping[str, float] | None = None
) -> ResolvedScalarLIF:
    """Resolve a neuron into the slice-1 scalar stable-LIF specialization."""

    if model.threshold is None or model.hazard is not None:
        raise CapabilityError(
            "scalar threshold resolution requires a threshold neuron; use "
            "resolve_escape_lif for an intrinsic hazard"
        )

    membranes = [state for state in model.states if state.role is StateRole.MEMBRANE]
    if len(model.states) != 1 or len(membranes) != 1:
        raise CapabilityError(
            "the scalar delta specialization accepts exactly one membrane state; "
            "use folded-alpha resolution for an explicitly mapped receptor"
        )
    membrane = membranes[0]
    if set(model.dynamics) != {membrane.name}:
        raise ResolutionError(f"expected exactly one equation for d{membrane.name}/dt")
    if model.threshold.readout != membrane.name:
        raise CapabilityError("slice 1 threshold readout must be the scalar membrane state")
    if set(model.reset) != {membrane.name}:
        raise ResolutionError(f"expected exactly one reset for '{membrane.name}'")

    definitions = {parameter.name: parameter for parameter in model.parameters}
    values = {name: float(value) for name, value in (bindings or {}).items()}
    unknown_bindings = set(values).difference(definitions)
    if unknown_bindings:
        raise ResolutionError(f"unknown parameter binding(s): {', '.join(sorted(unknown_bindings))}")
    for name, definition in definitions.items():
        value = values.setdefault(name, definition.default)
        if not math.isfinite(value):
            raise ResolutionError(f"parameter '{name}' must be finite")
        if definition.domain is ParameterDomain.POSITIVE and value <= 0.0:
            raise ResolutionError(f"parameter '{name}' must be positive")

    symbols = {name: sympy.Symbol(name, real=True) for name in definitions}
    symbols[membrane.name] = sympy.Symbol(membrane.name, real=True)
    parameter_subs = {symbols[name]: value for name, value in values.items()}
    state_symbol = symbols[membrane.name]

    rhs = _expression(model.dynamics[membrane.name], symbols, "membrane dynamics")
    jacobian = sympy.diff(rhs, state_symbol)
    if state_symbol in jacobian.free_symbols:
        raise CapabilityError(
            f"nonlinear state dependence survives the Jacobian: {sympy.srepr(jacobian)}"
        )
    a_expr = sympy.factor(jacobian)
    b_expr = sympy.factor(rhs.subs(state_symbol, 0))
    if sympy.simplify(rhs - (a_expr * state_symbol + b_expr)) != 0:
        raise ResolutionError("internal affine extraction check failed")

    level_expr = _expression(model.threshold.level, symbols, "threshold level")
    if state_symbol in level_expr.free_symbols:
        raise CapabilityError("moving thresholds are reserved for a future crossing capability")
    reset_expr = _expression(model.reset[membrane.name], symbols, "reset expression")
    if state_symbol in reset_expr.free_symbols:
        raise CapabilityError("state-dependent reset is not yet supported by the scalar specialization")

    used_parameters = set().union(
        rhs.free_symbols, level_expr.free_symbols, reset_expr.free_symbols
    ).difference({state_symbol})
    declared_symbols = {symbols[name] for name in definitions}
    unused = declared_symbols.difference(used_parameters)
    if unused:
        raise ResolutionError(
            "unused parameter(s): " + ", ".join(sorted(str(item) for item in unused))
        )

    a = _number(a_expr, parameter_subs, "A[0,0]")
    b = _number(b_expr, parameter_subs, "b[0]")
    threshold = _number(level_expr, parameter_subs, "threshold")
    reset = _number(reset_expr, parameter_subs, "reset")
    refractory = model.refractory.duration if model.refractory else 0.0
    if not math.isfinite(refractory) or refractory < 0.0:
        raise ResolutionError("refractory duration must be finite and nonnegative")
    if a >= 0.0:
        raise CapabilityError(
            f"slice 1 requires a stable real decay rate, but resolved A[0,0] is {a}"
        )
    if reset >= threshold:
        raise CapabilityError("slice 1 requires reset to be strictly below threshold")

    asymptote = -b / a
    dispatch = DispatchForm.REACTIVE if asymptote <= threshold else DispatchForm.CLOSED_FORM

    canonical_model = {
        "name": model.name,
        "parameters": [(item.name, item.domain.value) for item in model.parameters],
        "states": [(item.name, item.role.value) for item in model.states],
        "rhs": sympy.srepr(rhs),
        "threshold": (model.threshold.readout, sympy.srepr(level_expr), "rising"),
        "reset": (membrane.name, sympy.srepr(reset_expr)),
        "refractory": None
        if model.refractory is None
        else (model.refractory.mode, model.refractory.duration),
    }
    model_hash = _hash(canonical_model)
    resolution_key = _hash(
        {
            "model_hash": model_hash,
            "a": sympy.srepr(a_expr),
            "b": sympy.srepr(b_expr),
            "state_order": [membrane.name],
            "capability": "scalar_stable_lif_v1",
            "regime": "a<0",
        }
    )
    binding_dag = lower_expressions(
        {
            "a": a_expr,
            "b": b_expr,
            "threshold": level_expr,
            "reset": reset_expr,
        },
        parameters=tuple(definitions),
    )
    delta = sympy.Symbol("Delta", real=True, nonnegative=True)
    x0 = sympy.Symbol("x0", real=True)
    next_v = sympy.exp(a_expr * delta) * x0 + b_expr * delta * phi1(a_expr * delta)
    reset_runtime = reset_expr.xreplace({state_symbol: x0})
    propagation_dag = lower_expressions(
        {
            "next_v": next_v,
            "clamped_v": x0,
            "reset_v": reset_runtime,
            "crossing_decay": a_expr,
            "crossing_affine": b_expr,
            "crossing_threshold": level_expr,
        },
        parameters=tuple(definitions),
        variables=("Delta", "x0"),
    )
    forbidden_drive_dependencies = set().union(
        a_expr.free_symbols,
        level_expr.free_symbols,
        reset_expr.free_symbols,
    )
    drive_parameters = tuple(
        definition.name
        for definition in model.parameters
        if symbols[definition.name] in b_expr.free_symbols
        and symbols[definition.name] not in forbidden_drive_dependencies
    )
    return ResolvedScalarLIF(
        model_name=model.name,
        model_hash=model_hash,
        resolution_key=resolution_key,
        state_name=membrane.name,
        readout_index=0,
        a_expr=sympy.srepr(a_expr),
        b_expr=sympy.srepr(b_expr),
        bindings=values,
        a=a,
        b=b,
        threshold=threshold,
        reset=reset,
        refractory=refractory,
        dispatch=dispatch,
        binding_dag=binding_dag,
        propagation_dag=propagation_dag,
        normal_roots=("next_v",),
        clamped_roots=("clamped_v",),
        reset_roots=("reset_v",),
        root_hint=ScalarLogRootHint(
            decay_root="crossing_decay",
            affine_root="crossing_affine",
            threshold_root="crossing_threshold",
        ),
        drive_parameters=drive_parameters,
        parameter_domains={
            definition.name: definition.domain for definition in model.parameters
        },
    )


@_cached_resolution
def resolve_stepped_neuron(
    model: NeuronModel,
    bindings: Mapping[str, float] | None = None,
    *,
    numerical: NumericalConfig | None = None,
) -> ResolvedSteppedNeuron:
    """Lower a bounded autonomous ODE neuron to the generic C stepped capability."""

    config = numerical or NumericalConfig()
    tolerance_values = (
        config.relative_tolerance,
        config.absolute_tolerance,
        config.initial_step,
        config.minimum_step,
        config.maximum_step,
        config.event_tolerance,
    )
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in tolerance_values)
        or not config.minimum_step <= config.initial_step <= config.maximum_step
        or not isinstance(config.maximum_steps, int)
        or isinstance(config.maximum_steps, bool)
        or not 0 < config.maximum_steps <= 0xFFFFFFFF
        or not isinstance(config.maximum_rhs_evaluations, int)
        or isinstance(config.maximum_rhs_evaluations, bool)
        or not 7 <= config.maximum_rhs_evaluations <= 0xFFFFFFFF
    ):
        raise ResolutionError("invalid generic stepped numerical configuration")
    states = tuple(model.states)
    state_names = tuple(state.name for state in states)
    membranes = [state for state in states if state.role is StateRole.MEMBRANE]
    if not states or len(states) > 8:
        raise CapabilityError("generic stepped neurons require between one and eight states")
    if len(membranes) != 1:
        raise CapabilityError("generic stepped neurons require exactly one membrane state")
    if set(model.dynamics) != set(state_names):
        raise ResolutionError("generic stepped neurons require one equation for every state")
    if set(model.reset) != set(state_names):
        raise ResolutionError("generic stepped neurons require a simultaneous reset for every state")
    if model.threshold.readout not in state_names:
        raise ResolutionError("threshold readout must name a declared state")
    reserved = set(state_names).union(parameter.name for parameter in model.parameters)
    collisions = reserved.intersection(_SAFE_INTRINSICS)
    if collisions:
        raise ResolutionError(
            "state and parameter names cannot shadow intrinsic function(s): "
            + ", ".join(sorted(collisions))
        )

    values = _parameter_values(model.parameters, bindings, "neuron")
    symbols = {
        parameter.name: sympy.Symbol(parameter.name, real=True)
        for parameter in model.parameters
    }
    symbols.update({name: sympy.Symbol(name, real=True) for name in state_names})
    state_symbols = {symbols[name] for name in state_names}
    parameter_symbols = {symbols[item.name] for item in model.parameters}
    parameter_subs = {symbols[name]: value for name, value in values.items()}

    rhs = {
        name: _expression(model.dynamics[name], symbols, f"d{name}/dt")
        for name in state_names
    }
    level_expr = _expression(model.threshold.level, symbols, "threshold level")
    if level_expr.free_symbols.intersection(state_symbols):
        raise CapabilityError(
            "the first generic stepped capability requires a fixed scalar threshold"
        )
    reset_exprs = {
        name: _expression(model.reset[name], symbols, f"reset for {name}")
        for name in state_names
    }
    readout = model.threshold.readout
    if reset_exprs[readout].free_symbols.intersection(state_symbols):
        raise CapabilityError(
            "the first generic stepped capability requires a state-independent readout reset"
        )
    used = set().union(
        *(expression.free_symbols for expression in rhs.values()),
        level_expr.free_symbols,
        *(expression.free_symbols for expression in reset_exprs.values()),
    )
    unused = parameter_symbols.difference(used)
    if unused:
        raise ResolutionError(
            "unused parameter(s): " + ", ".join(sorted(str(item) for item in unused))
        )

    threshold = _number(level_expr, parameter_subs, "threshold")
    reset = _number(reset_exprs[readout], parameter_subs, "readout reset")
    refractory = model.refractory.duration if model.refractory else 0.0
    if not math.isfinite(refractory) or refractory < 0.0:
        raise ResolutionError("refractory duration must be finite and nonnegative")
    if reset >= threshold:
        raise CapabilityError("generic stepped readout reset must be strictly below threshold")

    variable_symbols = {
        symbols[name]: sympy.Symbol(f"x{index}", real=True)
        for index, name in enumerate(state_names)
    }
    lowered: dict[str, sympy.Expr] = {}
    rhs_roots: list[str] = []
    reset_roots: list[str] = []
    for index, name in enumerate(state_names):
        rhs_name = f"rhs_{index}"
        reset_name = f"reset_{index}"
        lowered[rhs_name] = rhs[name].xreplace(variable_symbols)
        lowered[reset_name] = reset_exprs[name].xreplace(variable_symbols)
        rhs_roots.append(rhs_name)
        reset_roots.append(reset_name)
    dag = lower_expressions(
        lowered,
        parameters=tuple(parameter.name for parameter in model.parameters),
        variables=("Time",) + tuple(f"x{index}" for index in range(len(states))),
    )
    rhs_parameter_symbols = set().union(
        *(expression.free_symbols for expression in rhs.values())
    )
    drive_parameters = tuple(
        parameter.name
        for parameter in model.parameters
        if symbols[parameter.name] in rhs_parameter_symbols
        and symbols[parameter.name] not in level_expr.free_symbols
        and symbols[parameter.name] not in reset_exprs[readout].free_symbols
    )
    canonical_model = {
        "name": model.name,
        "parameters": [(item.name, item.domain.value) for item in model.parameters],
        "states": [(item.name, item.role.value) for item in states],
        "rhs": {name: sympy.srepr(rhs[name]) for name in state_names},
        "threshold": (readout, sympy.srepr(level_expr), "rising"),
        "reset": {name: sympy.srepr(reset_exprs[name]) for name in state_names},
        "refractory": None
        if model.refractory is None
        else (model.refractory.mode, model.refractory.duration),
    }
    model_hash = _hash(canonical_model)
    resolution_key = _hash(
        {
            "model_hash": model_hash,
            "capability": "generic_ode_dopri54_fixed_threshold_v1",
            "state_order": state_names,
            "numerical": config,
        }
    )
    roots = tuple(rhs_roots)
    return ResolvedSteppedNeuron(
        model_name=model.name,
        model_hash=model_hash,
        resolution_key=resolution_key,
        state_names=state_names,
        readout_index=state_names.index(readout),
        bindings=values,
        threshold=threshold,
        reset=reset,
        refractory=refractory,
        dispatch=DispatchForm.STEPPED,
        propagation_dag=dag,
        normal_roots=roots,
        clamped_roots=roots,
        reset_roots=tuple(reset_roots),
        numerical=config,
        drive_parameters=drive_parameters,
        parameter_domains={item.name: item.domain for item in model.parameters},
    )


@_cached_resolution
def resolve_reactive_if(
    model: NeuronModel,
    bindings: Mapping[str, float] | None = None,
) -> ResolvedReactiveIF:
    """Resolve a scalar deposit-triggered integrate-and-fire definition."""

    if model.reactive is None:
        raise CapabilityError("reactive integrate-and-fire requires a reactive block")
    if not isinstance(model.reactive, ReactiveMode):
        raise ResolutionError("invalid reactive integrate-and-fire mode")
    states = tuple(model.states)
    membranes = [state for state in states if state.role is StateRole.MEMBRANE]
    if len(states) != 1 or len(membranes) != 1:
        raise CapabilityError(
            "reactive integrate-and-fire requires exactly one membrane state"
        )
    membrane = membranes[0]
    if set(model.dynamics) != {membrane.name}:
        raise ResolutionError(f"expected exactly one equation for d{membrane.name}/dt")
    if model.threshold.readout != membrane.name:
        raise CapabilityError("reactive threshold readout must be the membrane state")
    if set(model.reset) != {membrane.name}:
        raise ResolutionError(f"expected exactly one reset for '{membrane.name}'")

    values = _parameter_values(model.parameters, bindings, "neuron")
    symbols = {
        parameter.name: sympy.Symbol(parameter.name, real=True)
        for parameter in model.parameters
    }
    symbols[membrane.name] = sympy.Symbol(membrane.name, real=True)
    state_symbol = symbols[membrane.name]
    parameter_subs = {symbols[name]: value for name, value in values.items()}
    rhs = _expression(model.dynamics[membrane.name], symbols, "membrane dynamics")
    if sympy.simplify(rhs).is_zero is not True:
        raise CapabilityError(
            "reactive integrate-and-fire requires zero between-event dynamics"
        )
    level_expr = _expression(model.threshold.level, symbols, "threshold level")
    reset_expr = _expression(model.reset[membrane.name], symbols, "reset expression")
    if state_symbol in level_expr.free_symbols:
        raise CapabilityError("reactive integrate-and-fire requires a fixed threshold")
    if state_symbol in reset_expr.free_symbols:
        raise CapabilityError("reactive integrate-and-fire requires a fixed reset")
    declared = {symbols[item.name] for item in model.parameters}
    used = level_expr.free_symbols.union(reset_expr.free_symbols)
    unused = declared.difference(used)
    if unused:
        raise ResolutionError(
            "unused parameter(s): " + ", ".join(sorted(str(item) for item in unused))
        )

    threshold = _number(level_expr, parameter_subs, "threshold")
    reset = _number(reset_expr, parameter_subs, "reset")
    refractory = model.refractory.duration if model.refractory else 0.0
    if not math.isfinite(refractory) or refractory < 0.0:
        raise ResolutionError("refractory duration must be finite and nonnegative")

    x0 = sympy.Symbol("x0", real=True)
    dag = lower_expressions(
        {
            "next_v": x0,
            "clamped_v": x0,
            "reset_v": reset_expr.xreplace({state_symbol: x0}),
        },
        parameters=tuple(item.name for item in model.parameters),
        variables=("Delta", "x0"),
    )
    canonical_model = {
        "name": model.name,
        "parameters": [(item.name, item.domain.value) for item in model.parameters],
        "states": [(item.name, item.role.value) for item in states],
        "rhs": sympy.srepr(rhs),
        "threshold": (membrane.name, sympy.srepr(level_expr), "rising"),
        "reset": (membrane.name, sympy.srepr(reset_expr)),
        "refractory": None
        if model.refractory is None
        else (model.refractory.mode, model.refractory.duration),
        "reactive": model.reactive.value,
    }
    model_hash = _hash(canonical_model)
    resolution_key = _hash(
        {
            "model_hash": model_hash,
            "capability": "scalar_reactive_if_v1",
            "mode": model.reactive.value,
        }
    )
    return ResolvedReactiveIF(
        model_name=model.name,
        model_hash=model_hash,
        resolution_key=resolution_key,
        state_names=(membrane.name,),
        readout_index=0,
        bindings=values,
        threshold=threshold,
        reset=reset,
        refractory=refractory,
        dispatch=DispatchForm.REACTIVE,
        propagation_dag=dag,
        normal_roots=("next_v",),
        clamped_roots=("clamped_v",),
        reset_roots=("reset_v",),
        reactive_mode=model.reactive,
        drive_parameters=(),
        parameter_domains={item.name: item.domain for item in model.parameters},
    )


@_cached_resolution
def resolve_adaptive_lif(
    model: NeuronModel, bindings: Mapping[str, float] | None = None
) -> ResolvedAdaptiveLIF:
    """Resolve one stable LIF plus an independent spike-triggered adaptation current."""

    if model.threshold is None or model.hazard is not None:
        raise CapabilityError(
            "adaptive threshold resolution requires a threshold neuron; use "
            "resolve_adaptive_escape_lif for an intrinsic hazard"
        )

    membranes = [state for state in model.states if state.role is StateRole.MEMBRANE]
    adaptations = [state for state in model.states if state.role is StateRole.ADAPTATION]
    if len(model.states) != 2 or len(membranes) != 1 or len(adaptations) != 1:
        raise CapabilityError(
            "adaptive LIF requires exactly one membrane and one adaptation state"
        )
    membrane = membranes[0]
    adaptation = adaptations[0]
    state_names = (membrane.name, adaptation.name)
    if set(model.dynamics) != set(state_names):
        raise ResolutionError("adaptive LIF requires one equation for each intrinsic state")
    if model.threshold.readout != membrane.name:
        raise CapabilityError("adaptive LIF requires the membrane as threshold readout")
    if set(model.reset) != set(state_names):
        raise ResolutionError(
            "adaptive LIF reset must update both membrane and adaptation state"
        )

    definitions = {parameter.name: parameter for parameter in model.parameters}
    values = _parameter_values(model.parameters, bindings, "neuron")
    symbols = {name: sympy.Symbol(name, real=True) for name in definitions}
    symbols[membrane.name] = sympy.Symbol(membrane.name, real=True)
    symbols[adaptation.name] = sympy.Symbol(adaptation.name, real=True)
    parameter_subs = {symbols[name]: value for name, value in values.items()}
    v_symbol = symbols[membrane.name]
    w_symbol = symbols[adaptation.name]
    state_symbols = {v_symbol, w_symbol}

    v_rhs = _expression(model.dynamics[membrane.name], symbols, "membrane dynamics")
    w_rhs = _expression(model.dynamics[adaptation.name], symbols, "adaptation dynamics")
    a_expr = sympy.factor(sympy.diff(v_rhs, v_symbol))
    coupling_expr = sympy.factor(sympy.diff(v_rhs, w_symbol))
    b_expr = sympy.factor(v_rhs.subs({v_symbol: 0, w_symbol: 0}))
    q_expr = sympy.factor(sympy.diff(w_rhs, w_symbol))
    feedback_expr = sympy.factor(sympy.diff(w_rhs, v_symbol))
    w_offset_expr = sympy.factor(w_rhs.subs({v_symbol: 0, w_symbol: 0}))
    jacobian_terms = (a_expr, coupling_expr, q_expr, feedback_expr)
    if any(state_symbols.intersection(term.free_symbols) for term in jacobian_terms):
        raise CapabilityError("adaptive LIF dynamics must have a state-independent Jacobian")
    if sympy.simplify(
        v_rhs - (a_expr * v_symbol + coupling_expr * w_symbol + b_expr)
    ) != 0:
        raise CapabilityError("membrane dynamics must be affine in voltage and adaptation")
    if sympy.simplify(feedback_expr) != 0 or sympy.simplify(w_offset_expr) != 0:
        raise CapabilityError(
            "the first adaptive capability requires independent zero-centered adaptation decay"
        )
    if sympy.simplify(w_rhs - q_expr * w_symbol) != 0:
        raise CapabilityError("adaptation dynamics must be a single exponential decay")

    level_expr = _expression(model.threshold.level, symbols, "threshold level")
    if state_symbols.intersection(level_expr.free_symbols):
        raise CapabilityError("moving thresholds require a future crossing capability")
    reset_v_expr = _expression(model.reset[membrane.name], symbols, "membrane reset")
    reset_w_expr = _expression(model.reset[adaptation.name], symbols, "adaptation reset")
    if state_symbols.intersection(reset_v_expr.free_symbols):
        raise CapabilityError("adaptive LIF membrane reset must be state independent")
    jump_expr = sympy.factor(reset_w_expr - w_symbol)
    if state_symbols.intersection(jump_expr.free_symbols) or sympy.simplify(
        reset_w_expr - (w_symbol + jump_expr)
    ) != 0:
        raise CapabilityError("adaptation reset must have the form w <- w + beta")

    parameter_symbols = {symbols[name] for name in definitions}
    used_parameters = set().union(
        v_rhs.free_symbols,
        w_rhs.free_symbols,
        level_expr.free_symbols,
        reset_v_expr.free_symbols,
        reset_w_expr.free_symbols,
    ).intersection(parameter_symbols)
    unused = parameter_symbols.difference(used_parameters)
    if unused:
        raise ResolutionError(
            "unused parameter(s): " + ", ".join(sorted(str(item) for item in unused))
        )

    a = _number(a_expr, parameter_subs, "membrane decay")
    b = _number(b_expr, parameter_subs, "membrane affine input")
    coupling = _number(coupling_expr, parameter_subs, "adaptation coupling")
    q = _number(q_expr, parameter_subs, "adaptation decay")
    jump = _number(jump_expr, parameter_subs, "adaptation jump")
    threshold = _number(level_expr, parameter_subs, "threshold")
    reset = _number(reset_v_expr, parameter_subs, "membrane reset")
    refractory = model.refractory.duration if model.refractory else 0.0
    if not math.isfinite(refractory) or refractory < 0.0:
        raise ResolutionError("refractory duration must be finite and nonnegative")
    if a >= 0.0 or q >= 0.0:
        raise CapabilityError("adaptive LIF requires two stable real negative decay rates")
    if a == q:
        raise CapabilityError(
            "equal membrane and adaptation rates require the reserved repeated-real-mode variant"
        )
    if coupling >= 0.0:
        raise CapabilityError("adaptation must couple inhibitively into the membrane")
    if jump < 0.0:
        raise CapabilityError("the spike-triggered adaptation jump must be nonnegative")
    if reset >= threshold:
        raise CapabilityError("adaptive LIF reset must be strictly below threshold")

    delta = sympy.Symbol("Delta", real=True, nonnegative=True)
    x0, x1 = sympy.symbols("x0 x1", real=True)
    ea = sympy.exp(a_expr * delta)
    eq = sympy.exp(q_expr * delta)
    rate_gap = (q_expr - a_expr) * delta
    next_w = eq * x1
    next_v = (
        ea * x0
        + b_expr * delta * phi1(a_expr * delta)
        + coupling_expr * ea * x1 * delta * phi1(rate_gap)
    )
    asymptote = -b_expr / a_expr
    coefficient_two = coupling_expr * x1 / (q_expr - a_expr)
    coefficient_one = x0 - asymptote - coefficient_two
    crossing_limit = level_expr - asymptote
    crossing_coefficient_one = -coefficient_one
    crossing_coefficient_two = -coefficient_two
    crossing = level_expr - next_v
    crossing_derivative = -(a_expr * next_v + coupling_expr * next_w + b_expr)
    reset_v_runtime = reset_v_expr.xreplace({v_symbol: x0, w_symbol: x1})
    reset_w_runtime = reset_w_expr.xreplace({v_symbol: x0, w_symbol: x1})
    parameter_names = tuple(definitions)
    propagation_dag = lower_expressions(
        {
            "clamped_v": x0,
            "clamped_w": next_w,
            "next_v": next_v,
            "next_w": next_w,
            "reset_v": reset_v_runtime,
            "reset_w": reset_w_runtime,
            "crossing_g": crossing,
            "crossing_g_prime": crossing_derivative,
            "crossing_limit": crossing_limit,
            "crossing_coefficient_one": crossing_coefficient_one,
            "crossing_coefficient_two": crossing_coefficient_two,
            "crossing_rate_one": a_expr,
            "crossing_rate_two": q_expr,
            # Equation-derived voltage trajectory used by any consumer that
            # needs repeated readout evaluation (for example an integrated
            # hazard).  These are deliberately separate from the threshold
            # crossing coefficients, whose signs and limit include the level.
            "trajectory_limit": asymptote,
            "trajectory_coefficient_one": coefficient_one,
            "trajectory_coefficient_two": coefficient_two,
            "trajectory_rate_one": a_expr,
            "trajectory_rate_two": q_expr,
        },
        parameters=parameter_names,
        variables=("Delta", "x0", "x1"),
    )

    canonical_model = {
        "name": model.name,
        "parameters": [(item.name, item.domain.value) for item in model.parameters],
        "states": [(item.name, item.role.value) for item in model.states],
        "rhs": (sympy.srepr(v_rhs), sympy.srepr(w_rhs)),
        "threshold": (model.threshold.readout, sympy.srepr(level_expr), "rising"),
        "reset": (
            (membrane.name, sympy.srepr(reset_v_expr)),
            (adaptation.name, sympy.srepr(reset_w_expr)),
        ),
        "refractory": None
        if model.refractory is None
        else (model.refractory.mode, model.refractory.duration),
    }
    model_hash = _hash(canonical_model)
    resolution_key = _hash(
        {
            "model_hash": model_hash,
            "capability": "stable_lif_one_adaptation_two_real_exp_v2",
            "regime": ("a<0", "q<0", "a!=q", "coupling<0", "jump>=0"),
            "state_order": state_names,
            "dispatch": DispatchForm.ROOT_FIND.value,
        }
    )
    root_hint = TwoExpRootHint(
        g_root="crossing_g",
        g_prime_root="crossing_g_prime",
        limit_root="crossing_limit",
        coefficient_one_root="crossing_coefficient_one",
        coefficient_two_root="crossing_coefficient_two",
        rate_one_root="crossing_rate_one",
        rate_two_root="crossing_rate_two",
        relative_tolerance=1e-10,
        fastest_time_constant=min(-1.0 / a, -1.0 / q),
        iteration_cap=192,
    )
    forbidden_drive_dependencies = set().union(
        a_expr.free_symbols,
        coupling_expr.free_symbols,
        q_expr.free_symbols,
        level_expr.free_symbols,
        reset_v_expr.free_symbols,
        reset_w_expr.free_symbols,
    )
    drive_parameters = tuple(
        definition.name
        for definition in model.parameters
        if symbols[definition.name] in b_expr.free_symbols
        and symbols[definition.name] not in forbidden_drive_dependencies
    )
    parameter_domains = {
        definition.name: definition.domain for definition in model.parameters
    }
    return ResolvedAdaptiveLIF(
        model_name=model.name,
        model_hash=model_hash,
        resolution_key=resolution_key,
        state_names=state_names,
        readout_index=state_names.index(membrane.name),
        bindings=values,
        a=a,
        b=b,
        coupling=coupling,
        adaptation_decay=q,
        adaptation_jump=jump,
        threshold=threshold,
        reset=reset,
        refractory=refractory,
        dispatch=DispatchForm.ROOT_FIND,
        propagation_dag=propagation_dag,
        normal_roots=("next_v", "next_w"),
        clamped_roots=("clamped_v", "clamped_w"),
        reset_roots=("reset_v", "reset_w"),
        root_hint=root_hint,
        drive_parameters=drive_parameters,
        parameter_domains=parameter_domains,
    )


def _escape_surrogate(
    model: NeuronModel,
    bindings: Mapping[str, float] | None,
) -> tuple[NeuronModel, dict[str, float], ResolvedHazard]:
    """Separate exact deterministic dynamics from an exponential voltage hazard."""

    if model.hazard is None or model.threshold is not None:
        raise CapabilityError("escape resolution requires a hazard and no hard threshold")
    if model.reactive is not None:
        raise CapabilityError("intrinsic hazards cannot use event-batched reactive dynamics")
    membranes = [state for state in model.states if state.role is StateRole.MEMBRANE]
    if len(membranes) != 1:
        raise CapabilityError("an intrinsic voltage hazard requires one membrane state")
    membrane = membranes[0]
    values = _parameter_values(model.parameters, bindings, "neuron")
    definitions = {parameter.name: parameter for parameter in model.parameters}
    symbols = {name: sympy.Symbol(name, real=True) for name in definitions}
    symbols.update({state.name: sympy.Symbol(state.name, real=True) for state in model.states})
    state_symbols = {symbols[state.name] for state in model.states}
    parameter_symbols = {symbols[name] for name in definitions}
    rate = _expression(model.hazard.rate, symbols, "hazard rate")
    exponential_atoms = tuple(rate.atoms(sympy.exp))
    if len(exponential_atoms) != 1:
        raise CapabilityError(
            "the first intrinsic hazard capability requires one exponential term"
        )
    exponential = exponential_atoms[0]
    exponent = sympy.expand(exponential.args[0])
    voltage = symbols[membrane.name]
    voltage_gain_expr = sympy.simplify(sympy.diff(exponent, voltage))
    exponent_offset_expr = sympy.simplify(exponent - voltage_gain_expr * voltage)
    prefactor_expr = sympy.simplify(rate / exponential)
    if (
        state_symbols.intersection(voltage_gain_expr.free_symbols)
        or state_symbols.intersection(exponent_offset_expr.free_symbols)
        or state_symbols.intersection(prefactor_expr.free_symbols)
        or sympy.simplify(
            rate - prefactor_expr * sympy.exp(
                voltage_gain_expr * voltage + exponent_offset_expr
            )
        )
        != 0
    ):
        raise CapabilityError(
            "hazard log-rate must be affine in the membrane voltage only"
        )
    substitutions = {symbols[name]: value for name, value in values.items()}
    prefactor = _number(prefactor_expr, substitutions, "hazard prefactor")
    voltage_gain = _number(voltage_gain_expr, substitutions, "hazard voltage gain")
    exponent_offset = _number(
        exponent_offset_expr, substitutions, "hazard exponent offset"
    )
    if prefactor <= 0.0:
        raise CapabilityError("hazard prefactor must be strictly positive")
    if voltage_gain <= 0.0:
        raise CapabilityError("escape hazard must increase with membrane voltage")
    log_scale = math.log(prefactor) + exponent_offset
    if not math.isfinite(log_scale):
        raise ResolutionError("resolved hazard log scale must be finite")

    deterministic_text = tuple(model.dynamics.values()) + tuple(model.reset.values())
    deterministic_names = set().union(
        *(
            set(_TOKENS.findall(_DECIMAL_LITERAL.sub("", text)))
            for text in deterministic_text
        )
    )
    deterministic_parameters = tuple(
        definition
        for definition in model.parameters
        if definition.name in deterministic_names
    )
    used_parameters = set(rate.free_symbols)
    for text in deterministic_text:
        used_parameters.update(
            _expression(text, symbols, "escape dynamics/reset").free_symbols
        )
    used_parameters.intersection_update(parameter_symbols)
    unused = parameter_symbols.difference(used_parameters)
    if unused:
        raise ResolutionError(
            "unused parameter(s): " + ", ".join(sorted(str(item) for item in unused))
        )
    surrogate = replace(
        model,
        parameters=deterministic_parameters,
        threshold=ThresholdDefinition(membrane.name, repr(sys.float_info.max)),
        hazard=None,
    )
    hazard = ResolvedHazard(
        kind=HazardKind.EXPONENTIAL_VOLTAGE,
        log_scale=log_scale,
        voltage_gain=voltage_gain,
    )
    return surrogate, values, hazard


@_cached_resolution
def resolve_escape_lif(
    model: NeuronModel, bindings: Mapping[str, float] | None = None
) -> ResolvedScalarLIF:
    """Resolve a stable scalar LIF with an intrinsic exponential escape hazard."""

    surrogate, values, hazard = _escape_surrogate(model, bindings)
    deterministic_bindings = {
        definition.name: values[definition.name]
        for definition in surrogate.parameters
    }
    base = resolve_scalar_lif(surrogate, deterministic_bindings)
    model_hash = _hash(
        {
            "deterministic_model_hash": base.model_hash,
            "parameters": [
                (item.name, item.domain.value) for item in model.parameters
            ],
            "hazard": (
                hazard.kind.value,
                sympy.srepr(
                    _expression(
                        model.hazard.rate,
                        {
                            **{
                                item.name: sympy.Symbol(item.name, real=True)
                                for item in model.parameters
                            },
                            **{
                                item.name: sympy.Symbol(item.name, real=True)
                                for item in model.states
                            },
                        },
                        "hazard rate",
                    )
                ),
            ),
        }
    )
    return replace(
        base,
        model_name=model.name,
        model_hash=model_hash,
        resolution_key=_hash(
            {
                "model_hash": model_hash,
                "base": base.resolution_key,
                "capability": "integrated_exponential_voltage_hazard_v1",
            }
        ),
        bindings=values,
        threshold=sys.float_info.max,
        dispatch=DispatchForm.ROOT_FIND,
        parameter_domains={item.name: item.domain for item in model.parameters},
        hazard=hazard,
    )


@_cached_resolution
def resolve_adaptive_escape_lif(
    model: NeuronModel, bindings: Mapping[str, float] | None = None
) -> ResolvedAdaptiveLIF:
    """Resolve exact adaptive-LIF state evolution with an escape hazard."""

    surrogate, values, hazard = _escape_surrogate(model, bindings)
    deterministic_bindings = {
        definition.name: values[definition.name]
        for definition in surrogate.parameters
    }
    base = resolve_adaptive_lif(surrogate, deterministic_bindings)
    # The modal representation is substantially cheaper when a trajectory is
    # sampled many times.  Close rates can produce large cancelling modal
    # coefficients, so retain the stable phi1 expression evaluator there.
    rate_scale = max(abs(base.a), abs(base.adaptation_decay))
    if abs(base.adaptation_decay - base.a) > 1e-5 * rate_scale:
        hazard = replace(
            hazard,
            trajectory_limit_root="trajectory_limit",
            trajectory_coefficient_roots=(
                "trajectory_coefficient_one",
                "trajectory_coefficient_two",
            ),
            trajectory_rate_roots=(
                "trajectory_rate_one",
                "trajectory_rate_two",
            ),
        )
    model_hash = _hash(
        {
            "deterministic_model_hash": base.model_hash,
            "parameters": [
                (item.name, item.domain.value) for item in model.parameters
            ],
            "hazard": (
                hazard.kind.value,
                sympy.srepr(
                    _expression(
                        model.hazard.rate,
                        {
                            **{
                                item.name: sympy.Symbol(item.name, real=True)
                                for item in model.parameters
                            },
                            **{
                                item.name: sympy.Symbol(item.name, real=True)
                                for item in model.states
                            },
                        },
                        "hazard rate",
                    )
                ),
            ),
        }
    )
    return replace(
        base,
        model_name=model.name,
        model_hash=model_hash,
        resolution_key=_hash(
            {
                "model_hash": model_hash,
                "base": base.resolution_key,
                "capability": "adaptive_integrated_exponential_voltage_hazard_v2",
            }
        ),
        bindings=values,
        threshold=sys.float_info.max,
        dispatch=DispatchForm.ROOT_FIND,
        parameter_domains={item.name: item.domain for item in model.parameters},
        hazard=hazard,
    )


@_cached_resolution
def resolve_folded_alpha_lif(
    neuron: NeuronModel,
    synapse: SynapseModel,
    *,
    receptor: str,
    output: str,
    neuron_bindings: Mapping[str, float] | None = None,
    synapse_bindings: Mapping[str, float] | None = None,
) -> ResolvedAlphaLIF:
    """Resolve one intrinsic scalar LIF plus one structurally recognized alpha kernel.

    This milestone produces propagation and deposit programs only. The accepted
    augmented trajectory is classified ROOT_FIND, but spike prediction is enabled
    by the subsequent certified crossing milestone.
    """

    membranes = [state for state in neuron.states if state.role is StateRole.MEMBRANE]
    receptors = [state for state in neuron.states if state.role is StateRole.RECEPTOR]
    if len(membranes) != 1:
        raise CapabilityError("folded alpha requires exactly one membrane state")
    if len(receptors) != 1 or receptors[0].name != receptor:
        raise CapabilityError(
            "folded alpha currently requires exactly one explicitly mapped receptor"
        )
    membrane = membranes[0]
    if len(neuron.states) != 2:
        raise CapabilityError("additional intrinsic state is outside the one-alpha capability")
    if set(neuron.dynamics) != {membrane.name}:
        raise ResolutionError(
            "an intrinsic alpha-receiving neuron must define dynamics only for its membrane"
        )
    if neuron.threshold.readout != membrane.name:
        raise CapabilityError("slice 1 requires a fixed scalar membrane threshold")
    if set(neuron.reset) != {membrane.name}:
        raise ResolutionError(f"expected exactly one reset for '{membrane.name}'")

    neuron_values = _parameter_values(neuron.parameters, neuron_bindings, "neuron")
    neuron_symbols = {
        definition.name: sympy.Symbol(definition.name, real=True)
        for definition in neuron.parameters
    }
    neuron_symbols[membrane.name] = sympy.Symbol(membrane.name, real=True)
    neuron_symbols[receptor] = sympy.Symbol(receptor, real=True)
    neuron_subs = {
        neuron_symbols[name]: value for name, value in neuron_values.items()
    }
    v_symbol = neuron_symbols[membrane.name]
    receptor_symbol = neuron_symbols[receptor]
    rhs = _expression(neuron.dynamics[membrane.name], neuron_symbols, "membrane dynamics")
    a_expr = sympy.factor(sympy.diff(rhs, v_symbol))
    coupling_expr = sympy.factor(sympy.diff(rhs, receptor_symbol))
    if {v_symbol, receptor_symbol}.intersection(a_expr.free_symbols):
        raise CapabilityError("membrane dynamics are nonlinear in intrinsic state")
    if {v_symbol, receptor_symbol}.intersection(coupling_expr.free_symbols):
        raise CapabilityError("receptor coupling is nonlinear in intrinsic state")
    b_expr = sympy.factor(rhs.subs({v_symbol: 0, receptor_symbol: 0}))
    if sympy.simplify(rhs - (a_expr * v_symbol + coupling_expr * receptor_symbol + b_expr)) != 0:
        raise CapabilityError("membrane dynamics must be affine in membrane and receptor")

    level_expr = _expression(neuron.threshold.level, neuron_symbols, "threshold level")
    reset_expr = _expression(neuron.reset[membrane.name], neuron_symbols, "reset expression")
    if {v_symbol, receptor_symbol}.intersection(level_expr.free_symbols):
        raise CapabilityError("moving thresholds are outside the one-alpha capability")
    if {v_symbol, receptor_symbol}.intersection(reset_expr.free_symbols):
        raise CapabilityError("state-dependent reset is outside the one-alpha capability")

    neuron_parameter_symbols = {
        neuron_symbols[definition.name] for definition in neuron.parameters
    }
    neuron_used = set().union(
        rhs.free_symbols, level_expr.free_symbols, reset_expr.free_symbols
    ).intersection(neuron_parameter_symbols)
    neuron_unused = neuron_parameter_symbols.difference(neuron_used)
    if neuron_unused:
        raise ResolutionError(
            "unused neuron parameter(s): "
            + ", ".join(sorted(str(item) for item in neuron_unused))
        )

    a = _number(a_expr, neuron_subs, "membrane decay")
    b = _number(b_expr, neuron_subs, "membrane affine input")
    coupling = _number(coupling_expr, neuron_subs, "receptor coupling")
    threshold = _number(level_expr, neuron_subs, "threshold")
    reset = _number(reset_expr, neuron_subs, "reset")
    refractory = neuron.refractory.duration if neuron.refractory else 0.0
    if not math.isfinite(refractory) or refractory < 0.0:
        raise ResolutionError("refractory duration must be finite and nonnegative")
    if a >= 0.0:
        raise CapabilityError("folded alpha requires a stable real membrane decay rate")
    if coupling == 0.0:
        raise CapabilityError("the mapped receptor must have nonzero membrane coupling")
    if reset >= threshold:
        raise CapabilityError("slice 1 requires reset to be strictly below threshold")

    if len(synapse.states) != 2 or set(synapse.dynamics) != set(synapse.states):
        raise CapabilityError("the supported alpha kernel requires exactly two state equations")
    if output not in synapse.outputs:
        raise ResolutionError(f"synapse output '{output}' does not exist")
    if synapse.spike_target not in synapse.states:
        raise ResolutionError("on_spike target must be a declared synapse state")

    synapse_values = _parameter_values(synapse.parameters, synapse_bindings, "synapse")
    synapse_symbols = {
        definition.name: sympy.Symbol(definition.name, real=True)
        for definition in synapse.parameters
    }
    for name in synapse.states:
        synapse_symbols[name] = sympy.Symbol(name, real=True)
    weight_symbol = sympy.Symbol("w", real=True)
    expression_symbols = dict(synapse_symbols)
    expression_symbols["w"] = weight_symbol
    synapse_subs = {
        synapse_symbols[name]: value for name, value in synapse_values.items()
    }

    output_expr = _expression(synapse.outputs[output], synapse_symbols, "synapse output")
    if not output_expr.is_Symbol or str(output_expr) not in synapse.states:
        raise CapabilityError("the supported alpha output must directly expose one state")
    s_name = str(output_expr)
    z_name = synapse.spike_target
    if s_name == z_name or {s_name, z_name} != set(synapse.states):
        raise CapabilityError("the alpha deposit target must be the non-output kernel state")
    s_symbol = synapse_symbols[s_name]
    z_symbol = synapse_symbols[z_name]
    ds = _expression(synapse.dynamics[s_name], synapse_symbols, f"d{s_name}/dt")
    dz = _expression(synapse.dynamics[z_name], synapse_symbols, f"d{z_name}/dt")
    q_s = sympy.factor(sympy.diff(ds, s_symbol))
    q_z = sympy.factor(sympy.diff(dz, z_symbol))
    if sympy.simplify(q_s - q_z) != 0:
        raise CapabilityError("alpha kernel states must share one repeated decay rate")
    if sympy.simplify(sympy.diff(ds, z_symbol) - 1) != 0:
        raise CapabilityError("alpha output state must be driven by its deposit state with gain one")
    if sympy.simplify(sympy.diff(dz, s_symbol)) != 0:
        raise CapabilityError("alpha deposit state cannot be driven by the output state")
    if sympy.simplify(ds - (q_s * s_symbol + z_symbol)) != 0 or sympy.simplify(
        dz - q_s * z_symbol
    ) != 0:
        raise CapabilityError("synapse equations are outside the supported alpha structure")

    update_expr = _expression(synapse.spike_update, expression_symbols, "on_spike update")
    increment_expr = sympy.factor(update_expr - z_symbol)
    deposit_scale_expr = sympy.factor(sympy.diff(increment_expr, weight_symbol))
    if {s_symbol, z_symbol}.intersection(increment_expr.free_symbols) or sympy.simplify(
        increment_expr - deposit_scale_expr * weight_symbol
    ) != 0:
        raise CapabilityError("alpha on_spike update must add a state-independent linear weight")
    if sympy.simplify(deposit_scale_expr - q_s**2) != 0:
        raise CapabilityError("alpha deposits must use the unit-area repeated-rate normalization")

    synapse_parameter_symbols = {
        synapse_symbols[definition.name] for definition in synapse.parameters
    }
    synapse_used = set().union(
        ds.free_symbols,
        dz.free_symbols,
        update_expr.free_symbols,
        output_expr.free_symbols,
    ).intersection(synapse_parameter_symbols)
    synapse_unused = synapse_parameter_symbols.difference(synapse_used)
    if synapse_unused:
        raise ResolutionError(
            "unused synapse parameter(s): "
            + ", ".join(sorted(str(item) for item in synapse_unused))
        )

    q = _number(q_s, synapse_subs, "synaptic decay")
    if q >= 0.0:
        raise CapabilityError("folded alpha requires a stable real synaptic decay rate")

    neuron_runtime_symbols = {
        name: sympy.Symbol(f"neuron.{name}", real=True) for name in neuron_values
    }
    synapse_runtime_symbols = {
        name: sympy.Symbol(f"synapse.{name}", real=True) for name in synapse_values
    }
    neuron_namespace = {
        neuron_symbols[name]: symbol for name, symbol in neuron_runtime_symbols.items()
    }
    synapse_namespace = {
        synapse_symbols[name]: symbol for name, symbol in synapse_runtime_symbols.items()
    }
    a_runtime = a_expr.xreplace(neuron_namespace)
    b_runtime = b_expr.xreplace(neuron_namespace)
    coupling_runtime = coupling_expr.xreplace(neuron_namespace)
    q_runtime = q_s.xreplace(synapse_namespace)
    deposit_scale_runtime = deposit_scale_expr.xreplace(synapse_namespace)

    delta = sympy.Symbol("Delta", real=True, nonnegative=True)
    x0, x1, x2 = sympy.symbols("x0 x1 x2", real=True)
    ea = sympy.exp(a_runtime * delta)
    eq = sympy.exp(q_runtime * delta)
    rate_difference = q_runtime - a_runtime
    rate_gap = rate_difference * delta
    integral_s = delta * phi1(rate_gap)
    integral_z = delta**2 * phi1_derivative(rate_gap)
    next_v = (
        ea * x0
        + ((ea - 1) / a_runtime) * b_runtime
        + coupling_runtime * ea * (x1 * integral_s + x2 * integral_z)
    )
    next_s = eq * (x1 + x2 * delta)
    next_z = eq * x2
    threshold_runtime = level_expr.xreplace(neuron_namespace)
    asymptote = -b_runtime / a_runtime
    crossing = threshold_runtime - next_v
    crossing_derivative = -(
        a_runtime * next_v + b_runtime + coupling_runtime * next_s
    )
    runtime_parameter_names = tuple(
        [str(symbol) for symbol in neuron_runtime_symbols.values()]
        + [str(symbol) for symbol in synapse_runtime_symbols.values()]
    )
    propagation_expressions = {
        "clamped_v": x0,
        "clamped_s": next_s,
        "clamped_z": next_z,
        "next_v": next_v,
        "next_s": next_s,
        "next_z": next_z,
        "reset_v": reset_expr.xreplace(neuron_namespace).xreplace({v_symbol: x0}),
        "reset_s": x1,
        "reset_z": x2,
        "crossing_g": crossing,
        "crossing_g_prime": crossing_derivative,
    }
    if a != q:
        membrane_coefficient = (
            x0 - asymptote
            - coupling_runtime * x1 / rate_difference
            + coupling_runtime * x2 / rate_difference**2
        )
        synapse_constant = (
            coupling_runtime * x1 / rate_difference
            - coupling_runtime * x2 / rate_difference**2
        )
        synapse_linear = coupling_runtime * x2 / rate_difference
        extremum_a = a_runtime * membrane_coefficient
        extremum_d = synapse_linear + q_runtime * synapse_constant
        extremum_e = q_runtime * synapse_linear
        extremum_p = a_runtime - q_runtime
        propagation_expressions.update(
            {
                "extremum_f": extremum_a * sympy.exp(extremum_p * delta)
                + extremum_d
                + extremum_e * delta,
                "extremum_f_prime": extremum_a
                * extremum_p
                * sympy.exp(extremum_p * delta)
                + extremum_e,
                "crossing_asymptote": asymptote,
                "crossing_membrane_coefficient": membrane_coefficient,
                "crossing_synapse_constant": synapse_constant,
                "crossing_synapse_linear": synapse_linear,
                "crossing_membrane_rate": a_runtime,
                "crossing_synapse_rate": q_runtime,
                "crossing_threshold": threshold_runtime,
            }
        )
    else:
        propagation_expressions.update(
            {
                "crossing_limit": threshold_runtime - asymptote,
                "crossing_rate_0": a_runtime,
                "crossing_coefficient_0_0": -(x0 - asymptote),
                "crossing_coefficient_0_1": -coupling_runtime * x1,
                "crossing_coefficient_0_2": -coupling_runtime * x2 / 2,
            }
        )
    propagation_dag = lower_expressions(
        propagation_expressions,
        parameters=runtime_parameter_names,
        variables=("Delta", "x0", "x1", "x2"),
    )
    deposit_dag = lower_expressions(
        {"increment": deposit_scale_runtime * weight_symbol},
        parameters=runtime_parameter_names,
        variables=("w",),
    )
    runtime_bindings = {
        **{str(neuron_runtime_symbols[name]): value for name, value in neuron_values.items()},
        **{str(synapse_runtime_symbols[name]): value for name, value in synapse_values.items()},
    }

    canonical = {
        "neuron": neuron.name,
        "synapse": synapse.name,
        "neuron_parameters": [
            (item.name, item.domain.value) for item in neuron.parameters
        ],
        "synapse_parameters": [
            (item.name, item.domain.value) for item in synapse.parameters
        ],
        "mapping": (output, receptor),
        "membrane_rhs": sympy.srepr(rhs),
        "threshold": sympy.srepr(level_expr),
        "reset": sympy.srepr(reset_expr),
        "synapse_rhs": (sympy.srepr(ds), sympy.srepr(dz)),
        "deposit": sympy.srepr(update_expr),
        "state_order": (membrane.name, f"{synapse.name}.{s_name}", f"{synapse.name}.{z_name}"),
        "refractory": None
        if neuron.refractory is None
        else (neuron.refractory.mode, neuron.refractory.duration),
    }
    model_hash = _hash(canonical)
    resolution_key = _hash(
        {
            "model_hash": model_hash,
            "capability": "stable_lif_one_folded_alpha_v2",
            "regime": ("a<0", "q<0", "a==q" if a == q else "a!=q"),
            "tier": SynapseTier.FOLDED_SHARED.value,
            "dispatch": DispatchForm.ROOT_FIND.value,
        }
    )
    root_hint: RootFindHint | ExpPolyRootHint = (
        RootFindHint(
            g_root="crossing_g",
            g_prime_root="crossing_g_prime",
            extremum_root="extremum_f",
            extremum_prime_root="extremum_f_prime",
            asymptote_root="crossing_asymptote",
            membrane_coefficient_root="crossing_membrane_coefficient",
            synapse_constant_root="crossing_synapse_constant",
            synapse_linear_root="crossing_synapse_linear",
            membrane_rate_root="crossing_membrane_rate",
            synapse_rate_root="crossing_synapse_rate",
            threshold_root="crossing_threshold",
            relative_tolerance=1e-10,
            fastest_time_constant=min(-1.0 / a, -1.0 / q),
            iteration_cap=192,
        )
        if a != q
        else ExpPolyRootHint(
            limit_root="crossing_limit",
            rate_roots=("crossing_rate_0",),
            coefficient_roots=(
                (
                    "crossing_coefficient_0_0",
                    "crossing_coefficient_0_1",
                    "crossing_coefficient_0_2",
                ),
            ),
            relative_tolerance=1e-10,
            fastest_time_constant=-1.0 / a,
            iteration_cap=2048,
        )
    )
    forbidden_drive_dependencies = set().union(
        a_expr.free_symbols,
        coupling_expr.free_symbols,
        level_expr.free_symbols,
        reset_expr.free_symbols,
    )
    drive_parameters = tuple(
        str(neuron_runtime_symbols[definition.name])
        for definition in neuron.parameters
        if neuron_symbols[definition.name] in b_expr.free_symbols
        and neuron_symbols[definition.name] not in forbidden_drive_dependencies
    )
    parameter_domains = {
        str(neuron_runtime_symbols[definition.name]): definition.domain
        for definition in neuron.parameters
    } | {
        str(synapse_runtime_symbols[definition.name]): definition.domain
        for definition in synapse.parameters
    }
    return ResolvedAlphaLIF(
        neuron_name=neuron.name,
        synapse_name=synapse.name,
        model_hash=model_hash,
        resolution_key=resolution_key,
        state_names=canonical["state_order"],
        readout_index=canonical["state_order"].index(membrane.name),
        bindings=runtime_bindings,
        a=a,
        b=b,
        coupling=coupling,
        synaptic_decay=q,
        threshold=threshold,
        reset=reset,
        refractory=refractory,
        dispatch=DispatchForm.ROOT_FIND,
        tier=SynapseTier.FOLDED_SHARED,
        propagation_dag=propagation_dag,
        deposit_dag=deposit_dag,
        normal_roots=("next_v", "next_s", "next_z"),
        clamped_roots=("clamped_v", "clamped_s", "clamped_z"),
        reset_roots=("reset_v", "reset_s", "reset_z"),
        deposit_root="increment",
        deposit_index=2,
        root_hint=root_hint,
        drive_parameters=drive_parameters,
        parameter_domains=parameter_domains,
    )


@_cached_resolution
def resolve_per_edge_lif(
    neuron: NeuronModel,
    instances: Sequence[PerEdgeSynapseInstance],
    *,
    neuron_bindings: Mapping[str, float] | None = None,
) -> ResolvedPerEdgeLIF:
    """Resolve stable scalar LIF plus bounded edge-scoped linear kernels.

    The initial capability recognizes scalar exponential and unit-area alpha
    kernels. Structurally identical kernels with equal resolved rates share a
    state block. Heterogeneous blocks remain separate. Deposits are reduced to
    one resolved target and one numeric scale per edge, so event delivery never
    performs symbolic kernel work.
    """

    normalized = tuple(instances)
    if not normalized:
        raise ResolutionError("per-edge resolution requires at least one synapse edge")
    if len({item.edge_id for item in normalized}) != len(normalized):
        raise ResolutionError("per-edge synapse identifiers must be unique")
    for item in normalized:
        initial = (
            tuple(float(value) for value in item.initial)
            if isinstance(item.initial, (tuple, list))
            else (float(item.initial),)
        )
        if not initial or any(not math.isfinite(value) for value in initial):
            raise ResolutionError("per-edge initial states must be finite")

    membranes = [state for state in neuron.states if state.role is StateRole.MEMBRANE]
    receptors = [state for state in neuron.states if state.role is StateRole.RECEPTOR]
    receptor_names = {item.receptor for item in normalized}
    if len(membranes) != 1 or len(receptors) != 1 or receptor_names != {
        receptors[0].name
    }:
        raise CapabilityError(
            "the first PER_EDGE capability requires one membrane and one mapped receptor"
        )
    membrane = membranes[0]
    receptor = receptors[0]
    if len(neuron.states) != 2 or set(neuron.dynamics) != {membrane.name}:
        raise CapabilityError(
            "the first PER_EDGE capability accepts no additional intrinsic state"
        )
    if neuron.threshold.readout != membrane.name or set(neuron.reset) != {
        membrane.name
    }:
        raise CapabilityError(
            "the first PER_EDGE capability requires a fixed membrane threshold and reset"
        )

    neuron_values = _parameter_values(neuron.parameters, neuron_bindings, "neuron")
    neuron_symbols = {
        definition.name: sympy.Symbol(definition.name, real=True)
        for definition in neuron.parameters
    }
    neuron_symbols[membrane.name] = sympy.Symbol(membrane.name, real=True)
    neuron_symbols[receptor.name] = sympy.Symbol(receptor.name, real=True)
    neuron_subs = {
        neuron_symbols[name]: value for name, value in neuron_values.items()
    }
    v_symbol = neuron_symbols[membrane.name]
    receptor_symbol = neuron_symbols[receptor.name]
    rhs = _expression(neuron.dynamics[membrane.name], neuron_symbols, "membrane dynamics")
    a_expr = sympy.factor(sympy.diff(rhs, v_symbol))
    coupling_expr = sympy.factor(sympy.diff(rhs, receptor_symbol))
    b_expr = sympy.factor(rhs.subs({v_symbol: 0, receptor_symbol: 0}))
    if {v_symbol, receptor_symbol}.intersection(
        a_expr.free_symbols | coupling_expr.free_symbols
    ) or sympy.simplify(
        rhs - (a_expr * v_symbol + coupling_expr * receptor_symbol + b_expr)
    ) != 0:
        raise CapabilityError(
            "PER_EDGE membrane dynamics must be affine in voltage and receptor current"
        )
    level_expr = _expression(neuron.threshold.level, neuron_symbols, "threshold level")
    reset_expr = _expression(neuron.reset[membrane.name], neuron_symbols, "reset expression")
    if {v_symbol, receptor_symbol}.intersection(
        level_expr.free_symbols | reset_expr.free_symbols
    ):
        raise CapabilityError("PER_EDGE requires fixed threshold and state-independent reset")
    neuron_parameter_symbols = {
        neuron_symbols[definition.name] for definition in neuron.parameters
    }
    neuron_used = set().union(
        rhs.free_symbols, level_expr.free_symbols, reset_expr.free_symbols
    ).intersection(neuron_parameter_symbols)
    neuron_unused = neuron_parameter_symbols.difference(neuron_used)
    if neuron_unused:
        raise ResolutionError(
            "unused neuron parameter(s): "
            + ", ".join(sorted(str(item) for item in neuron_unused))
        )
    a = _number(a_expr, neuron_subs, "membrane decay")
    b = _number(b_expr, neuron_subs, "membrane affine input")
    coupling = _number(coupling_expr, neuron_subs, "receptor coupling")
    threshold = _number(level_expr, neuron_subs, "threshold")
    reset = _number(reset_expr, neuron_subs, "reset")
    refractory = neuron.refractory.duration if neuron.refractory else 0.0
    if not math.isfinite(refractory) or refractory < 0.0:
        raise ResolutionError("refractory duration must be finite and nonnegative")
    if a >= 0.0 or coupling == 0.0:
        raise CapabilityError(
            "PER_EDGE requires stable membrane decay and nonzero receptor coupling"
        )
    if reset >= threshold:
        raise CapabilityError("PER_EDGE membrane reset must remain below threshold")

    validated: list[dict[str, object]] = []
    synapse_shapes: dict[int, object] = {}
    for item in sorted(normalized, key=lambda value: value.edge_id):
        synapse = item.synapse
        if len(synapse.states) not in (1, 2) or set(synapse.dynamics) != set(
            synapse.states
        ):
            raise CapabilityError(
                f"edge {item.edge_id} requires a scalar exponential or two-state alpha synapse"
            )
        if item.output not in synapse.outputs or synapse.spike_target not in synapse.states:
            raise ResolutionError(
                f"edge {item.edge_id} has an invalid synapse output or deposit target"
            )
        values = _parameter_values(synapse.parameters, item.bindings, "synapse")
        symbols = {
            definition.name: sympy.Symbol(definition.name, real=True)
            for definition in synapse.parameters
        }
        for state_name in synapse.states:
            symbols[state_name] = sympy.Symbol(state_name, real=True)
        subs = {symbols[name]: value for name, value in values.items()}
        output_expr = _expression(
            synapse.outputs[item.output], symbols, f"edge {item.edge_id} output"
        )
        update_symbols = dict(symbols)
        weight_symbol = sympy.Symbol("w", real=True)
        update_symbols["w"] = weight_symbol
        update = _expression(
            synapse.spike_update, update_symbols, f"edge {item.edge_id} on_spike"
        )
        if len(synapse.states) == 1:
            kind = "exp"
            state_name = synapse.states[0]
            state_symbol = symbols[state_name]
            dynamics = _expression(
                synapse.dynamics[state_name],
                symbols,
                f"edge {item.edge_id} dynamics",
            )
            q_expr = sympy.factor(sympy.diff(dynamics, state_symbol))
            if state_symbol in q_expr.free_symbols or sympy.simplify(
                dynamics - q_expr * state_symbol
            ) != 0:
                raise CapabilityError(
                    f"edge {item.edge_id} synapse must be zero-centered scalar decay"
                )
            if output_expr != state_symbol:
                raise CapabilityError(
                    f"edge {item.edge_id} output must directly expose its decaying state"
                )
            if sympy.simplify(update - (state_symbol + weight_symbol)) != 0:
                raise CapabilityError(
                    f"edge {item.edge_id} on_spike must have the form "
                    f"{state_name} <- {state_name} + w"
                )
            deposit_scale_expr = sympy.Integer(1)
            state_order = (state_name,)
            initial = (
                tuple(float(value) for value in item.initial)
                if isinstance(item.initial, (tuple, list))
                else (float(item.initial),)
            )
            if len(initial) != 1:
                raise ResolutionError(
                    f"edge {item.edge_id} scalar exponential initial state must contain [s]"
                )
            dynamics_expressions = (dynamics,)
        else:
            kind = "alpha"
            if not output_expr.is_Symbol or str(output_expr) not in synapse.states:
                raise CapabilityError(
                    f"edge {item.edge_id} alpha output must directly expose one state"
                )
            s_name = str(output_expr)
            z_name = synapse.spike_target
            if s_name == z_name or {s_name, z_name} != set(synapse.states):
                raise CapabilityError(
                    f"edge {item.edge_id} alpha deposit must target the non-output state"
                )
            s_symbol = symbols[s_name]
            z_symbol = symbols[z_name]
            ds = _expression(
                synapse.dynamics[s_name], symbols, f"edge {item.edge_id} d{s_name}/dt"
            )
            dz = _expression(
                synapse.dynamics[z_name], symbols, f"edge {item.edge_id} d{z_name}/dt"
            )
            q_expr = sympy.factor(sympy.diff(ds, s_symbol))
            q_z = sympy.factor(sympy.diff(dz, z_symbol))
            if (
                sympy.simplify(q_expr - q_z) != 0
                or sympy.simplify(sympy.diff(ds, z_symbol) - 1) != 0
                or sympy.simplify(sympy.diff(dz, s_symbol)) != 0
                or sympy.simplify(ds - (q_expr * s_symbol + z_symbol)) != 0
                or sympy.simplify(dz - q_expr * z_symbol) != 0
            ):
                raise CapabilityError(
                    f"edge {item.edge_id} equations are outside the supported alpha structure"
                )
            increment = sympy.factor(update - z_symbol)
            deposit_scale_expr = sympy.factor(sympy.diff(increment, weight_symbol))
            if (
                {s_symbol, z_symbol}.intersection(increment.free_symbols)
                or sympy.simplify(increment - deposit_scale_expr * weight_symbol) != 0
                or sympy.simplify(deposit_scale_expr - q_expr**2) != 0
            ):
                raise CapabilityError(
                    f"edge {item.edge_id} alpha deposit must use unit-area q^2 normalization"
                )
            state_order = (s_name, z_name)
            if isinstance(item.initial, (tuple, list)):
                initial = tuple(float(value) for value in item.initial)
                if len(initial) != 2:
                    raise ResolutionError(
                        f"edge {item.edge_id} alpha initial state must contain [s, z]"
                    )
            else:
                scalar_initial = float(item.initial)
                if scalar_initial != 0.0:
                    raise ResolutionError(
                        f"edge {item.edge_id} alpha initial state must contain [s, z]"
                    )
                initial = (0.0, 0.0)
            dynamics_expressions = (ds, dz)
        parameter_symbols = {symbols[value.name] for value in synapse.parameters}
        used = set().union(
            *(expression.free_symbols for expression in dynamics_expressions),
            output_expr.free_symbols,
            update.free_symbols,
        ).intersection(parameter_symbols)
        unused = parameter_symbols.difference(used)
        if unused:
            raise ResolutionError(
                f"edge {item.edge_id} has unused synapse parameter(s): "
                + ", ".join(sorted(str(value) for value in unused))
            )
        q = _number(q_expr, subs, f"edge {item.edge_id} decay")
        if q >= 0.0:
            raise CapabilityError(
                f"edge {item.edge_id} requires a stable negative synaptic decay"
            )
        deposit_scale = _number(
            deposit_scale_expr, subs, f"edge {item.edge_id} deposit scale"
        )
        shape = {
            "name": synapse.name,
            "kind": kind,
            "parameters": [
                (definition.name, definition.domain.value)
                for definition in synapse.parameters
            ],
            "states": synapse.states,
            "dynamics": tuple(sympy.srepr(value) for value in dynamics_expressions),
            "deposit": sympy.srepr(update),
            "output": (item.output, sympy.srepr(output_expr)),
            "mapping": item.receptor,
        }
        validated.append(
            {
                "item": item,
                "kind": kind,
                "q": q,
                "q_expr": q_expr,
                "values": values,
                "symbols": symbols,
                "state_order": state_order,
                "initial": initial,
                "deposit_scale": deposit_scale,
                "shape_hash": _hash(shape),
            }
        )
        synapse_shapes[item.edge_id] = {
            **shape,
        }

    groups_by_rate: dict[tuple[str, str], list[dict[str, object]]] = {}
    for value in validated:
        key = (str(value["shape_hash"]), float(value["q"]).hex())
        groups_by_rate.setdefault(key, []).append(value)
    ordered_groups = tuple(
        groups_by_rate[key]
        for key in sorted(
            groups_by_rate,
            key=lambda item: (float.fromhex(item[1]), item[0]),
        )
    )
    state_count = 1 + sum(
        len(group[0]["state_order"]) for group in ordered_groups
    )
    if state_count > 8:
        raise CapabilityError(
            "PER_EDGE exceeds the bounded eight-state analytical cluster; "
            "reduce distinct synaptic kernel state"
        )

    neuron_runtime_symbols = {
        name: sympy.Symbol(f"neuron.{name}", real=True) for name in neuron_values
    }
    neuron_namespace = {
        neuron_symbols[name]: symbol for name, symbol in neuron_runtime_symbols.items()
    }
    a_runtime = a_expr.xreplace(neuron_namespace)
    b_runtime = b_expr.xreplace(neuron_namespace)
    coupling_runtime = coupling_expr.xreplace(neuron_namespace)
    threshold_runtime = level_expr.xreplace(neuron_namespace)
    reset_runtime = reset_expr.xreplace(neuron_namespace)
    runtime_bindings = {
        str(neuron_runtime_symbols[name]): value for name, value in neuron_values.items()
    }
    parameter_domains = {
        str(neuron_runtime_symbols[definition.name]): definition.domain
        for definition in neuron.parameters
    }
    group_rates: list[sympy.Expr] = []
    numeric_rates: list[float] = []
    group_kinds: list[str] = []
    group_offsets: list[int] = []
    group_edge_ids: list[tuple[int, ...]] = []
    group_initials: list[float] = []
    edge_deposits: dict[int, tuple[int, float]] = {}
    state_names = [membrane.name]
    for group_index, group in enumerate(ordered_groups):
        representative_record = group[0]
        representative = representative_record["item"]
        assert isinstance(representative, PerEdgeSynapseInstance)
        q = float(representative_record["q"])
        q_expr = representative_record["q_expr"]
        values = representative_record["values"]
        symbols = representative_record["symbols"]
        state_order = representative_record["state_order"]
        kind = str(representative_record["kind"])
        assert isinstance(q_expr, sympy.Expr)
        assert isinstance(values, dict)
        assert isinstance(symbols, dict)
        assert isinstance(state_order, tuple)
        runtime_symbols = {
            name: sympy.Symbol(f"edge_group.{group_index}.{name}", real=True)
            for name in values
        }
        q_runtime = q_expr.xreplace(
            {symbols[name]: symbol for name, symbol in runtime_symbols.items()}
        )
        group_rates.append(q_runtime)
        numeric_rates.append(q)
        group_kinds.append(kind)
        group_offsets.append(len(state_names))
        for name, value in values.items():
            runtime_bindings[str(runtime_symbols[name])] = value
        for definition in representative.synapse.parameters:
            parameter_domains[str(runtime_symbols[definition.name])] = definition.domain
        ids = tuple(record["item"].edge_id for record in group)
        group_edge_ids.append(ids)
        for component in range(len(state_order)):
            group_initials.append(
                sum(float(record["initial"][component]) for record in group)
            )
        deposit_component = 0 if kind == "exp" else 1
        for record in group:
            item = record["item"]
            edge_deposits[item.edge_id] = (
                group_offsets[-1] + deposit_component,
                float(record["deposit_scale"]),
            )
        for state_name in state_order:
            state_names.append(
                f"edge_mode[{group_index}].{representative.synapse.name}.{state_name}"
            )

    delta = sympy.Symbol("Delta", real=True, nonnegative=True)
    variables = sympy.symbols(f"x0:{len(state_names)}", real=True)
    ea = sympy.exp(a_runtime * delta)
    next_v = ea * variables[0] + b_runtime * delta * phi1(a_runtime * delta)
    asymptote = -b_runtime / a_runtime
    rate_blocks: dict[str, dict[str, object]] = {
        float(a).hex(): {
            "numeric": a,
            "rate": a_runtime,
            "coefficients": [variables[0] - asymptote],
        }
    }
    next_modes: list[sympy.Expr] = []
    for index, (kind, rate, numeric_rate, offset) in enumerate(
        zip(group_kinds, group_rates, numeric_rates, group_offsets)
    ):
        rate_key = numeric_rate.hex()
        block = rate_blocks.setdefault(
            rate_key,
            {"numeric": numeric_rate, "rate": rate, "coefficients": []},
        )
        coefficients = block["coefficients"]
        assert isinstance(coefficients, list)

        def add_coefficient(target: list[sympy.Expr], degree: int, value: sympy.Expr) -> None:
            """Merge one term into a polynomial coefficient vector."""

            while len(target) <= degree:
                target.append(sympy.Integer(0))
            target[degree] = sympy.factor(target[degree] + value)

        xs = variables[offset]
        gap = rate - a_runtime
        next_v += coupling_runtime * ea * xs * delta * phi1(gap * delta)
        if kind == "exp":
            next_modes.append(sympy.exp(rate * delta) * xs)
            if numeric_rate == a:
                add_coefficient(coefficients, 1, coupling_runtime * xs)
            else:
                add_coefficient(coefficients, 0, coupling_runtime * xs / gap)
                membrane = rate_blocks[float(a).hex()]["coefficients"]
                assert isinstance(membrane, list)
                add_coefficient(membrane, 0, -coupling_runtime * xs / gap)
        else:
            xz = variables[offset + 1]
            next_v += (
                coupling_runtime
                * ea
                * xz
                * delta**2
                * phi1_derivative(gap * delta)
            )
            eq = sympy.exp(rate * delta)
            next_modes.extend((eq * (xs + xz * delta), eq * xz))
            if numeric_rate == a:
                add_coefficient(coefficients, 1, coupling_runtime * xs)
                add_coefficient(coefficients, 2, coupling_runtime * xz / 2)
            else:
                add_coefficient(
                    coefficients,
                    0,
                    coupling_runtime * xs / gap - coupling_runtime * xz / gap**2,
                )
                add_coefficient(coefficients, 1, coupling_runtime * xz / gap)
                membrane = rate_blocks[float(a).hex()]["coefficients"]
                assert isinstance(membrane, list)
                add_coefficient(
                    membrane,
                    0,
                    -coupling_runtime * xs / gap + coupling_runtime * xz / gap**2,
                )

    ordered_rate_blocks = tuple(
        rate_blocks[key]
        for key in sorted(rate_blocks, key=lambda value: float.fromhex(value))
    )
    expressions: dict[str, sympy.Expr] = {
        "next_v": next_v,
        "clamped_v": variables[0],
        "reset_v": reset_runtime.xreplace({v_symbol: variables[0]}),
        "crossing_limit": threshold_runtime - asymptote,
    }
    normal_roots = ["next_v"]
    clamped_roots = ["clamped_v"]
    reset_roots = ["reset_v"]
    for index, next_mode in enumerate(next_modes):
        expressions[f"next_edge_{index}"] = next_mode
        expressions[f"clamped_edge_{index}"] = next_mode
        expressions[f"reset_edge_{index}"] = variables[index + 1]
        normal_roots.append(f"next_edge_{index}")
        clamped_roots.append(f"clamped_edge_{index}")
        reset_roots.append(f"reset_edge_{index}")
    coefficient_roots: list[tuple[str, ...]] = []
    rate_roots: list[str] = []
    for block_index, block in enumerate(ordered_rate_blocks):
        rate_name = f"crossing_rate_{block_index}"
        rate_expression = block["rate"]
        coefficients = block["coefficients"]
        assert isinstance(rate_expression, sympy.Expr)
        assert isinstance(coefficients, list)
        expressions[rate_name] = rate_expression
        rate_roots.append(rate_name)
        names: list[str] = []
        for degree, expression in enumerate(coefficients):
            name = f"crossing_coefficient_{block_index}_{degree}"
            expressions[name] = -expression
            names.append(name)
        coefficient_roots.append(tuple(names))

    single_distinct_alpha = (
        len(ordered_groups) == 1
        and group_kinds[0] == "alpha"
        and numeric_rates[0] != a
    )
    all_distinct_scalars = all(kind == "exp" for kind in group_kinds) and all(
        rate != a for rate in numeric_rates
    )
    if single_distinct_alpha:
        rate = group_rates[0]
        offset = group_offsets[0]
        xs = variables[offset]
        xz = variables[offset + 1]
        gap = rate - a_runtime
        membrane_coefficient = (
            variables[0]
            - asymptote
            - coupling_runtime * xs / gap
            + coupling_runtime * xz / gap**2
        )
        synapse_constant = (
            coupling_runtime * xs / gap - coupling_runtime * xz / gap**2
        )
        synapse_linear = coupling_runtime * xz / gap
        extremum_a = a_runtime * membrane_coefficient
        extremum_d = synapse_linear + rate * synapse_constant
        extremum_e = rate * synapse_linear
        extremum_p = a_runtime - rate
        expressions.update(
            {
                "crossing_g": threshold_runtime - next_v,
                "crossing_g_prime": -(
                    a_runtime * next_v
                    + b_runtime
                    + coupling_runtime * next_modes[0]
                ),
                "extremum_f": extremum_a * sympy.exp(extremum_p * delta)
                + extremum_d
                + extremum_e * delta,
                "extremum_f_prime": extremum_a
                * extremum_p
                * sympy.exp(extremum_p * delta)
                + extremum_e,
                "crossing_asymptote": asymptote,
                "crossing_membrane_coefficient": membrane_coefficient,
                "crossing_synapse_constant": synapse_constant,
                "crossing_synapse_linear": synapse_linear,
                "crossing_membrane_rate": a_runtime,
                "crossing_synapse_rate": rate,
                "crossing_threshold": threshold_runtime,
            }
        )
    runtime_parameter_names = tuple(runtime_bindings)
    propagation_dag = lower_expressions(
        expressions,
        parameters=runtime_parameter_names,
        variables=("Delta", *(f"x{index}" for index in range(len(state_names)))),
    )
    tier = (
        SynapseTier.FOLDED_SHARED
        if len(ordered_groups) == 1
        else SynapseTier.PER_EDGE
    )
    canonical = {
        "neuron": neuron.name,
        "neuron_parameters": [
            (item.name, item.domain.value) for item in neuron.parameters
        ],
        "mapping": receptor.name,
        "membrane_rhs": sympy.srepr(rhs),
        "threshold": sympy.srepr(level_expr),
        "reset": sympy.srepr(reset_expr),
        "synapses": sorted(
            {
                _hash(synapse_shapes[item.edge_id]): synapse_shapes[item.edge_id]
                for item in normalized
            }.values(),
            key=_hash,
        ),
        "refractory": None
        if neuron.refractory is None
        else (neuron.refractory.mode, neuron.refractory.duration),
    }
    model_hash = _hash(canonical)
    resolution_key = _hash(
        {
            "model_hash": model_hash,
            "capability": "stable_lif_bounded_per_edge_linear_kernel_v2",
            "tier": tier.value,
            "mode_count": len(ordered_rate_blocks),
            "mode_shapes": tuple(
                tuple(
                    sorted(
                        {
                            _hash(synapse_shapes[item["item"].edge_id])
                            for item in group
                        }
                    )
                )
                for group in ordered_groups
            ),
            "regime": (
                "all_rates<0",
                "repeated_membrane_rate"
                if any(value == a for value in numeric_rates)
                else "distinct_membrane_rate",
                tuple(group_kinds),
            ),
        }
    )
    forbidden_drive_dependencies = set().union(
        a_expr.free_symbols,
        coupling_expr.free_symbols,
        level_expr.free_symbols,
        reset_expr.free_symbols,
    )
    drive_parameters = tuple(
        str(neuron_runtime_symbols[definition.name])
        for definition in neuron.parameters
        if neuron_symbols[definition.name] in b_expr.free_symbols
        and neuron_symbols[definition.name] not in forbidden_drive_dependencies
    )
    return ResolvedPerEdgeLIF(
        neuron_name=neuron.name,
        model_hash=model_hash,
        resolution_key=resolution_key,
        state_names=tuple(state_names),
        readout_index=0,
        bindings=runtime_bindings,
        a=a,
        b=b,
        coupling=coupling,
        synaptic_decays=tuple(numeric_rates),
        threshold=threshold,
        reset=reset,
        refractory=refractory,
        dispatch=DispatchForm.ROOT_FIND,
        tier=tier,
        propagation_dag=propagation_dag,
        normal_roots=tuple(normal_roots),
        clamped_roots=tuple(clamped_roots),
        reset_roots=tuple(reset_roots),
        root_hint=(
            RootFindHint(
                g_root="crossing_g",
                g_prime_root="crossing_g_prime",
                extremum_root="extremum_f",
                extremum_prime_root="extremum_f_prime",
                asymptote_root="crossing_asymptote",
                membrane_coefficient_root="crossing_membrane_coefficient",
                synapse_constant_root="crossing_synapse_constant",
                synapse_linear_root="crossing_synapse_linear",
                membrane_rate_root="crossing_membrane_rate",
                synapse_rate_root="crossing_synapse_rate",
                threshold_root="crossing_threshold",
                relative_tolerance=1e-10,
                fastest_time_constant=min(-1.0 / a, -1.0 / numeric_rates[0]),
                iteration_cap=192,
            )
            if single_distinct_alpha
            else MultiExpRootHint(
                limit_root="crossing_limit",
                coefficient_roots=tuple(
                    names[0] for names in coefficient_roots
                ),
                rate_roots=tuple(rate_roots),
                relative_tolerance=1e-10,
                fastest_time_constant=min(
                    (-1.0 / a, *(-1.0 / value for value in numeric_rates))
                ),
                iteration_cap=2048,
            )
            if all_distinct_scalars
            else ExpPolyRootHint(
                limit_root="crossing_limit",
                rate_roots=tuple(rate_roots),
                coefficient_roots=tuple(coefficient_roots),
                relative_tolerance=1e-10,
                fastest_time_constant=min(
                    (-1.0 / a, *(-1.0 / value for value in numeric_rates))
                ),
                iteration_cap=2048,
            )
        ),
        drive_parameters=drive_parameters,
        parameter_domains=parameter_domains,
        edge_deposits=edge_deposits,
        group_edge_ids=tuple(group_edge_ids),
        group_initials=tuple(group_initials),
    )
