"""Resolve authored graphs using the selected runtime's scalar arithmetic.

Symbolic analysis selects an analytical family. Parameter-only expressions are
then evaluated by that profile's C evaluator before descriptors are constructed.
Runtime expressions remain expressions, including mutable drive parameters.
"""

from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json
import sys

import sympy

from .errors import CoreError, PrecisionResolutionError, ResolutionError
from .expr import ExprDAG, ExprNode, ExprOp, lower_expressions
from .graph import Graph, _resolve_graph
from .ir import (
    DispatchForm, NumericalConfig, ResolvedPerEdgeLIF, ResolvedReactiveIF,
    ResolvedScalarLIF, ResolvedSteppedNeuron, StateRole,
)
from .precision import PrecisionProfile
from .precision_authoring import prepare_precision_graph
from .learning import resolve_learning
from .numerical_policy import numerical_policy
from .resolver import _expression


_FLOAT32_MAX = float.fromhex("0x1.fffffep127")
_EPS32 = float.fromhex("0x1p-23")
_EPS16 = float.fromhex("0x1p-10")
_MIN16 = float.fromhex("0x1p-24")
_UNARY = frozenset((
    ExprOp.NEG, ExprOp.EXP, ExprOp.LOG, ExprOp.PHI1, ExprOp.PHI1_DERIV,
    ExprOp.SIN, ExprOp.COS, ExprOp.TANH,
))


def _dag(dag, core, *, hazard=False):
    """Round scalar literals and preserve every runtime expression node."""
    nodes = []
    for node in dag.nodes:
        value = node.value
        if hazard and node.op is ExprOp.CONST and value == sys.float_info.max:
            value = 65504.0 if core.precision is PrecisionProfile.FLOAT16 else _FLOAT32_MAX
        if node.op is ExprOp.CONST:
            value = core.precision.round_real(value, name="expression constant")
        nodes.append(replace(node, value=value))
    return replace(dag, nodes=tuple(nodes))


def _validate_half_powers(dag, bindings, core, *, mutable_parameters=()):
    """Approve only the half backend's fixed, parameter-only exponent set."""
    if core.precision is not PrecisionProfile.FLOAT16:
        return
    allowed = {-2.0, -1.0, 0.0, 0.5, *(float(value) for value in range(1, 9)),
               core.precision.round_real(-0.2), core.precision.round_real(0.2)}
    constant = []
    for index, node in enumerate(dag.nodes):
        if node.op is ExprOp.CONST:
            known = True
        elif node.op is ExprOp.PARAM:
            known = dag.parameters[node.binding] not in mutable_parameters
        elif node.op is ExprOp.VAR:
            known = False
        elif node.op in _UNARY:
            known = constant[node.lhs]
        else:
            known = constant[node.lhs] and constant[node.rhs]
        if node.op is ExprOp.POW:
            if not constant[node.rhs]:
                raise PrecisionResolutionError(
                    f"float16 POW node {index} requires a parameter-only fixed exponent"
                )
            probe = replace(dag, roots={"exponent": node.rhs})
            exponent = core.evaluate_expr(
                probe, parameters={name: bindings[name] for name in dag.parameters},
                variables={name: 0.0 for name in dag.variables}, roots=("exponent",),
            )["exponent"]
            if exponent not in allowed:
                raise PrecisionResolutionError(
                    f"float16 POW node {index} has unsupported fixed exponent {exponent!r}"
                )
        constant.append(known)


def _static_values(dag, bindings, core):
    """Evaluate only parameter-dependent nodes without inventing a state."""
    _validate_half_powers(dag, bindings, core)
    constant = []
    for node in dag.nodes:
        if node.op in (ExprOp.CONST, ExprOp.PARAM):
            known = True
        elif node.op is ExprOp.VAR:
            known = False
        elif node.op in _UNARY:
            known = constant[node.lhs]
        else:
            known = constant[node.lhs] and constant[node.rhs]
        constant.append(known)
    roots = {str(i): i for i, known in enumerate(constant) if known}
    if not roots:
        return {}
    probe = replace(dag, roots=roots)
    values = core.evaluate_expr(
        probe, parameters={name: bindings[name] for name in dag.parameters},
        variables={name: 0.0 for name in dag.variables}, roots=tuple(roots)
    )
    return {name: values[str(index)] for name, index in dag.roots.items() if constant[index]}


def _evaluate(expressions, bindings, core):
    dag = _dag(lower_expressions(expressions, parameters=tuple(bindings)), core)
    _validate_half_powers(dag, bindings, core)
    return core.evaluate_expr(
        dag, parameters={name: bindings[name] for name in dag.parameters}, variables={},
    )


def _native_sum(values, core):
    """Aggregate edge states with the same scalar additions used at runtime."""
    nodes = [ExprNode(ExprOp.CONST, value=0.0)]
    last = 0
    for value in values:
        nodes.append(ExprNode(ExprOp.CONST, value=value))
        nodes.append(ExprNode(ExprOp.ADD, lhs=last, rhs=len(nodes) - 1))
        last = len(nodes) - 1
    dag = ExprDAG(tuple(nodes), (), (), {"sum": last})
    return core.evaluate_expr(dag, parameters={}, variables={})["sum"]


def _native_ratio(numerator, denominator, core):
    dag = ExprDAG((
        ExprNode(ExprOp.PARAM, binding=0),
        ExprNode(ExprOp.PARAM, binding=1),
        ExprNode(ExprOp.DIV, lhs=0, rhs=1),
    ), ("numerator", "denominator"), (), {"ratio": 2})
    return core.evaluate_expr(dag, parameters={
        "numerator": numerator, "denominator": denominator,
    }, variables={})["ratio"]


def _rate_roots(hint):
    if hint is None:
        return ()
    if hasattr(hint, "rate_roots"):
        return hint.rate_roots
    if hasattr(hint, "rate_one_root"):
        return (hint.rate_one_root, hint.rate_two_root)
    if hasattr(hint, "membrane_rate_root"):
        return (hint.membrane_rate_root, hint.synapse_rate_root)
    if hasattr(hint, "decay_root"):
        return (hint.decay_root,)
    return ()


def _hazard(authored, bindings, core, hazard):
    symbols = {name: sympy.Symbol(name, real=True) for name in bindings}
    symbols.update({state.name: sympy.Symbol(state.name, real=True) for state in authored.states})
    rate = _expression(authored.hazard.rate, symbols, "hazard")
    exponential, = rate.atoms(sympy.exp)
    voltage = symbols[next(s.name for s in authored.states if s.role is StateRole.MEMBRANE)]
    exponent = sympy.expand(exponential.args[0])
    gain = sympy.simplify(sympy.diff(exponent, voltage))
    prefactor = sympy.simplify(rate / exponential)
    offset = sympy.simplify(exponent - gain * voltage)
    values = _evaluate({
        "gain": gain, "prefactor": prefactor,
        "log_scale": sympy.log(prefactor) + offset,
    }, bindings, core)
    if values["gain"] <= 0.0 or values["prefactor"] <= 0.0:
        raise PrecisionResolutionError("hazard gain and prefactor must remain positive")
    settings = {"relative_tolerance": 64 * _EPS32, "time_tolerance": 8 * _EPS32}
    if core.precision is PrecisionProfile.FLOAT16:
        settings = {"relative_tolerance": 8 * _EPS16, "absolute_tolerance": _MIN16,
                    "time_tolerance": _EPS16}
    return replace(hazard, voltage_gain=values["gain"], log_scale=values["log_scale"], **settings)


def _model(model, authored, bindings, core):
    changes = {}
    is_hazard = getattr(model, "hazard", None) is not None
    for item in fields(model):
        value = getattr(model, item.name)
        if isinstance(value, ExprDAG):
            value = _dag(value, core, hazard=is_hazard)
            changes[item.name] = value
    dag = changes["propagation_dag"]
    hint = getattr(model, "root_hint", None)
    rate_roots = _rate_roots(hint)
    bound_rates = core.evaluate_expr(
        dag, parameters={name: model.bindings[name] for name in dag.parameters},
        variables={name: 0.0 for name in dag.variables}, roots=rate_roots,
    ) if rate_roots else {}
    rates = tuple(bound_rates[root] for root in rate_roots)
    if any(rate >= 0.0 for rate in rates):
        raise PrecisionResolutionError("analytical decay must remain strictly negative")
    if len(set(rates)) != len(rates):
        raise PrecisionResolutionError(
            "distinct analytical decay rates collapse in target arithmetic"
        )
    for value in changes.values():
        _validate_half_powers(value, model.bindings, core, mutable_parameters=model.drive_parameters)
        _static_values(value, model.bindings, core)
    constants = _static_values(dag, model.bindings, core)
    symbols = {name: sympy.Symbol(name, real=True) for name in bindings}
    if not is_hazard:
        values = _evaluate({
            "threshold": _expression(authored.threshold.level, symbols, "threshold"),
        }, bindings, core)
        changes["threshold"] = values["threshold"]
    else:
        changes["threshold"] = 65504.0 if core.precision is PrecisionProfile.FLOAT16 else _FLOAT32_MAX
        changes["hazard"] = _hazard(authored, bindings, core, model.hazard)
    if isinstance(model, ResolvedScalarLIF):
        bound = _static_values(changes["binding_dag"], model.bindings, core)
        changes.update({name: bound[name] for name in ("a", "b", "reset")})
        if changes["a"] >= 0.0:
            raise PrecisionResolutionError("analytical decay must remain strictly negative")
        if not is_hazard:
            asymptote = _native_ratio(-changes["b"], changes["a"], core)
            changes["dispatch"] = (
                DispatchForm.CLOSED_FORM if asymptote > changes["threshold"]
                else DispatchForm.REACTIVE
            )
    else:
        readout_reset = model.reset_roots[model.readout_index]
        if readout_reset in constants:
            changes["reset"] = constants[readout_reset]
    if hint is not None and hasattr(hint, "relative_tolerance"):
        # The reciprocal is a clock interval, evaluated with native scalar inputs.
        fastest = _evaluate(
            {"tau": -1 / sympy.Symbol("rate", real=True)},
            {"rate": min(rates)}, core,
        )["tau"]
        changes["root_hint"] = replace(
            hint, relative_tolerance=numerical_policy(core.precision).crossing_relative_tolerance,
            fastest_time_constant=fastest,
        )
    if hasattr(model, "numerical"):
        config = model.numerical
        default = NumericalConfig()
        if config == default:
            config = numerical_policy(core.precision).step_defaults
        changes["numerical"] = config
    if "reset" in changes and changes["reset"] >= changes["threshold"]:
        raise PrecisionResolutionError("reset and threshold are not separated in target arithmetic")
    changes["resolution_key"] = hashlib.sha256(json.dumps(
        [model.resolution_key, numerical_policy(core.precision).cache_key, "native-bind-v1"],
        separators=(",", ":"),
    ).encode()).hexdigest()
    return replace(model, **changes)


def _edge_states(resolved, core):
    edges = list(resolved.edges)
    models = list(resolved.models)
    initials = list(resolved.initial_values)
    by_id = {edge.id: edge for edge in resolved.graph.edges}
    positions = {edge.id: index for index, edge in enumerate(
        sorted(resolved.graph.edges, key=lambda edge: edge.id)
    )}
    for index, model in enumerate(models):
        if not isinstance(model, ResolvedPerEdgeLIF):
            continue
        grouped = []
        deposits = dict(model.edge_deposits)
        for ids in model.group_edge_ids:
            representative = by_id[ids[0]]
            synapse = resolved._parsed_synapses[representative.synapse]
            width = len(synapse.states)
            for component in range(width):
                values = []
                for edge_id in ids:
                    initial = by_id[edge_id].initial
                    value = initial[component] if isinstance(initial, (tuple, list)) else initial
                    values.append(core.precision.round_real(value))
                grouped.append(_native_sum(values, core))
            for edge_id in ids:
                edge = by_id[edge_id]
                source = resolved._parsed_synapses[edge.synapse]
                position = positions[edge_id]
                bindings = resolved._edge_synapse_bindings[position]
                symbols = {name: sympy.Symbol(name, real=True) for name in bindings}
                symbols.update({name: sympy.Symbol(name, real=True) for name in source.states})
                symbols["w"] = sympy.Symbol("w", real=True)
                update = _expression(source.spike_update, symbols, "synaptic deposit")
                scale = sympy.simplify(sympy.diff(update, symbols["w"]))
                value = _evaluate({"scale": scale}, bindings, core)["scale"]
                if deposits[edge_id][1] != 0.0 and value == 0.0:
                    raise PrecisionResolutionError("synaptic deposit scale underflows")
                deposits[edge_id] = (deposits[edge_id][0], value)
                edges[position] = replace(edges[position], deposit_scale=value)
        models[index] = replace(model, group_initials=tuple(grouped), edge_deposits=deposits)
        initials[index] = (initials[index][0], *grouped)
    return replace(resolved, models=tuple(models), initial_values=tuple(initials), edges=tuple(edges))


def resolve_target_graph(graph: Graph, core):
    """Compile from authored values, never from a cast float64 execution plan."""
    if core.precision is PrecisionProfile.FLOAT64:
        return graph.resolve()

    def record(value, path, is_time):
        convert = core.precision.round_time if is_time else core.precision.round_real
        return convert(value, name=path)

    try:
        original = graph.resolve()
        prepared, parsed, synapses = prepare_precision_graph(graph, core.precision, record=record)
        if core.precision is PrecisionProfile.FLOAT16:
            for edge in prepared.edges:
                if edge.plasticity is None:
                    continue
                learning = resolve_learning(edge.plasticity)
                bindings = dict(zip(learning.program.parameter_names, learning.parameter_values))
                dags = [event.expressions for event in learning.program.events]
                if learning.program.observer is not None:
                    dags.append(learning.program.observer.expressions)
                for dag in dags:
                    _validate_half_powers(_dag(dag, core), bindings, core)
        resolved = _resolve_graph(prepared, parsed_models=parsed, parsed_synapse_models=synapses)
        for before, after in zip(original.models, resolved.models):
            if isinstance(after, ResolvedSteppedNeuron) and not isinstance(before, ResolvedSteppedNeuron):
                raise PrecisionResolutionError(
                    "target rounding changes analytical dynamics to stepped execution"
                )
        nodes_by_id = {node.id: node for node in prepared.nodes}
        models = tuple(
            _model(model, parsed[nodes_by_id[node_id].model], bindings, core)
            for model, node_id, bindings in zip(
                resolved.models, resolved.node_ids, resolved._node_bindings,
            )
        )
        for node_id, model, initial in zip(resolved.node_ids, models, resolved.initial_values):
            values = initial if isinstance(initial, tuple) else (initial,)
            if not isinstance(model, ResolvedReactiveIF) and getattr(model, "hazard", None) is None:
                if core.precision.round_real(values[model.readout_index]) >= model.threshold:
                    raise PrecisionResolutionError(
                        f"node {node_id} initial state must remain below its target threshold"
                    )
        identity = hashlib.sha256(json.dumps([
            prepared.to_text(), numerical_policy(core.precision).cache_key,
            [model.resolution_key for model in models], "native-bind-v1",
        ], separators=(",", ":")).encode()).hexdigest()
        resolved = replace(
            resolved, models=models, precision=core.precision, target_binding_key=identity,
        )
        return _edge_states(resolved, core)
    except PrecisionResolutionError:
        raise
    except (CoreError, ResolutionError, ValueError, OverflowError) as exc:
        raise PrecisionResolutionError(f"{core.precision.value}: {exc}") from exc
