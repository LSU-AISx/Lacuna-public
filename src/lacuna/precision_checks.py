"""Conservative representation checks for non-executable model preflight.

These checks do not simulate a neuron or certify target solver accuracy. Only
parameter-only algebraic DAG fragments are inspected between scalar checks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, replace
import math
import sys
from numbers import Real

from .errors import PrecisionResolutionError
from .expr import ExprDAG, ExprOp
from .ir import NumericalConfig, ParameterDomain, ResolvedScalarLIF
from .numerical_policy import numerical_policy
from .precision import PrecisionProfile, normalize_precision


_Record = Callable[[float, str, bool], float]


def _rate_partition(
    original: tuple[float, ...], rounded: tuple[float, ...], context: str
) -> None:
    """Preserve the equality assumptions used to derive crossing expressions."""
    for index, rate in enumerate(rounded):
        if rate >= 0.0:
            raise PrecisionResolutionError(
                f"{context} decay rate {index} must remain strictly negative"
            )
        for previous in range(index):
            if (original[index] == original[previous]) != (
                rounded[index] == rounded[previous]
            ):
                raise PrecisionResolutionError(
                    f"{context} distinct decay rates collapse in target precision "
                    "and require a new crossing derivation"
                )


def _static_dag(
    dag: ExprDAG,
    bindings: Mapping[str, float],
    profile: PrecisionProfile,
    context: str,
    record: _Record,
) -> dict[str, float | None]:
    """Check scalar constants without evaluating state or time trajectories."""
    values: list[float | None] = []
    for index, node in enumerate(dag.nodes):
        path = f"{context}.nodes[{index}]"
        value = None
        if node.op is ExprOp.CONST:
            value = record(node.value, f"{path}.constant", False)
        elif node.op is ExprOp.PARAM:
            value = bindings[dag.parameters[node.binding]]
        elif node.op is ExprOp.VAR:
            # Runtime equation variables, including Time, use model precision.
            pass
        else:
            left = values[node.lhs]
            right = values[node.rhs]
            if node.op is ExprOp.DIV and right == 0.0:
                raise PrecisionResolutionError(
                    f"{path} parameter-only denominator becomes zero in target precision"
                )
            if node.op is ExprOp.LOG and left is not None and left <= 0.0:
                raise PrecisionResolutionError(
                    f"{path} logarithm argument is nonpositive in target precision"
                )
            if node.op is ExprOp.POW and left is not None and right is not None:
                if (left == 0.0 and right < 0.0) or (
                    left < 0.0 and not right.is_integer()
                ):
                    raise PrecisionResolutionError(
                        f"{path} power has an invalid parameter-only target domain"
                    )
            try:
                if node.op is ExprOp.NEG and left is not None:
                    value = -left
                elif left is not None and right is not None:
                    if node.op is ExprOp.ADD:
                        value = left + right
                    elif node.op is ExprOp.SUB:
                        value = left - right
                    elif node.op is ExprOp.MUL:
                        value = left * right
                    elif node.op is ExprOp.DIV:
                        value = left / right
                    elif node.op is ExprOp.POW and right.is_integer():
                        value = math.pow(left, right)
                    elif node.op is ExprOp.MAX:
                        value = max(left, right)
            except (OverflowError, ValueError, ZeroDivisionError) as exc:
                raise PrecisionResolutionError(
                    f"{path} parameter-only algebra is not finite in target precision"
                ) from exc
            if value is not None:
                value = record(value, f"{path}.algebraic_constant", False)
        values.append(value)
    return {name: values[index] for name, index in dag.roots.items()}


def validate_precision_dag(
    dag: ExprDAG,
    bindings: Mapping[str, float],
    profile: PrecisionProfile | str,
    *,
    context: str,
    record: _Record,
) -> None:
    """Audit constants in a learning or model DAG without evaluating state."""
    rounded = {
        name: record(bindings[name], f"{context}.bindings.{name}", False)
        for name in dag.parameters
    }
    _static_dag(dag, rounded, normalize_precision(profile), context, record)


def validate_resolved_precision(
    model: object,
    profile: PrecisionProfile | str,
    *,
    context: str,
    record: _Record,
) -> None:
    """Reject known target-representation failures without producing a model.

    The callback rounds model or time values and retains the caller's audit
    record. Passing these checks is not approval for reduced-precision execution.
    """
    selected = normalize_precision(profile)
    hazard = getattr(model, "hazard", None)

    bindings = {
        name: record(value, f"{context}.bindings.{name}", False)
        for name, value in model.bindings.items()
    }
    domains = getattr(model, "parameter_domains", {})
    for name, domain in domains.items():
        if domain is ParameterDomain.POSITIVE and bindings[name] <= 0.0:
            raise PrecisionResolutionError(
                f"{context}.bindings.{name} must remain positive"
            )

    scalars = {}
    for name in (
        "a", "b", "coupling", "adaptation_decay", "adaptation_jump",
        "synaptic_decay", "threshold", "reset",
    ):
        if hasattr(model, name):
            value = getattr(model, name)
            if hazard is not None and name == "threshold" and selected.real_bits < 64:
                value = 65504.0 if selected.real_bits == 16 else float.fromhex("0x1.fffffep127")
            scalars[name] = record(value, f"{context}.{name}", False)
    if scalars["reset"] >= scalars["threshold"] and hazard is None:
        raise PrecisionResolutionError(
            f"{context} reset must remain strictly below threshold"
        )
    refractory = record(model.refractory, f"{context}.refractory", True)
    if refractory < 0.0:
        raise PrecisionResolutionError(f"{context} refractory duration is negative")

    rates = []
    rounded_rates = []
    for name in ("a", "adaptation_decay", "synaptic_decay"):
        if name in scalars:
            rates.append(getattr(model, name))
            rounded_rates.append(scalars[name])
    for index, value in enumerate(getattr(model, "synaptic_decays", ())):
        rates.append(value)
        rounded_rates.append(record(value, f"{context}.synaptic_decays[{index}]", False))
    _rate_partition(tuple(rates), tuple(rounded_rates), context)

    if isinstance(model, ResolvedScalarLIF):
        asymptote = record(-scalars["b"] / scalars["a"], f"{context}.asymptote", False)
        if hazard is None and (asymptote <= scalars["threshold"]) != (
            model.asymptote <= model.threshold
        ):
            raise PrecisionResolutionError(
                f"{context} target rounding changes autonomous crossing classification"
            )
    for index, value in enumerate(getattr(model, "group_initials", ())):
        record(value, f"{context}.group_initials[{index}]", False)
    for edge, (_, value) in getattr(model, "edge_deposits", {}).items():
        record(value, f"{context}.edge_deposits[{edge}].scale", False)

    for name in ("root_hint", "numerical", "hazard"):
        config = getattr(model, name, None)
        if config is None:
            continue
        if selected is PrecisionProfile.FLOAT16:
            if name == "numerical" and config == NumericalConfig():
                config = numerical_policy(selected).step_defaults
            elif name == "root_hint" and hasattr(config, "relative_tolerance"):
                config = replace(config, relative_tolerance=numerical_policy(selected).crossing_relative_tolerance)
            elif name == "hazard":
                config = replace(config, relative_tolerance=8 * float.fromhex("0x1p-10"),
                                 absolute_tolerance=float.fromhex("0x1p-24"),
                                 time_tolerance=float.fromhex("0x1p-10"))
        for field in fields(config):
            value = getattr(config, field.name)
            if isinstance(value, Real) and not isinstance(value, bool) and field.name not in {
                "iteration_cap", "maximum_steps", "maximum_rhs_evaluations",
                "maximum_quadrature_depth", "maximum_root_iterations",
            }:
                is_time = field.name in {
                    "fastest_time_constant", "initial_step", "minimum_step",
                    "maximum_step", "event_tolerance", "time_tolerance",
                }
                rounded = record(value, f"{context}.{name}.{field.name}", is_time)
                if (
                    "tolerance" in field.name
                    or "time_constant" in field.name
                    or field.name.endswith("_step")
                ) and rounded <= 0.0:
                    raise PrecisionResolutionError(
                        f"{context}.{name}.{field.name} must remain positive"
                    )

    hint = getattr(model, "root_hint", None)
    rate_names = tuple(getattr(hint, "rate_roots", ())) or tuple(
        getattr(hint, name)
        for name in (
            "decay_root", "rate_one_root", "rate_two_root",
            "membrane_rate_root", "synapse_rate_root",
        )
        if hasattr(hint, name)
    )
    for name in ("binding_dag", "propagation_dag", "deposit_dag"):
        dag = getattr(model, name, None)
        if dag is not None:
            if hazard is not None and selected.real_bits < 64:
                dag = replace(dag, nodes=tuple(
                    replace(node, value=65504.0 if selected.real_bits == 16 else float.fromhex("0x1.fffffep127"))
                    if node.op is ExprOp.CONST and node.value == sys.float_info.max
                    else node for node in dag.nodes
                ))
            roots = _static_dag(dag, bindings, selected, f"{context}.{name}", record)
            static_rates = tuple(
                roots[root] for root in rate_names if roots.get(root) is not None
            )
            if static_rates and any(value >= 0.0 for value in static_rates):
                raise PrecisionResolutionError(
                    f"{context}.{name} parameter-only rate must remain strictly negative"
                )
            if len(set(static_rates)) != len(static_rates):
                raise PrecisionResolutionError(
                    f"{context}.{name} distinct crossing rates collapse during "
                    "target parameter-only algebra"
                )
