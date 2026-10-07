from __future__ import annotations

import math

import pytest
import sympy

from lacuna import ExprOp, lower_expressions, root_variable_dependencies
from lacuna.errors import ResolutionError
from lacuna.ffi import CoreEvaluator


def test_shared_lif_expressions_are_lowered_and_evaluated_in_c(core: CoreEvaluator) -> None:
    tau, delta, value, asymptote, threshold = sympy.symbols(
        "tau delta value asymptote threshold"
    )
    decay = sympy.exp(-delta / tau)
    next_value = asymptote + (value - asymptote) * decay
    dag = lower_expressions(
        {"next": next_value, "g": threshold - next_value},
        parameters=("tau", "asymptote", "threshold"),
        variables=("delta", "value"),
    )
    assert sum(node.op is ExprOp.EXP for node in dag.nodes) == 1
    result = core.evaluate_expr(
        dag,
        parameters={"tau": 10.0, "asymptote": -45.0, "threshold": -50.0},
        variables={"delta": 10.0, "value": -65.0},
    )
    expected = -45.0 - 20.0 * math.exp(-1.0)
    assert result["next"] == pytest.approx(expected, abs=1e-12)
    assert result["g"] == pytest.approx(-50.0 - expected, abs=1e-12)


def test_expression_bindings_must_be_complete(core: CoreEvaluator) -> None:
    x = sympy.Symbol("x")
    dag = lower_expressions({"root": x + 1}, variables=("x",))
    with pytest.raises(ValueError, match="variable bindings"):
        core.evaluate_expr(dag, variables={})


def test_unknown_expression_symbol_is_rejected() -> None:
    x, y = sympy.symbols("x y")
    with pytest.raises(ResolutionError, match="unknown expression symbol 'y'"):
        lower_expressions({"root": x + y}, variables=("x",))


def test_root_variable_dependencies_follow_only_the_selected_root() -> None:
    delta, v, s, z = sympy.symbols("Delta x0 x1 x2")
    dag = lower_expressions(
        {
            "v": v + delta * s + delta**2 * z,
            "s": s + delta * z,
            "z": z * sympy.exp(-delta),
        },
        variables=("Delta", "x0", "x1", "x2"),
    )

    assert root_variable_dependencies(dag, "v") == frozenset(
        {"Delta", "x0", "x1", "x2"}
    )
    assert root_variable_dependencies(dag, "s") == frozenset(
        {"Delta", "x1", "x2"}
    )
    assert root_variable_dependencies(dag, "z") == frozenset({"Delta", "x2"})
