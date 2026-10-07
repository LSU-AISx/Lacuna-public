"""Lower backend-neutral expression DAGs for evaluation by the C core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Sequence

import sympy

from .errors import CapabilityError, ResolutionError


class ExprOp(IntEnum):
    """Operations supported by the C expression evaluator."""

    CONST = 0
    PARAM = 1
    VAR = 2
    NEG = 3
    ADD = 4
    SUB = 5
    MUL = 6
    DIV = 7
    POW = 8
    EXP = 9
    LOG = 10
    PHI1 = 11
    PHI1_DERIV = 12
    SIN = 13
    COS = 14
    TANH = 15
    MAX = 16


class _Phi1(sympy.Function):
    nargs = 1


class _Phi1Derivative(sympy.Function):
    nargs = 1


def phi1(value: sympy.Expr) -> sympy.Expr:
    """Stable symbolic marker for expm1(x) / x, including its x=0 limit."""

    return _Phi1(value)


def phi1_derivative(value: sympy.Expr) -> sympy.Expr:
    """Stable symbolic marker for d(phi1(x))/dx, including its x=0 limit."""

    return _Phi1Derivative(value)


@dataclass(frozen=True)
class ExprNode:
    """One node in a topologically ordered expression DAG."""

    op: ExprOp
    lhs: int = 0
    rhs: int = 0
    binding: int = 0
    value: float = 0.0


@dataclass(frozen=True)
class ExprDAG:
    """Shared expression nodes and named output roots."""

    nodes: tuple[ExprNode, ...]
    parameters: tuple[str, ...]
    variables: tuple[str, ...]
    roots: Mapping[str, int]


def root_parameter_dependencies(dag: ExprDAG, root: str) -> frozenset[str]:
    """Return parameter names reachable from one named DAG root."""

    if root not in dag.roots:
        raise ResolutionError(f"unknown expression root '{root}'")
    memo: dict[int, frozenset[str]] = {}

    def visit(index: int) -> frozenset[str]:
        """Collect parameter leaves reachable from one expression node."""

        if index in memo:
            return memo[index]
        node = dag.nodes[index]
        if node.op is ExprOp.PARAM:
            result = frozenset((dag.parameters[node.binding],))
        elif node.op in (
            ExprOp.NEG,
            ExprOp.EXP,
            ExprOp.LOG,
            ExprOp.PHI1,
            ExprOp.PHI1_DERIV,
            ExprOp.SIN,
            ExprOp.COS,
            ExprOp.TANH,
        ):
            result = visit(node.lhs)
        elif node.op in (
            ExprOp.ADD,
            ExprOp.SUB,
            ExprOp.MUL,
            ExprOp.DIV,
            ExprOp.POW,
            ExprOp.MAX,
        ):
            result = visit(node.lhs).union(visit(node.rhs))
        else:
            result = frozenset()
        memo[index] = result
        return result

    return visit(dag.roots[root])


def root_variable_dependencies(dag: ExprDAG, root: str) -> frozenset[str]:
    """Return variable names reachable from one named DAG root."""

    if root not in dag.roots:
        raise ResolutionError(f"unknown expression root '{root}'")
    memo: dict[int, frozenset[str]] = {}

    def visit(index: int) -> frozenset[str]:
        """Collect variable leaves reachable from one expression node."""

        if index in memo:
            return memo[index]
        node = dag.nodes[index]
        if node.op is ExprOp.VAR:
            result = frozenset((dag.variables[node.binding],))
        elif node.op in (
            ExprOp.NEG,
            ExprOp.EXP,
            ExprOp.LOG,
            ExprOp.PHI1,
            ExprOp.PHI1_DERIV,
            ExprOp.SIN,
            ExprOp.COS,
            ExprOp.TANH,
        ):
            result = visit(node.lhs)
        elif node.op in (
            ExprOp.ADD,
            ExprOp.SUB,
            ExprOp.MUL,
            ExprOp.DIV,
            ExprOp.POW,
            ExprOp.MAX,
        ):
            result = visit(node.lhs).union(visit(node.rhs))
        else:
            result = frozenset()
        memo[index] = result
        return result

    return visit(dag.roots[root])


def lower_expressions(
    expressions: Mapping[str, sympy.Expr],
    *,
    parameters: Sequence[str] = (),
    variables: Sequence[str] = (),
) -> ExprDAG:
    """Lower named SymPy expressions together so common work is shared."""

    if not expressions:
        raise ResolutionError("at least one named expression is required")
    parameter_names = tuple(parameters)
    variable_names = tuple(variables)
    if len(parameter_names) != len(set(parameter_names)):
        raise ResolutionError("parameter names must be unique")
    if len(variable_names) != len(set(variable_names)):
        raise ResolutionError("variable names must be unique")
    overlap = set(parameter_names).intersection(variable_names)
    if overlap:
        raise ResolutionError("names cannot be both parameters and variables")

    root_names = tuple(sorted(expressions))
    replacements, reduced = sympy.cse(
        [sympy.sympify(expressions[name]) for name in root_names], order="canonical"
    )
    replacement_map = dict(replacements)
    parameter_index = {name: index for index, name in enumerate(parameter_names)}
    variable_index = {name: index for index, name in enumerate(variable_names)}
    nodes: list[ExprNode] = []
    memo: dict[sympy.Expr, int] = {}

    def append(node: ExprNode) -> int:
        """Append a node and return its stable index."""

        index = len(nodes)
        nodes.append(node)
        return index

    def emit(expr: sympy.Expr) -> int:
        """Lower an expression and reuse any previously emitted node."""

        expression = sympy.sympify(expr)
        if expression in memo:
            return memo[expression]
        if expression in replacement_map:
            index = emit(replacement_map[expression])
            memo[expression] = index
            return index
        if expression.is_Number:
            value = float(expression)
            if not math.isfinite(value):
                raise ResolutionError(f"non-finite expression constant: {expression}")
            index = append(ExprNode(ExprOp.CONST, value=value))
        elif expression.is_Symbol:
            name = str(expression)
            if name in parameter_index:
                index = append(ExprNode(ExprOp.PARAM, binding=parameter_index[name]))
            elif name in variable_index:
                index = append(ExprNode(ExprOp.VAR, binding=variable_index[name]))
            else:
                raise ResolutionError(f"unknown expression symbol '{name}'")
        elif expression.func is sympy.exp:
            index = append(ExprNode(ExprOp.EXP, lhs=emit(expression.args[0])))
        elif expression.func is sympy.log:
            index = append(ExprNode(ExprOp.LOG, lhs=emit(expression.args[0])))
        elif expression.func is sympy.sin:
            index = append(ExprNode(ExprOp.SIN, lhs=emit(expression.args[0])))
        elif expression.func is sympy.cos:
            index = append(ExprNode(ExprOp.COS, lhs=emit(expression.args[0])))
        elif expression.func is sympy.tanh:
            index = append(ExprNode(ExprOp.TANH, lhs=emit(expression.args[0])))
        elif expression.func is sympy.Max:
            arguments = [emit(argument) for argument in expression.args]
            index = arguments[0]
            for argument in arguments[1:]:
                index = append(ExprNode(ExprOp.MAX, lhs=index, rhs=argument))
        elif expression.func is _Phi1:
            index = append(ExprNode(ExprOp.PHI1, lhs=emit(expression.args[0])))
        elif expression.func is _Phi1Derivative:
            index = append(ExprNode(ExprOp.PHI1_DERIV, lhs=emit(expression.args[0])))
        elif expression.is_Add:
            arguments = [emit(argument) for argument in expression.args]
            index = arguments[0]
            for argument in arguments[1:]:
                index = append(ExprNode(ExprOp.ADD, lhs=index, rhs=argument))
        elif expression.is_Mul:
            arguments = [emit(argument) for argument in expression.args]
            index = arguments[0]
            for argument in arguments[1:]:
                index = append(ExprNode(ExprOp.MUL, lhs=index, rhs=argument))
        elif expression.is_Pow:
            index = append(
                ExprNode(ExprOp.POW, lhs=emit(expression.args[0]), rhs=emit(expression.args[1]))
            )
        else:
            raise CapabilityError(
                f"expression operation '{expression.func.__name__}' is not lowered yet"
            )
        memo[expression] = index
        return index

    roots = {name: emit(expression) for name, expression in zip(root_names, reduced)}
    return ExprDAG(tuple(nodes), parameter_names, variable_names, roots)
