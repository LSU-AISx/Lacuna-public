"""Host-side target-width constant binding, separate from graph execution."""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Iterable, Mapping

from .errors import PrecisionResolutionError, ResolutionError
from .expr import ExprDAG, ExprNode, ExprOp
from .numerical_policy import NumericalPolicy, numerical_policy
from .precision import PrecisionProfile, normalize_precision


_BINDING_REVISION = 1
_COMPANION_ABI = 1
_UINT32_MAX = (1 << 32) - 1
_SUPPORTED = ("CONST", "PARAM", "VAR", "NEG", "ADD", "SUB", "MUL", "DIV", "MAX")
_FAILURES = {
    1: "invalid expression descriptor",
    2: "unsupported floating-point environment",
    3: "nonfinite value or intermediate overflow",
    4: "nonzero value or intermediate underflows to zero",
    5: "division by zero",
}


class TargetBindingKind(IntEnum):
    """Whether a node can be bound without evaluating runtime state."""

    DYNAMIC = 0
    BOUND = 1
    UNSUPPORTED = 2


@dataclass(frozen=True)
class TargetBoundValue:
    """A target-width constant, or an expression that remains unresolved."""

    kind: TargetBindingKind
    value: float | None


@dataclass(frozen=True)
class TargetBoundProgram:
    """Inspection artifact that does not authorize target execution.

    Binding preserves the lowered DAG's operation order, not necessarily the
    order of the authored equation before symbolic lowering. Unsupported math
    remains in the source DAG. No simulation descriptor is produced here.
    """

    precision: PrecisionProfile
    policy: NumericalPolicy
    source_dag: ExprDAG
    immutable_parameters: tuple[tuple[str, float], ...]
    mutable_parameters: tuple[str, ...]
    values: tuple[TargetBoundValue, ...]
    binding_key: str
    binding_revision: int = _BINDING_REVISION
    companion_abi: int = _COMPANION_ABI
    supported_operations: tuple[str, ...] = _SUPPORTED
    executable: bool = field(default=False, init=False)

    def root_value(self, name: str) -> float | None:
        """Return a bound root, or None when it needs later evaluation."""

        return self.values[self.source_dag.roots[name]].value

    def root_kind(self, name: str) -> TargetBindingKind:
        """Distinguish dynamic roots from unsupported constant expressions."""

        return self.values[self.source_dag.roots[name]].kind

    def to_document(self) -> dict[str, object]:
        """Export exact retained constants and unresolved source expressions."""

        return {
            "schema": "lacuna-target-bound-program-v1",
            "precision": self.precision.to_record(),
            "numerical_policy": self.policy.to_record(),
            "binding_key": self.binding_key,
            "binding_revision": self.binding_revision,
            "companion_abi": self.companion_abi,
            "supported_operations": list(self.supported_operations),
            "executable": False,
            "source_dag": _dag_document(self.source_dag),
            "immutable_parameters": [
                [name, value.hex()] for name, value in self.immutable_parameters
            ],
            "mutable_parameters": list(self.mutable_parameters),
            "values": [
                {
                    "kind": value.kind.name.lower(),
                    "value": None if value.value is None else value.value.hex(),
                }
                for value in self.values
            ],
        }


class _TargetExprNode(ct.Structure):
    # This is the host companion's fixed wire ABI, not the runtime node layout.
    _fields_ = [
        ("op", ct.c_uint32), ("lhs", ct.c_uint32), ("rhs", ct.c_uint32),
        ("binding", ct.c_uint32), ("value", ct.c_double),
    ]


class TargetBindingEvaluator:
    """Bind primitive algebra through the optional host C companion.

    This library is not linked into the simulation core or needed on an MCU.
    Its double-valued wire buffers carry exact widened float32 results when
    float32 is selected. Arithmetic itself takes place at the selected width.
    """

    def __init__(
        self,
        library: str | Path | None = None,
        *,
        precision: PrecisionProfile | str = PrecisionProfile.FLOAT64,
        time_precision: str | None = None,
    ):
        self._precision = normalize_precision(precision, time_precision)
        if self._precision is PrecisionProfile.FLOAT16:
            raise ResolutionError(
                "the optional primitive-binding companion does not support float16; "
                "use the matching core's target-aware graph compiler"
            )
        if library is None:
            configured = os.environ.get("LACUNA_TARGET_BIND_LIB")
            suffix = ".dylib" if sys.platform == "darwin" else ".so"
            filename = "lacuna_target_bind.dll" if os.name == "nt" else (
                "liblacuna_target_bind" + suffix
            )
            library = configured or Path(__file__).resolve().parents[2] / "build" / filename
        try:
            self._lib = ct.CDLL(str(library))
        except OSError as exc:
            raise ResolutionError(
                "target binding requires the separate host library built with "
                "LACUNA_BUILD_TARGET_BINDER=ON"
            ) from exc
        self._bind_abi()

    @property
    def precision(self) -> PrecisionProfile:
        """Return the requested model and clock representation."""

        return self._precision

    def _bind_abi(self) -> None:
        try:
            version = self._lib.lc_target_abi_version
            size = self._lib.lc_target_sizeof_expr_node
            offset = self._lib.lc_target_expr_node_offset
        except AttributeError as exc:
            raise ResolutionError("library is not a target-binding companion") from exc
        for query in (version, size):
            query.argtypes = []
            query.restype = ct.c_uint32
        offset.argtypes = [ct.c_uint32]
        offset.restype = ct.c_uint32
        if version() != _COMPANION_ABI or size() != ct.sizeof(_TargetExprNode):
            raise ResolutionError("target-binding companion ABI mismatch")
        for index, (name, _) in enumerate(_TargetExprNode._fields_):
            if offset(index) != getattr(_TargetExprNode, name).offset:
                raise ResolutionError("target-binding companion node layout mismatch")
        try:
            bind = self._lib.lc_target_bind
        except AttributeError as exc:
            raise ResolutionError("target-binding companion is incomplete") from exc
        bind.argtypes = [
            ct.POINTER(_TargetExprNode), ct.c_uint32,
            ct.POINTER(ct.c_double), ct.c_uint32, ct.POINTER(ct.c_uint8),
            ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_uint8), ct.c_uint32, ct.POINTER(ct.c_uint32),
        ]
        bind.restype = ct.c_uint32

    def bind(
        self,
        dag: ExprDAG,
        parameters: Mapping[str, float],
        *,
        mutable_parameters: Iterable[str],
    ) -> TargetBoundProgram:
        """Bind only immutable closures, retaining mutable inputs by name.

        Callers must declare every parameter that can change during a run.
        This contract is metadata only until native mutation enforcement and
        target-aware solver lowering are implemented.
        """

        source = _snapshot_dag(dag)
        if isinstance(mutable_parameters, (str, bytes)):
            raise ResolutionError("mutable parameters must be a collection of names")
        try:
            mutable = tuple(mutable_parameters)
        except TypeError as exc:
            raise ResolutionError("mutable parameters must be a collection of names") from exc
        if any(not isinstance(name, str) for name in mutable):
            raise ResolutionError("mutable parameters must be DAG parameter names")
        if len(set(mutable)) != len(mutable) or any(
            name not in source.parameters for name in mutable
        ):
            raise ResolutionError("mutable parameters must be unique DAG parameter names")
        mutable_set = set(mutable)
        mutable = tuple(name for name in source.parameters if name in mutable_set)
        if not isinstance(parameters, Mapping) or set(parameters) != set(source.parameters):
            raise ResolutionError("bindings must match the DAG parameter names exactly")
        raw_values: list[float] = []
        immutable: list[tuple[str, float]] = []
        for name in source.parameters:
            raw = _finite_number(parameters[name], f"parameter {name}")
            try:
                rounded = self.precision.round_real(raw, name=f"parameter {name}")
            except (ValueError, OverflowError) as exc:
                raise PrecisionResolutionError(str(exc)) from exc
            raw_values.append(raw)
            if name not in mutable_set:
                immutable.append((name, rounded))
        nodes = (_TargetExprNode * len(source.nodes))(*(
            _TargetExprNode(int(n.op), n.lhs, n.rhs, n.binding, n.value)
            for n in source.nodes
        ))
        bindings = (ct.c_double * len(raw_values))(*raw_values)
        fixed = (ct.c_uint8 * len(raw_values))(*(
            int(name not in mutable_set) for name in source.parameters
        ))
        values = (ct.c_double * len(source.nodes))()
        kinds = (ct.c_uint8 * len(source.nodes))()
        failed = ct.c_uint32(_UINT32_MAX)
        status = self._lib.lc_target_bind(
            nodes, len(nodes), bindings, len(bindings), fixed,
            len(source.variables), self.precision.real_bits,
            values, kinds, len(values), ct.byref(failed),
        )
        if status:
            location = "" if failed.value == _UINT32_MAX else f" at node {failed.value}"
            raise PrecisionResolutionError(
                f"{self.precision.value} target binding{location}: "
                + _FAILURES.get(status, f"unknown companion status {status}")
            )
        bound: list[TargetBoundValue] = []
        for kind, value in zip(kinds, values):
            try:
                binding_kind = TargetBindingKind(kind)
            except ValueError as exc:
                raise ResolutionError("companion returned an invalid binding kind") from exc
            if binding_kind is TargetBindingKind.BOUND and not math.isfinite(value):
                raise ResolutionError("companion returned a nonfinite bound value")
            bound.append(TargetBoundValue(
                binding_kind, value if binding_kind is TargetBindingKind.BOUND else None
            ))
        policy = numerical_policy(self.precision)
        identity = {
            "source_dag": _dag_document(source),
            "immutable_parameters": [(name, value.hex()) for name, value in immutable],
            "mutable_parameters": mutable,
            "precision": self.precision.to_record(),
            "policy": policy.to_record(),
            "companion_abi": _COMPANION_ABI,
            "binding_revision": _BINDING_REVISION,
            "supported_operations": _SUPPORTED,
            # Retained bits also distinguish host-specific signed-zero MAX ties.
            "bound_values": [
                [int(value.kind), None if value.value is None else value.value.hex()]
                for value in bound
            ],
        }
        key = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        return TargetBoundProgram(
            self.precision, policy, source, tuple(immutable), mutable, tuple(bound), key
        )


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ResolutionError(f"{context} must be a finite number")
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ResolutionError(f"{context} must be a finite number") from exc
    if not math.isfinite(number):
        raise PrecisionResolutionError(f"{context} must be finite")
    return number


def _uint32(value: int, context: str) -> int:
    if type(value) is not int or not 0 <= value <= _UINT32_MAX:
        raise ResolutionError(f"{context} must be an unsigned 32-bit integer")
    return value


def _snapshot_dag(dag: ExprDAG) -> ExprDAG:
    if not isinstance(dag, ExprDAG) or not dag.nodes or not dag.roots:
        raise ResolutionError("target binding requires a nonempty expression DAG")
    parameters, variables = tuple(dag.parameters), tuple(dag.variables)
    names = parameters + variables
    if any(not isinstance(name, str) or not name for name in names) or (
        len(set(names)) != len(names)
    ):
        raise ResolutionError("DAG parameter and variable names must be distinct strings")
    for count in (len(dag.nodes), len(parameters), len(variables)):
        _uint32(count, "DAG count")
    nodes: list[ExprNode] = []
    for index, node in enumerate(dag.nodes):
        if not isinstance(node, ExprNode):
            raise ResolutionError("DAG nodes must be ExprNode records")
        if not isinstance(node.op, (int, ExprOp)) or isinstance(node.op, bool):
            raise ResolutionError(f"unknown operation at node {index}")
        try:
            op = ExprOp(node.op)
        except ValueError as exc:
            raise ResolutionError(f"unknown operation at node {index}") from exc
        nodes.append(ExprNode(
            op, _uint32(node.lhs, "left reference"),
            _uint32(node.rhs, "right reference"), _uint32(node.binding, "binding"),
            _finite_number(node.value, f"node {index} value"),
        ))
    if not isinstance(dag.roots, Mapping):
        raise ResolutionError("DAG roots must map names to node indices")
    for name in dag.roots:
        if not isinstance(name, str) or not name:
            raise ResolutionError("DAG roots require nonempty string names")
    roots = {}
    for name, index in sorted(dag.roots.items()):
        if _uint32(index, "root index") >= len(nodes):
            raise ResolutionError("DAG root index is out of range")
        roots[name] = index
    return ExprDAG(tuple(nodes), parameters, variables, MappingProxyType(roots))


def _dag_document(dag: ExprDAG) -> dict[str, object]:
    return {
        "nodes": [
            [int(n.op), n.lhs, n.rhs, n.binding, n.value.hex()] for n in dag.nodes
        ],
        "parameters": list(dag.parameters),
        "variables": list(dag.variables),
        "roots": dict(dag.roots),
    }
