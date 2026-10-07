"""Target constant binding must not silently become target simulation."""

from dataclasses import FrozenInstanceError, fields
import importlib.util
import json
import math
from pathlib import Path

import pytest

from lacuna import (
    Engine, ExprDAG, ExprNode, ExprOp, LIF, PrecisionProfile,
    PrecisionResolutionError, TargetBindingEvaluator, TargetBindingKind,
    TargetBoundProgram, resolve_scalar_lif,
)
from lacuna.dsl import parse_neuron
from lacuna.errors import CapabilityError, ResolutionError
from lacuna.target_binding import _SUPPORTED

_LEGACY_PROFILES = (PrecisionProfile.FLOAT64, PrecisionProfile.FLOAT32_TIME64, PrecisionProfile.FLOAT32)


def test_half_primitive_companion_is_explicitly_unsupported():
    with pytest.raises(ResolutionError, match="does not support float16"):
        TargetBindingEvaluator(precision=PrecisionProfile.FLOAT16)

_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "binding_matrix", Path(__file__).with_name("test_execution_plan_matrix.py")
)
_MATRIX = importlib.util.module_from_spec(_MATRIX_SPEC)
_MATRIX_SPEC.loader.exec_module(_MATRIX)


def _dag(*nodes, parameters=(), variables=(), roots=None):
    return ExprDAG(tuple(nodes), parameters, variables, roots or {"value": len(nodes) - 1})


def _bind(dag, profile="float32", parameters=None, mutable=()):
    return TargetBindingEvaluator(precision=profile).bind(
        dag, {} if parameters is None else parameters, mutable_parameters=mutable
    )


@pytest.mark.parametrize("profile,expected", [
    ("float32", 0.0), ("float32-time64", 0.0), ("float64", 1.0),
])
def test_target_rounding_preserves_lowered_operation_order(profile, expected):
    dag = _dag(
        ExprNode(ExprOp.CONST, value=2.0 ** 24),
        ExprNode(ExprOp.CONST, value=1.0),
        ExprNode(ExprOp.ADD, lhs=0, rhs=1),
        ExprNode(ExprOp.SUB, lhs=2, rhs=0),
    )
    result = _bind(dag, profile)
    assert result.root_value("value") == expected
    assert result.source_dag.nodes == dag.nodes
    assert result.executable is False


@pytest.mark.parametrize("profile", _LEGACY_PROFILES)
def test_mutable_closure_is_never_bound_and_updates_do_not_change_key(profile):
    dag = _dag(
        ExprNode(ExprOp.PARAM, binding=0), ExprNode(ExprOp.PARAM, binding=1),
        ExprNode(ExprOp.ADD, lhs=0, rhs=1), ExprNode(ExprOp.VAR, binding=0),
        ExprNode(ExprOp.MUL, lhs=2, rhs=3),
        parameters=("gain", "drive"), variables=("state",),
        roots={"gain": 0, "drive": 1, "affine": 2, "state": 4},
    )
    first = _bind(dag, profile, {"gain": 0.1, "drive": 1.0}, ("drive",))
    second = _bind(dag, profile, {"gain": 0.1, "drive": 2.0}, ("drive",))
    assert first.binding_key == second.binding_key
    assert first.immutable_parameters == (("gain", profile.round_real(0.1)),)
    assert first.root_kind("gain") is TargetBindingKind.BOUND
    for name in ("drive", "affine", "state"):
        assert first.root_kind(name) is TargetBindingKind.DYNAMIC
        assert first.root_value(name) is None
    frozen = _bind(dag, profile, {"gain": 0.1, "drive": 1.0})
    assert frozen.binding_key != first.binding_key
    assert frozen.root_kind("affine") is TargetBindingKind.BOUND


@pytest.mark.parametrize("op", [
    ExprOp.POW, ExprOp.EXP, ExprOp.LOG, ExprOp.PHI1, ExprOp.PHI1_DERIV,
    ExprOp.SIN, ExprOp.COS, ExprOp.TANH,
])
def test_unreviewed_math_is_retained_even_when_constant(op):
    dag = _dag(
        ExprNode(ExprOp.CONST, value=-1.0),
        ExprNode(ExprOp.CONST, value=0.5),
        ExprNode(op, lhs=0, rhs=1),
        ExprNode(ExprOp.ADD, lhs=2, rhs=1),
    )
    result = _bind(dag)
    assert result.values[2].kind is TargetBindingKind.UNSUPPORTED
    assert result.root_kind("value") is TargetBindingKind.UNSUPPORTED
    assert result.root_value("value") is None
    assert result.source_dag.nodes[2].op is op


@pytest.mark.parametrize("profile", _LEGACY_PROFILES)
def test_signed_zero_and_subnormal_storage(profile):
    smallest = 2.0 ** (-149 if profile.real_bits == 32 else -1074)
    dag = _dag(
        ExprNode(ExprOp.CONST, value=-0.0),
        ExprNode(ExprOp.NEG, lhs=0),
        ExprNode(ExprOp.CONST, value=smallest),
        roots={"negative": 0, "positive": 1, "subnormal": 2},
    )
    result = _bind(dag, profile)
    assert result.root_value("negative").hex() == "-0x0.0p+0"
    assert result.root_value("positive").hex() == "0x0.0p+0"
    assert result.root_value("subnormal") == smallest


@pytest.mark.parametrize("nodes,message", [
    ((ExprNode(ExprOp.CONST, value=1e39),), "nonfinite"),
    ((ExprNode(ExprOp.CONST, value=1e-50),), "underflows"),
    ((ExprNode(ExprOp.CONST, value=3e38),
      ExprNode(ExprOp.CONST, value=2.0),
      ExprNode(ExprOp.MUL, lhs=0, rhs=1),
      ExprNode(ExprOp.DIV, lhs=2, rhs=1)), "node 2.*overflow"),
    ((ExprNode(ExprOp.CONST, value=2.0 ** -149),
      ExprNode(ExprOp.CONST, value=0.5),
      ExprNode(ExprOp.MUL, lhs=0, rhs=1)), "node 2.*underflows"),
    ((ExprNode(ExprOp.CONST, value=1.0),
      ExprNode(ExprOp.CONST, value=0.0),
      ExprNode(ExprOp.DIV, lhs=0, rhs=1)), "node 2.*division by zero"),
])
def test_primitive_failures_are_not_hidden_by_later_algebra(nodes, message):
    with pytest.raises(PrecisionResolutionError, match=message):
        _bind(_dag(*nodes))


def test_native_width_reciprocal_collision_is_visible():
    dag = _dag(
        ExprNode(ExprOp.CONST, value=-1.0),
        ExprNode(ExprOp.PARAM, binding=0), ExprNode(ExprOp.PARAM, binding=1),
        ExprNode(ExprOp.DIV, lhs=0, rhs=1), ExprNode(ExprOp.DIV, lhs=0, rhs=2),
        parameters=("tau_a", "tau_b"), roots={"a": 3, "b": 4},
    )
    parameters = {"tau_a": 24.00001335144043, "tau_b": 24.000015258789062}
    narrow = _bind(dag, "float32", parameters)
    wide = _bind(dag, "float64", parameters)
    assert narrow.root_value("a") == narrow.root_value("b")
    assert wide.root_value("a") != wide.root_value("b")


def test_artifact_snapshot_and_identity_are_immutable():
    roots = {"value": 0}
    dag = _dag(ExprNode(ExprOp.CONST, value=-0.0), roots=roots)
    programs = [_bind(dag, profile) for profile in _LEGACY_PROFILES]
    assert len({item.binding_key for item in programs}) == 3
    result = programs[0]
    roots["value"] = 100
    assert result.source_dag.roots["value"] == 0
    with pytest.raises(TypeError):
        result.source_dag.roots["value"] = 10
    with pytest.raises(FrozenInstanceError):
        result.executable = True
    assert result.supported_operations == _SUPPORTED
    document = json.loads(json.dumps(result.to_document()))
    assert document["values"][0]["value"] == "-0x0.0p+0"
    assert document["executable"] is False
    positive = _bind(_dag(ExprNode(ExprOp.CONST, value=0.0)), programs[0].precision)
    assert positive.binding_key != result.binding_key


@pytest.mark.parametrize("dag", [
    _dag(ExprNode(ExprOp.ADD, lhs=1, rhs=0)),
    _dag(ExprNode(ExprOp.VAR, binding=1), variables=("v",)),
    _dag(ExprNode(ExprOp.PARAM, binding=1), parameters=("p",)),
    _dag(ExprNode(ExprOp.CONST), roots={"value": 12}),
    _dag(ExprNode(ExprOp.CONST, lhs=-1)),
    _dag(ExprNode(ExprOp.CONST, binding=2 ** 32)),
    _dag(ExprNode(ExprOp.CONST), parameters=("p", "p")),
    _dag(ExprNode(ExprOp.CONST), parameters=("p",), variables=("p",)),
    _dag(ExprNode(0.0)),
    _dag(ExprNode(True)),
    _dag(ExprNode(ExprOp.CONST), roots={"a": 0, 1: 0}),
])
def test_invalid_descriptors_fail_before_they_can_be_used(dag):
    with pytest.raises(ResolutionError):
        _bind(dag, parameters={name: 1.0 for name in dag.parameters})


@pytest.mark.parametrize("bindings,mutable", [
    ({}, ()), ({"p": 1.0, "extra": 2.0}, ()),
    ({"p": 1.0}, ("unknown",)), ({"p": 1.0}, ("p", "p")),
    ({"p": float("nan")}, ()), ({"p": 1e-50}, ()),
    ({"p": 1.0}, None), ({"p": 1.0}, (["p"],)),
    ({"p": 1.0}, "p"),
])
def test_binding_contract_rejects_invalid_inputs(bindings, mutable):
    dag = _dag(ExprNode(ExprOp.PARAM), parameters=("p",))
    with pytest.raises(ResolutionError):
        _bind(dag, parameters=bindings, mutable=mutable)


def test_companion_is_optional_and_cannot_be_substituted_by_the_core(core, tmp_path):
    with pytest.raises(ResolutionError, match="host library"):
        TargetBindingEvaluator(tmp_path / "missing")
    with pytest.raises(ResolutionError, match="not a target-binding companion"):
        TargetBindingEvaluator(core._lib._name)


@pytest.mark.parametrize("mismatch", ["version", "size", "offset", "partial"])
def test_integer_handshake_precedes_precision_bearing_calls(monkeypatch, mismatch):
    from lacuna import target_binding

    class Query:
        def __init__(self, value):
            self.value = value

        def __call__(self, *args):
            return self.value(*args) if callable(self.value) else self.value

    class Library:
        lc_target_abi_version = Query(999 if mismatch == "version" else 1)
        lc_target_sizeof_expr_node = Query(999 if mismatch == "size" else 24)
        lc_target_expr_node_offset = Query(
            lambda field: 999 if mismatch == "offset" else (0, 4, 8, 12, 16)[field]
        )

        def __getattr__(self, name):
            if name == "lc_target_bind":
                if mismatch == "partial":
                    raise AttributeError(name)
                pytest.fail("typed binding was accessed before ABI validation")
            raise AttributeError(name)

    monkeypatch.setattr(target_binding.ct, "CDLL", lambda path: Library())
    with pytest.raises(ResolutionError, match="mismatch|incomplete"):
        TargetBindingEvaluator("fake")


def test_bound_artifacts_cannot_be_executed(core):
    result = _bind(_dag(ExprNode(ExprOp.CONST, value=1.0)))
    assert isinstance(result, TargetBoundProgram)
    with pytest.raises(TypeError, match="ExecutionPlan"):
        core.compile_execution_plan(result)
    with pytest.raises(ResolutionError, match="Network"):
        Engine(core._lib._name).compile(result)


@pytest.mark.parametrize("profile", _LEGACY_PROFILES)
@pytest.mark.parametrize("name,factory", _MATRIX.PLAN_CASES, ids=[
    name for name, _ in _MATRIX.PLAN_CASES
])
def test_existing_model_dags_bind_without_freezing_runtime_inputs(name, factory, profile):
    graph = factory()
    original = graph.to_text()
    evaluator = TargetBindingEvaluator(precision=profile)
    count = 0
    for model in graph.resolve().models:
        for attribute in fields(model):
            dag = getattr(model, attribute.name)
            if not isinstance(dag, ExprDAG):
                continue
            parameters = {key: model.bindings[key] for key in dag.parameters}
            mutable = tuple(key for key in model.drive_parameters if key in dag.parameters)
            result = evaluator.bind(dag, parameters, mutable_parameters=mutable)
            assert result.executable is False
            assert result.source_dag.nodes == dag.nodes
            for node, bound in zip(dag.nodes, result.values):
                if node.op is ExprOp.VAR or (
                    node.op is ExprOp.PARAM and dag.parameters[node.binding] in mutable
                ):
                    assert bound.kind is TargetBindingKind.DYNAMIC
            count += 1
    assert count > 0
    assert graph.to_text() == original


def test_scalar_coefficients_are_not_cast_from_host_resolved_fields():
    model = LIF(name="native-binding", drive=24.0, tau_m=20.0).model
    resolved = resolve_scalar_lif(parse_neuron(model.source))
    bound = _bind(
        resolved.binding_dag, parameters=resolved.bindings,
        mutable=resolved.drive_parameters,
    )
    # The current lowering uses POW for reciprocals, which is not certified here.
    assert bound.root_kind("a") is TargetBindingKind.UNSUPPORTED
    assert bound.root_value("a") is None
    assert bound.root_kind("b") is TargetBindingKind.DYNAMIC
    assert bound.root_value("reset") == resolved.reset
    assert math.isfinite(resolved.a)
