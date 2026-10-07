"""Uniform half precision and the host-only bit transport contract."""

import ctypes
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from lacuna import AugmentedState, CoreEvaluator, ExprDAG, ExprNode, ExprOp, Graph, LIF, PrecisionProfile
from lacuna.errors import CoreError, PrecisionResolutionError
from lacuna.ffi import _native_precision_info, _native_types
from lacuna.numerical_policy import NumericalStatus, numerical_policy
from lacuna.precision import normalize_precision, require_time_progress
from lacuna.target_lowering import _validate_half_powers, resolve_target_graph


def test_uniform_float16_identity_and_no_mixed_half_aliases():
    profile = PrecisionProfile.FLOAT16
    assert profile.native_id == 4
    assert profile.cache_key == ("float16", 16, 16, 1)
    assert PrecisionProfile.from_record(profile.to_record()) is profile
    assert normalize_precision("float16", "float16") is profile
    for clock in ("float32", "float64"):
        with pytest.raises(ValueError, match="float16"):
            normalize_precision(profile, clock)
    for model in ("float32", "float64", "float32-time64"):
        with pytest.raises(ValueError, match="float16"):
            normalize_precision(model, "float16")


@pytest.mark.parametrize("value", (0.0, -0.0, 0.1, 65504.0, 2.0 ** -24, 2049.0))
def test_half_representation_matches_ieee_binary16_bits(value):
    rounded = PrecisionProfile.FLOAT16.round_real(value)
    expected = struct.unpack("e", struct.pack("e", value))[0]
    assert rounded.hex() == expected.hex()
    assert PrecisionProfile.FLOAT16.round_time(value).hex() == expected.hex()


@pytest.mark.parametrize("value", (65520.0, 1e10, 2.0 ** -26, float("inf"), float("nan")))
def test_half_invalid_inputs_fail_before_bit_packing(value):
    types = _native_types(PrecisionProfile.FLOAT16)
    for convert in (types.real_value, types.time_value, types.real_argument.from_param,
                    lambda x: types._CInputSpike(0.0, 0, x),
                    lambda x: types.real_array(1)(x)):
        with pytest.raises(PrecisionResolutionError):
            convert(value)


def test_half_arrays_scalars_and_structures_keep_half_storage_and_float_views():
    types = _native_types(PrecisionProfile.FLOAT16)
    assert types.real_type is types.time_type is types.profile_type is ctypes.c_uint16
    values = types.real_array(3)(0.1, -0.0, 65504.0)
    assert ctypes.sizeof(values) == 6
    assert list(values) == values[:] == [0.0999755859375, -0.0, 65504.0]
    assert bytes(values) == struct.pack("=eee", 0.1, -0.0, 65504.0)
    raw = ctypes.cast(values, ctypes.POINTER(ctypes.c_uint16))
    raw[0] = 0x3C00
    assert values[0] == 1.0
    stamp = types.time_value(0.1)
    ctypes.cast(ctypes.byref(stamp), ctypes.POINTER(ctypes.c_uint16))[0] = 0x4000
    assert stamp.value == 2.0
    assert types.time_argument.from_param(stamp).value == 0x4000
    state = types._CState(0.1, 1.25)
    assert ctypes.sizeof(state) == 4
    assert state.value == 0.0999755859375
    assert state.t_last == 1.25
    trace = types._CTraceRecord()
    trace.before[0] = 0.1
    assert trace.before[0] == 0.0999755859375
    with pytest.raises(PrecisionResolutionError):
        trace.before[0] = 1e-30


def test_half_time_progress_and_uncertified_local_targets_are_explicit():
    with pytest.raises(ValueError, match="does not advance"):
        require_time_progress(2048.0, 1.0, "float16")
    policy = numerical_policy("float16")
    assert not policy.authorizes_execution
    assert set(policy.capabilities.to_record().values()) == {NumericalStatus.UNCERTIFIED.value}
    config = policy.step_defaults
    assert config.relative_tolerance == 8 * 2.0 ** -10
    assert config.absolute_tolerance == config.minimum_step == 2.0 ** -24
    assert config.event_tolerance == 2.0 ** -10
    assert policy.crossing_relative_tolerance == 2 * 2.0 ** -10


class _IntegerFunction:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


@pytest.mark.parametrize("field,bad", ((3, 32), (4, 32), (5, 24), (7, 128), (9, 32), (10, 24)))
def test_half_metadata_rejects_any_wider_native_type(field, bad):
    values = [1, 4, 16, 16, 11, 11, 16, 16, 16, 11, 16, 2, 1]
    values[field - 1] = bad
    class Library:
        lc_numeric_property = _IntegerFunction(lambda index: values[index - 1])
        lc_numeric_profile_check = _IntegerFunction(lambda *args: 0)
    with pytest.raises(CoreError):
        _native_precision_info(Library(), PrecisionProfile.FLOAT16)


@pytest.fixture
def half_core():
    if not tuple(Path("build-float16").glob("liblacuna_half_host.*")):
        pytest.skip("native half core and host transport bridge are not built")
    return CoreEvaluator(precision="float16")


def test_native_half_expression_has_per_operation_half_rounding(half_core):
    dag = ExprDAG((ExprNode(ExprOp.CONST, value=2048.0),
                   ExprNode(ExprOp.CONST, value=1.0),
                   ExprNode(ExprOp.ADD, lhs=0, rhs=1),
                   ExprNode(ExprOp.SUB, lhs=2, rhs=0)), (), (), {"difference": 3})
    assert half_core.abi_version == 19
    assert half_core.precision_info.wide_bits == 16
    assert half_core.evaluate_expr(dag) == {"difference": 0.0}
    assert CoreEvaluator().evaluate_expr(dag) == {"difference": 1.0}


@pytest.mark.parametrize("last,time", (
    (0.0, 1.0), (0.0001, 1.0), (2048.0, 2050.0), (0.0, 2.0 ** -24),
))
def test_half_selected_inspection_computes_elapsed_time_in_native_core(
    half_core, monkeypatch, last, time,
):
    neuron = LIF()
    model = resolve_target_graph(
        Graph(models=(neuron.model,), nodes=(neuron.node(0),)), half_core,
    ).models[0]
    state = AugmentedState((-60.0,), last)
    expected = half_core.advance_analytical(model, state, time)
    native_selected = half_core._lib.lc_expr_evaluate_selected
    calls = []

    def selected(*arguments):
        nodes, count, _, _, variables, variable_count = arguments[:6]
        status = native_selected(*arguments)
        calls.append((tuple(nodes[index].op for index in range(count)),
                      tuple(variables[index] for index in range(variable_count)),
                      tuple(arguments[8][index] for index in range(arguments[7]))))
        return status

    monkeypatch.setattr(half_core._lib, "lc_expr_evaluate_selected", selected)
    actual = half_core.advance_analytical_selected(model, state, time, (0,))
    assert actual == expected.values
    assert len(calls) == 2
    operations, operands, elapsed = calls[0]
    assert operations == (ExprOp.VAR, ExprOp.VAR, ExprOp.SUB)
    assert operands == (half_core.precision.round_time(time), half_core.precision.round_time(last))
    assert calls[1][1][0] == elapsed[0]
    if last == 0.0001:
        assert elapsed == (1.0,)


def test_half_selected_inspection_rejects_collapsed_clock_before_native_work(half_core, monkeypatch):
    neuron = LIF()
    model = resolve_target_graph(
        Graph(models=(neuron.model,), nodes=(neuron.node(0),)), half_core,
    ).models[0]
    monkeypatch.setattr(half_core._lib, "lc_expr_evaluate_selected",
                        lambda *args: pytest.fail("collapsed inspection reached evaluation"))
    with pytest.raises(PrecisionResolutionError, match="does not advance time"):
        half_core.advance_analytical_selected(model, AugmentedState((-60.0,), 2048.0), 2049.0, (0,))


def _power_dag(exponent_node):
    return ExprDAG((ExprNode(ExprOp.VAR, binding=0), exponent_node,
                   ExprNode(ExprOp.POW, lhs=0, rhs=1)),
                  ("exponent",), ("value",), {"power": 2})


@pytest.mark.parametrize("exponent", (-2.0, -1.0, 0.0, 0.5, *range(1, 9), -0.2, 0.2))
def test_half_fixed_supported_exponents_are_native_validated(half_core, exponent):
    dag = _power_dag(ExprNode(ExprOp.PARAM, binding=0))
    _validate_half_powers(dag, {"exponent": exponent}, half_core)
    result = half_core.evaluate_expr(dag, parameters={"exponent": exponent}, variables={"value": 1.0})
    assert result == {"power": 1.0}


@pytest.mark.parametrize("exponent", (0.25, 3.25, -3.0, 9.0))
def test_half_unsupported_static_exponents_are_rejected(half_core, exponent):
    dag = _power_dag(ExprNode(ExprOp.PARAM, binding=0))
    with pytest.raises(PrecisionResolutionError, match="unsupported fixed exponent"):
        _validate_half_powers(dag, {"exponent": exponent}, half_core)


def test_half_state_or_mutable_exponent_is_rejected_before_native_evaluation(half_core, monkeypatch):
    monkeypatch.setattr(half_core, "evaluate_expr", lambda *args, **kwargs: pytest.fail("dynamic exponent was evaluated"))
    for node, mutable in ((ExprNode(ExprOp.VAR, binding=0), ()),
                          (ExprNode(ExprOp.PARAM, binding=0), ("exponent",))):
        with pytest.raises(PrecisionResolutionError, match="parameter-only fixed exponent"):
            _validate_half_powers(_power_dag(node), {"exponent": 2.0}, half_core,
                                  mutable_parameters=mutable)


@pytest.mark.parametrize("order", (
    ("float64", "float32", "float32-time64", "float16"),
    ("float16", "float32-time64", "float32", "float64"),
    ("float32-time64", "float16", "float64", "float32"),
))
def test_half_host_dispatch_is_independent_of_library_load_order(half_core, order):
    code = """
import sys
from lacuna import CoreEvaluator, Presentation, RegularRateEncoder
cores = [CoreEvaluator(precision=profile) for profile in sys.argv[1:]]
for core in cores:
    with core.create_streaming_encoder_run((RegularRateEncoder(1.0, 1.0),)) as run:
        first = run.advance_until(2.0, presentations=(Presentation(0.0, 3.0, 0, 1.0),))
        final = run.finish(3.0)
        assert tuple(item.t for item in first.spikes + final.spikes) == (1.0, 2.0)
    assert core.precision.value in sys.argv[1:]
"""
    subprocess.run([sys.executable, "-c", code, *order], check=True, capture_output=True,
                   text=True, env={**os.environ, "PYTHONPATH": str(Path("src").resolve())})
